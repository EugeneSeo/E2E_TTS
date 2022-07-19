import functools
from numba import jit, prange

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import betabinom


class BetaBinomialInterpolator:
    """
        This module calculates alignment prior matrices (based on beta-binomial distribution) using cached popular sizes and image interpolation.
        The implementation is taken from https://github.com/NVIDIA/DeepLearningExamples.
    """

    def __init__(self, round_mel_len_to=50, round_text_len_to=10, cache_size=500):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(maxsize=cache_size)(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    x = np.arange(0, phoneme_count)
    mel_text_probs = []
    for i in range(1, mel_count + 1):
        a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
        mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def binarize_attention(attn, in_len, out_len):
    """Convert soft attention matrix to hard attention matrix.

    Args:
        attn (torch.Tensor): B x 1 x max_mel_len x max_text_len. Soft attention matrix.
        in_len (torch.Tensor): B. Lengths of texts.
        out_len (torch.Tensor): B. Lengths of spectrograms.

    Output:
        attn_out (torch.Tensor): B x 1 x max_mel_len x max_text_len. Hard attention matrix, final dim max_text_len should sum to 1.
    """
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = torch.zeros_like(attn)
        for ind in range(b_size):
            hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
            attn_out[ind, 0, : out_len[ind], : in_len[ind]] = torch.tensor(hard_attn, device=attn.device)
    return attn_out


def binarize_attention_parallel(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.get_device())


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf # start position
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out