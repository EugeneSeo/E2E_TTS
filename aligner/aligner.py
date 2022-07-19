# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn
from aligner.aligner_losses import ForwardSumLoss, BinLoss
# from aligner.utils import get_mask_from_lengths, binarize_attention, binarize_attention_parallel
from aligner.utils import binarize_attention, binarize_attention_parallel
from utils import get_mask_from_lengths



class AlignerModel(nn.Module):
    """Speech-to-text alignment model (https://arxiv.org/pdf/2108.10447.pdf) that is used to learn alignments between mel spectrogram and text."""
    def __init__(self, n_mel_channels=80, n_text_channels=512, n_att_channels=128, temperature=0.0005):
        
        # TODO : build encoders (q_encs for mel_spec & k_encs for text)
        super(AlignerModel, self).__init__()
        '''
        self.q_encs = nn.Conv1d(n_mel_channels, n_att_channels)
        self.k_encs = nn.Conv1d(n_text_channels, n_att_channels)
        '''
        # reference: https://github.com/NVIDIA/NeMo/blob/fa2e55e607f787d129141523c724b29dab7a411c/nemo/collections/tts/modules/aligner.py        
        # self.k_encs = nn.Sequential(
        #     nn.Conv1d(n_text_channels, n_text_channels * 2, kernel_size=3, padding=1, bias=True),
        #     torch.nn.ReLU(),
        #     nn.Conv1d(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        # )
        self.k_encs = nn.Sequential(
            nn.Conv1d(n_text_channels, n_att_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(n_att_channels, n_att_channels, kernel_size=1, bias=True),
        )
        self.q_encs = nn.Sequential(
            nn.Conv1d(n_mel_channels, n_mel_channels * 2, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(n_mel_channels * 2, n_mel_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )        

        self.temperature = temperature
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.add_bin_loss = True
        self.bin_loss_scale = 1.0

    def forward(self, spec, spec_len, text, text_len, mask, attn_prior=None):
        """Forward pass of the aligner encoder.

        Args:
            spec (torch.tensor): B x C1 x T1 tensor.
            spec_len (torch.tensor): B. tensor.
            text (torch.tensor): B x C2 x T2 tensor.
            text_len (torch.tensor): B. tensor.
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries [False, False, ..., True, True].
            attn_prior (torch.tensor): prior for attention matrix.
        Output:
            attn_soft (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """

        with torch.cuda.amp.autocast(enabled=False):
            queries = self.q_encs(torch.transpose(spec, 1, 2)) # B x n_attn_dims x T1 
            keys = self.k_encs(torch.transpose(text, 1, 2)) # B x n_attn_dims x T2
            
            # Simplistic Gaussian Isotopic Attention
            attn = (queries[:, :, :, None] - keys[:, :, None]) ** 2 # B x n_attn_dims x T1 x T2
            attn = -self.temperature * attn.sum(1, keepdim=True)  # B x 1 x T1 x T2

            if mask is not None:
                attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -np.inf)
            
            attn_logprob = self.log_softmax(attn)
            if attn_prior is not None:
                attn_logprob = attn_logprob + torch.log(attn_prior[:, None] + 1e-8)
            
            attn = torch.exp(attn_logprob)

        return attn, attn_logprob

    def metrics(self, attn_soft, attn_logprob, spec_len, text_len):
        loss, bin_loss, attn_hard = 0.0, None, None

        forward_sum_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=text_len, out_lens=spec_len)
        loss += forward_sum_loss

        if self.add_bin_loss:
            attn_hard = binarize_attention(attn_soft, text_len, spec_len)
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            loss += bin_loss * self.bin_loss_scale

        return loss, forward_sum_loss, bin_loss, attn_hard

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.

        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.
        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and return mean distance.

        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.
        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        if mask is not None:
            dist = dist.masked_fill(mask.permute(0, 2, 1).unsqueeze(2), 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(torch.arange(t2_size), repeats=durations[dist_idx]),
                    ]
                )
            )

        return torch.tensor(mean_dist_by_durations, dtype=dist.dtype, device=dist.device)

        

      
