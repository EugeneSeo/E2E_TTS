import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec
import glob


def process_meta(data_path, meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        sid = []
        for line in f.readlines():
            n, t, s = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
            sid.append(s)
        return name, text, sid


def process_meta_ljspeech(data_path, meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        phone = []
        name = []
        for line in f.readlines():
            n, p = line.strip('\n').split('|')
            name.append(n)
            phone.append(p)
        return name, phone


def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data)//2, 2, squeeze=False)
    mel_max = max(data[0].shape[1], data[2].shape[1])
    wav_max = max(data[1].shape[1], data[3].shape[1])
    gs = gridspec.GridSpec(nrows = len(data)//2, ncols = 2, height_ratios = [1, 1], width_ratios = [mel_max, wav_max])

    fig.tight_layout()
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)//2): # 0: Synthesized, 1: GT
        for j in range(2): # 0: Mel, 1: Random window Audio
            ax = plt.subplot(gs[i*2+j])
            spectrogram = data[i*2+j]
            ax.imshow(spectrogram, origin='lower')
            ax.set_aspect(2.5, adjustable='box')
            ax.set_ylim(0, 80)
            ax.set_title(titles[i*2+j], fontsize='medium')
            ax.tick_params(labelsize='x-small', left=False, labelleft=False) 
            ax.set_anchor('W')
    
    plt.savefig(filename, dpi=200)
    plt.close()


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, t_path)


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)