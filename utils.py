import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec

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

def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data)//2, 2, squeeze=False)
    # print("shape: ", data[0].shape[1])
    mel_max = max(data[0].shape[1], data[2].shape[1])
    wav_max = data[1].shape[1]
    gs = gridspec.GridSpec(nrows = len(data)//2, ncols = 2, height_ratios = [1, 1], width_ratios = [mel_max, wav_max])

    fig.tight_layout()
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)//2): # 0: Synthesized, 1: GT
        for j in range(2): # 0: Mel, 1: Random window Audio
            ax = plt.subplot(gs[i*2+j])
            spectrogram = data[i*2+j]
            # axes[i][j].imshow(spectrogram, origin='lower')
            # axes[i][j].set_aspect(2.5, adjustable='box')
            # axes[i][j].set_ylim(0, 80)
            # axes[i][j].set_title(titles[i*2+j], fontsize='medium')
            # axes[i][j].tick_params(labelsize='x-small', left=False, labelleft=False) 
            # axes[i][j].set_anchor('W')
            ax.imshow(spectrogram, origin='lower')
            ax.set_aspect(2.5, adjustable='box')
            ax.set_ylim(0, 80)
            ax.set_title(titles[i*2+j], fontsize='medium')
            ax.tick_params(labelsize='x-small', left=False, labelleft=False) 
            ax.set_anchor('W')
    
    plt.savefig(filename, dpi=200)
    plt.close()

def get_mask_from_lengths(lengths, device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    # print("mask shape: ", mask.shape, ", lengths: ", lengths)
    return mask