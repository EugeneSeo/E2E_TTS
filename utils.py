import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


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

def get_mask_from_lengths(lengths, device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    # print("mask shape: ", mask.shape, ", lengths: ", lengths)
    return mask