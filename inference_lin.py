###### HiFi-GAN ######
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from models.Hifigan import *
from models.StyleSpeech import StyleSpeech
from dataloader import prepare_dataloader, parse_batch
import utils

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def create_lin(config, batch):
    # device =  torch.device('cuda:{:d}'.format(0))

    sid, text, mel_target, mel_start_idx, wav, \
                D, log_D, f0, energy, \
                src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
    lin = utils.lin_spectrogram(wav, config.n_fft, config.sampling_rate, config.hop_size, config.win_size)
    mel = utils.mel_spectrogram(wav, config.n_fft, config.n_mel_channels, config.sampling_rate, config.hop_size, config.win_size,
                                          config.fmin, config.fmax_for_loss)
    # print(mel.shape, ", ", lin.shape)
    

def inference_f(args, config, max_inf=None):
    validation_loader = prepare_dataloader(args.data_path, "val.txt", shuffle=False, batch_size=1, val=True) 
    count = 0

    for j, batch in enumerate(validation_loader):
        if (count == max_inf):
            break
        basename = batch["id"][0]
        create_lin(config, batch)
        count += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='test')
    parser.add_argument('--checkpoint_path', default='cp_20220315')
    parser.add_argument('--checkpoint_step', default='00020000')
    # parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    # parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--max_inf', default=10, type=int)

    args = parser.parse_args()

    # args.config = "./" + args.checkpoint_path + "/config.json"
    args.config = "./configs/config.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = utils.AttrDict(config)

    # gpu_ids = [0]
    # h.num_gpus = len(gpu_ids)

    inference_f(args, config, args.max_inf)

if __name__ == '__main__':
    main()
