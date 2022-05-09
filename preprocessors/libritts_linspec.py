from text import _clean_text
import numpy as np
import librosa
import os
import torch
from pathlib import Path
from scipy.io.wavfile import write
from joblib import Parallel, delayed
import tgt
import pyworld as pw
from preprocessors.utils import remove_outlier, get_alignment, average_by_duration
from scipy.interpolate import interp1d
import json


hann_window = {}

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    return spec

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config["sampling_rate"]

        self.n_fft = config["n_fft"]
        self.hop_length = config["hop_size"]
        self.win_length = config["win_size"]

        self.max_seq_len = config["max_seq_len"]

    def build_from_path(self, data_dir, out_dir):
        with open(os.path.join(out_dir, 'metadata.csv'), encoding='utf-8') as f:
            basenames = []
            for line in f:
                parts = line.strip().split('|')
                basename = parts[0]
                basenames.append(basename)

        data = Parallel(n_jobs=10, verbose=1)(delayed(self.process_utterance)(data_dir, out_dir, basename) for basename in basenames)
        print(data[0])
        return 


    def process_utterance(self, in_dir, out_dir, basename, dataset='libritts'):
        sid = basename.split('_')[0]
        wav_path = os.path.join(in_dir, 'wav{}'.format(self.sampling_rate//1000), sid, '{}.wav'.format(basename))
        tg_path = os.path.join(out_dir, 'TextGrid', sid, '{}.TextGrid'.format(basename)) 

        if not os.path.exists(wav_path) or not os.path.exists(tg_path):
            # return None
            return "1: " + wav_path + ", " + tg_path
        
        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        _, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'), self.sampling_rate, self.hop_length)

        if start >= end:
            # return None
            return "2"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=None)
        wav = wav[int(self.sampling_rate*start):int(self.sampling_rate*end)].astype(np.float32)
        # return str(wav.shape)
        wav = torch.from_numpy(wav).unsqueeze(0)

        # Compute linear spectrogram and energy
        lin_spectrogram = spectrogram_torch(wav, self.n_fft, self.sampling_rate, self.hop_length, self.win_length)
        lin_spectrogram = lin_spectrogram.squeeze(0)
        # return str(lin_spectrogram.shape)
        lin_spectrogram = lin_spectrogram[:, :sum(duration)]
        
        if lin_spectrogram.shape[1] >= self.max_seq_len:
            # return None
            return "3"
        
        # Save spectrogram
        spec_filename = '{}-spec-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'spectrogram', spec_filename), lin_spectrogram.T, allow_pickle=False)

        return None