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


def create_wav(a, h, c, G, SS, batch, exp_code, basename):
    device =  torch.device('cuda:{:d}'.format(0))

    sid, text, mel_target, mel_start_idx, wav, \
                D, log_D, f0, energy, \
                src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
    
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = SS(
                device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(acoustic_adaptor_output, hidden_output)
    wav_output_mel = utils.mel_spectrogram(wav_output.squeeze(1), h.n_fft, h.num_mels, c.sampling_rate, h.hop_size, h.win_size,
                                                        h.fmin, h.fmax_for_loss)
    mel_crop = torch.transpose(mel_target, 1, 2)
    wav_crop = torch.unsqueeze(wav, 1)
    length = mel_len[0].item()
    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
    wav_target_mel = mel_crop[0].detach().cpu()
    wav_mel = wav_output_mel[0].detach().cpu()
    
    # plotting
    synth_path = os.path.join(a.save_path, basename)
    os.makedirs(synth_path, exist_ok=True)

    utils.plot_data([mel.numpy(), wav_mel.numpy(), mel_target.numpy(), wav_target_mel.numpy()], 
        ['Synthesized Spectrogram', 'Swav', 'Ground-Truth Spectrogram', 'GTwav'], 
        filename=os.path.join(synth_path, 'exp_{0}_{1}.jpg'.format(exp_code, a.checkpoint_step)))
    wav_output_val_path = os.path.join(synth_path, 'exp_{0}_{1}.wav'.format(exp_code, a.checkpoint_step))
    wav_val_path = os.path.join(synth_path, 'exp_gt.wav'.format(exp_code))
    sf.write(wav_output_val_path, wav_output.squeeze(1)[0].detach().cpu(), c.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
    if not (os.path.exists(wav_val_path)):
        sf.write(wav_val_path, wav[0].detach().cpu(), c.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
    
def inference_f(a, h, c, max_inf=None):
    validation_loader = prepare_dataloader(a.data_path, "val.txt", shuffle=False, batch_size=1, val=True) 
    count = 0

    os.makedirs(a.save_path, exist_ok=True)
    device =  torch.device('cuda:{:d}'.format(0))
    generator = Generator_intpol4(h).to(device)
    stylespeech = StyleSpeech(c).to(device)

    generator = torch.nn.DataParallel(generator, [0])
    cp_ss = os.path.join(a.checkpoint_path, 'ss_{}'.format(a.checkpoint_step))
    cp_g = os.path.join(a.checkpoint_path, 'g_{}'.format(a.checkpoint_step))

    state_dict_ss = utils.load_checkpoint(cp_ss, device)
    state_dict_g = utils.load_checkpoint(cp_g, device)
    stylespeech.load_state_dict(state_dict_ss['stylespeech'])
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    stylespeech.eval()

    exp_code = a.checkpoint_path[3:]
    for j, batch in enumerate(validation_loader):
        if (count == max_inf):
            break
        basename = batch["id"][0]
        create_wav(a, h, c, generator, stylespeech, batch, exp_code, basename)
        count += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='test')
    parser.add_argument('--checkpoint_path', default='cp_20220111_5')
    parser.add_argument('--checkpoint_step', default='00005000')
    parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--max_inf', default=10, type=int)

    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = utils.AttrDict(json_config)

    with open(a.config_ss) as f_ss:
        data_ss = f_ss.read()
    json_config_ss = json.loads(data_ss)
    config = utils.AttrDict(json_config_ss)

    gpu_ids = [0]
    h.num_gpus = len(gpu_ids)

    inference_f(a, h, config, a.max_inf)

if __name__ == '__main__':
    main()
