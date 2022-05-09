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


def create_wav(args, config, G, SS, batch, exp_code, basename):
    # device =  torch.device('cuda:{:d}'.format(0))

    sid, text, mel_target, mel_start_idx, wav, \
                D, log_D, f0, energy, \
                src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
    
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = SS(
                text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(torch.transpose(mel_output, 1, 2))
    wav_output_mel = utils.mel_spectrogram(wav_output.squeeze(1), config.n_fft, config.n_mel_channels, config.sampling_rate, config.hop_size, config.win_size,
                                                        config.fmin, config.fmax_for_loss)
    mel_crop = torch.transpose(mel_target, 1, 2)
    wav_crop = torch.unsqueeze(wav, 1)
    length = mel_len[0].item()
    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
    wav_target_mel = mel_crop[0].detach().cpu()
    wav_mel = wav_output_mel[0].detach().cpu()
    
    # plotting
    synth_path = os.path.join(args.save_path, basename)
    os.makedirs(synth_path, exist_ok=True)

    utils.plot_data([mel.numpy(), wav_mel.numpy(), mel_target.numpy(), wav_target_mel.numpy()], 
        ['Synthesized Spectrogram', 'Swav', 'Ground-Truth Spectrogram', 'GTwav'], 
        filename=os.path.join(synth_path, 'exp_{0}_{1}.jpg'.format(exp_code, args.checkpoint_step)))
    wav_output_val_path = os.path.join(synth_path, 'exp_{0}_{1}.wav'.format(exp_code, args.checkpoint_step))
    wav_val_path = os.path.join(synth_path, 'exp_gt.wav'.format(exp_code))
    sf.write(wav_output_val_path, wav_output.squeeze(1)[0].detach().cpu(), config.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
    if not (os.path.exists(wav_val_path)):
        sf.write(wav_val_path, wav[0].detach().cpu(), config.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
    
def inference_f(args, config, max_inf=None):
    validation_loader = prepare_dataloader(args.data_path, "val.txt", shuffle=False, batch_size=1, val=True) 
    count = 0

    os.makedirs(args.save_path, exist_ok=True)
    
    with open("./configs/config_hifi.json") as f:
        data = f.read()
    hifi_config = json.loads(data)
    hifi_config = utils.AttrDict(hifi_config)
    generator = Generator(hifi_config).cuda()
    stylespeech = StyleSpeech(config).cuda()

    cp_ss = os.path.join(args.checkpoint_path, 'ss_{}'.format(args.checkpoint_step))
    cp_g = "./cp_hifigan/g_02500000"

    utils.load_checkpoint(cp_ss, stylespeech, "stylespeech", 0)
    utils.load_checkpoint(cp_g, generator, "generator", 0)
    
    generator.eval()
    stylespeech.eval()

    exp_code = args.checkpoint_path[3:]
    for j, batch in enumerate(validation_loader):
        if (count == max_inf):
            break
        basename = batch["id"][0]
        create_wav(args, config, generator, stylespeech, batch, exp_code, basename)
        count += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='test_mel2wav')
    parser.add_argument('--checkpoint_path', default='cp_20220315')
    parser.add_argument('--checkpoint_step', default='00020000')
    # parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    # parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--max_inf', default=10, type=int)

    args = parser.parse_args()

    args.config = "./" + args.checkpoint_path + "/config.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = utils.AttrDict(config)

    # gpu_ids = [0]
    # h.num_gpus = len(gpu_ids)

    inference_f(args, config, args.max_inf)

if __name__ == '__main__':
    main()
