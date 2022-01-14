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
from hifi_gan.env import AttrDict, build_env
from hifi_gan.meldataset import mel_spectrogram
from hifi_gan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from hifi_gan.utils import load_checkpoint
torch.backends.cudnn.benchmark = True
##### StyleSpeech #####
from StyleSpeech.models.StyleSpeech import StyleSpeech
import StyleSpeech.utils as utils_ss
torch.backends.cudnn.enabled = True
##### E2E_TTS #####
from model import parse_batch
from dataloader import prepare_dataloader
from utils import plot_data

def create_wav(a, h, c, G, SS, batch, exp_code, basename):
    device =  torch.device('cuda:{:d}'.format(0))

    sid, text, mel_target, mel_start_idx, wav, \
                D, log_D, f0, energy, \
                src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = SS(
                device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(torch.transpose(mel_output, 1, 2))
    wav_output_mel = mel_spectrogram(wav_output.squeeze(1), h.n_fft, h.num_mels, c.sampling_rate, h.hop_size, h.win_size,
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

    plot_data([mel.numpy(), wav_mel.numpy(), mel_target.numpy(), wav_target_mel.numpy()], 
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
    generator = Generator(h).to(device)
    stylespeech = StyleSpeech(c).to(device)

    generator = torch.nn.DataParallel(generator, [0])
    cp_ss = os.path.join(a.checkpoint_path, 'ss_{}'.format(a.checkpoint_step))
    cp_g = os.path.join(a.checkpoint_path, 'g_{}'.format(a.checkpoint_step))

    state_dict_ss = load_checkpoint(cp_ss, device)
    state_dict_g = load_checkpoint(cp_g, device)
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

def inference(a, h, c, max_inf=None):
    validation_loader = prepare_dataloader(a.data_path, "val.txt", shuffle=False, batch_size=1, val=True) 
    count = 0

    os.makedirs(a.save_path, exist_ok=True)
    device =  torch.device('cuda:{:d}'.format(0))
    generator = Generator(h).to(device)
    stylespeech = StyleSpeech(c).to(device)

    stylespeech.load_state_dict(torch.load("./cp_StyleSpeech/stylespeech.pth.tar")['model'])
    state_dict_g = load_checkpoint("./cp_hifigan/g_02500000", device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    stylespeech.eval()

    exp_code = "no_finetuning"
    a.checkpoint_step = "0"
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
    parser.add_argument('--checkpoint_path', default='cp_20220111_2')
    parser.add_argument('--checkpoint_step', default='00003000')
    parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--max_inf', default=10, type=int)
    parser.add_argument('--finetuning', default=True, type=bool)

    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    with open(a.config_ss) as f_ss:
        data_ss = f_ss.read()
    json_config_ss = json.loads(data_ss)
    config = utils_ss.AttrDict(json_config_ss)

    gpu_ids = [0]
    h.num_gpus = len(gpu_ids)
    if a.finetuning:
        inference_f(a, h, config, a.max_inf)
    else:
        inference(a, h, config, a.max_inf)

if __name__ == '__main__':
    main()
