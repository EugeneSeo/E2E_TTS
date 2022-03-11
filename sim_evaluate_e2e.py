import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy import spatial
import glob
import os
import math
import sys
from pathlib import Path

import matplotlib.pyplot as p
import argparse
########################################################
import argparse
import json
import tgt
import torch
import numpy as np
from dataloader import prepare_dataloader
from model import parse_batch

import StyleSpeech.utils as utils_ss
from StyleSpeech.models.StyleSpeech import StyleSpeech
from hifi_gan.env import AttrDict
from hifi_gan.utils import load_checkpoint
from hifi_gan.models import Generator_interpolation
########################################################
sampling_rate = 16000

def create_wav(SS, G, device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len):
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = SS(
                device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(acoustic_adaptor_output, hidden_output).squeeze(1)
    return wav_output

def sim_eval(a, gpu_ids, h, c):
    device = torch.device('cuda:{:d}'.format(gpu_ids[0]))
    eval_loader = prepare_dataloader(a.data_path, "{}.txt".format(a.val_type), shuffle=False, batch_size=1, val=True) 
    
    #####################################
    generator = Generator_interpolation(h).to(device)
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
    #####################################

    encoder = VoiceEncoder()

    result = []
    for i, batch in enumerate(eval_loader):
        # parse batch
        sid, text, mel_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
        
        #####################################
        with torch.no_grad():
            wav_pred = create_wav(stylespeech, generator, device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            # wav_pred = generator(torch.transpose(mel_target, 1, 2)).squeeze(1)
        #####################################
      
        wav = wav.cpu().numpy()
        wav_pred = wav_pred.cpu().numpy()
        gt_embed = encoder.embed_utterance(wav[0])
        pred_embed = encoder.embed_utterance(wav_pred[0])

        sim = 1 - spatial.distance.cosine(gt_embed, pred_embed)        
        result.append(sim)

    print("SIM:", np.average(result))
    # SIM: 0.8410108296713041 (val.txt, gt vs. no_finetuning)
    # SIM: 0.877558888464737 (val.txt, gt vs. finetuning, 20220111_5, 00005000)
    # SIM: 0.8302679061629367 (unseen.txt, gt vs. no_finetuning)
    # SIM: 0.8699259363928216 (unseen.txt, gt vs. finetuning, 20220111_5, 00005000)
    # SIM: 0.9368604759644675 (val.txt, gt vs. gt -> hifi-gan)
    # SIM: 0.937864129329873 (unseen.txt, gt vs. gt -> hifi-gan)


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--checkpoint_path', default='cp_default')
    parser.add_argument('--checkpoint_step', default='00005000')
    # parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--val_type', default='val') # val or unseen
    parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    
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

    sim_eval(a, gpu_ids, h, config)

if __name__=='__main__':
    main()

    