from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer
########################################################
import os
import argparse
import json
import tgt
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

def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn', '']
    phones = []
    start_time = 0
    end_idx = 0
    for t in tier._objects:
        s, p = t.start_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_idx = len(phones)
        else:
            phones.append(p)

    # Trimming tailing silences  
    return phones[:end_idx]

def create_wav(SS, G, device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len):
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = SS(
                device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(acoustic_adaptor_output, hidden_output).squeeze(1)
    return wav_output

def wer_eval(a, gpu_ids, h=None, c=None):
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

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    result = []
    for i, batch in enumerate(eval_loader):
        # parse batch
        sid, text, mel_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
        
        #####################################
        if (generator != None) and (stylespeech != None): 
            wav = create_wav(stylespeech, generator, device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
        #####################################

        # get gt text
        basename = batch["id"][0]
        real_sid, _, _, _ = basename.split("_")
        tg_path = os.path.join(a.data_path, "TextGrid", str(real_sid), "{}.TextGrid".format(basename))
        textgrid = tgt.io.read_textgrid(tg_path)
        text = get_alignment(textgrid.get_tier_by_name('words'))
        text = [' '.join(text).upper()]
        
        # ASR step: create transcription
        input_values = processor(wav[0], sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        result.append(wer(text, transcription))

    print("WER:", np.average(result)*100)
    # WER: 4.9696267765553755 (gt)
    # WER: 6.233312314870909 (val.txt, no_finetuning)
    # WER: 6.1237605380231284 (val.txt, finrtuning, 20220111_5, 00005000)
    # WER: 6.969311073233368 (unseen.txt, no_finetuning)
    # WER: 6.448718703380852 (unseen.txt, finrtuning, 20220111_5, 00005000)

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--checkpoint_path', default='cp_default')
    parser.add_argument('--checkpoint_step', default='00005000')
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

    # wer_eval(a, gpu_ids)
    wer_eval(a, gpu_ids, h, config)

if __name__=='__main__':
    main()