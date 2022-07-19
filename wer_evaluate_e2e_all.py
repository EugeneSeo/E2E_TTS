from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer
import os
import argparse
import json
import tgt
import numpy as np
from dataloader_lin import prepare_dataloader, parse_batch

from models.StyleSpeech import *
from models.Hifigan import *
import utils

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

def create_wav(SS, G, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len):
    mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = SS( \
                text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
    wav_output = G(acoustic_adaptor_output, hidden_output).squeeze(1)
    return wav_output

def wer_eval(args, config=None):
    eval_loader = prepare_dataloader(args.data_path, "{}.txt".format(args.val_type), shuffle=False, batch_size=1, val=True) 
    
    #####################################
    generator = Generator_intpol_conv(config).cuda()
    stylespeech = StyleSpeech_attn(config).cuda()
    # stylespeech = StyleSpeech_transformer(config).cuda()
    
    cp_ss = os.path.join(args.checkpoint_path, 'ss_{}'.format(args.checkpoint_step))
    cp_g = os.path.join(args.checkpoint_path, 'g_{}'.format(args.checkpoint_step))

    utils.load_checkpoint(cp_ss, stylespeech, "stylespeech", 0)
    utils.load_checkpoint(cp_g, generator, "generator", 0)

    generator.eval()
    stylespeech.eval()
    #####################################

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    result = []
    for i, batch in enumerate(eval_loader):
        # parse batch
        sid, text, mel_target, spec_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
        
        if (generator != None) and (stylespeech != None): 
            wav = create_wav(stylespeech, generator, text, src_len, spec_target, mel_len, D, f0, energy, max_src_len, max_mel_len)

        # get gt text
        basename = batch["id"][0]
        real_sid, _, _, _ = basename.split("_")
        tg_path = os.path.join(args.data_path, "TextGrid", str(real_sid), "{}.TextGrid".format(basename))
        textgrid = tgt.io.read_textgrid(tg_path)
        text = get_alignment(textgrid.get_tier_by_name('words'))
        text = [' '.join(text).upper()]
        
        # ASR step: create transcription
        input_values = processor(wav[0], sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values.cuda()).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        result.append(wer(text, transcription))

    print(args.checkpoint_step, " - ", "WER:", np.average(result)*100)
    # WER: 4.9696267765553755 (gt)
    # WER: 6.233312314870909 (val.txt, no_finetuning)
    # WER: 6.1237605380231284 (val.txt, finetuning, 20220111_5, 00005000)
    # WER: 6.969311073233368 (unseen.txt, no_finetuning)
    # WER: 6.448718703380852 (unseen.txt, finetuning, 20220111_5, 00005000)
    return np.average(result)*100

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/aitrics_ext/ext01/kevin/dataset_en/LibriTTS_ss/preprocessed16/')
    parser.add_argument('--exp_code', default='default')
    parser.add_argument('--min_step', default=1)
    parser.add_argument('--max_step', default=27)
    parser.add_argument('--val_type', default='val') # val or unseen
    
    args = parser.parse_args()

    args.checkpoint_path = os.path.join("/mnt/aitrics_ext/ext01/eugene/Exp_results/", "cp_{}".format(args.exp_code))
    args.config = args.checkpoint_path + "/config.json"
    
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = utils.AttrDict(config)

    best_iter = 0
    best_wer = 100
    assert int(args.min_step) <= int(args.max_step)
    for i in range(int(args.min_step), int(args.max_step)+1):
        args.checkpoint_step = "00"
        if i < 100:
            args.checkpoint_step = "000"
        if i < 10:
            args.checkpoint_step = "0000"
        args.checkpoint_step = args.checkpoint_step + str(i * 1000)
        temp = wer_eval(args, config)
        if temp < best_wer:
            best_wer = temp
            best_iter = args.checkpoint_step

    print("==========================")
    print("best step: ", best_iter)
    print("best wer : ", best_wer)

if __name__=='__main__':
    main()