import os
import argparse
import random
import StyleSpeech.preprocessors.ljspeech as ljspeech
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,6,7"

def make_train_files(out_dir, datas):
    random.shuffle(datas)
    num_train = int(len(datas)*0.95)
    train_set = datas[:num_train]
    val_set = datas[num_train:]
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train_set:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val_set:
            f.write(m + '\n')


def make_folders(out_dir):
    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)


def main(data_dir, out_dir, config):
    preprocessor = ljspeech.Preprocessor(config)
    make_folders(out_dir)
    # preprocessor.write_metadata(data_dir, out_dir)
    datas = preprocessor.build_from_path(data_dir, out_dir)
    make_train_files(out_dir, datas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='dataset/')
    # parser.add_argument('--output_path', type=str, default='dataset/')
    parser.add_argument('--data_path', type=str, default='/v9/dongchan/TTS/dataset/LJSpeech/')
    parser.add_argument('--output_path', type=str, default='/v9/dongchan/TTS/dataset/LJSpeech/preprocessed/')
    parser.add_argument('--config_ss', default='./config.json') # Configurations for Speech model
    
    args = parser.parse_args()

    with open(args.config_ss) as f_ss:
        data_ss = f_ss.read()
    config = json.loads(data_ss)
    # config = utils_ss.AttrDict(json_config_ss)

    main(args.data_path, args.output_path, config)
