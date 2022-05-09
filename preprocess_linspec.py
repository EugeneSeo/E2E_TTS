import os
import argparse
import random
import preprocessors.libritts_linspec as libritts_linspec
import json
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5,6,7"

def make_folders(out_dir):
    spec_out_dir = os.path.join(out_dir, "spectrogram")
    if not os.path.exists(spec_out_dir):
        os.makedirs(spec_out_dir, exist_ok=True)


def main(data_dir, out_dir, config):
    preprocessor = libritts_linspec.Preprocessor(config)
    make_folders(out_dir)
    preprocessor.build_from_path(data_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/v9/dongchan/TTS/dataset/LibriTTS/')
    parser.add_argument('--output_path', type=str, default="/v9/dongchan/TTS/dataset/LibriTTS/preprocessed/")
    parser.add_argument('--config', default='./configs/config.json') # Configurations for Speech model
    
    args = parser.parse_args()

    with open(args.config) as f_ss:
        data = f_ss.read()
    config = json.loads(data)
    # config = utils_ss.AttrDict(json_config_ss)

    main(args.data_path, args.output_path, config)
