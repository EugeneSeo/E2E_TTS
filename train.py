###### HiFi-GAN ######
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from hifi_gan.env import AttrDict, build_env
# from hifi_gan.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from hifi_gan.meldataset import mel_spectrogram
from hifi_gan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from hifi_gan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
torch.backends.cudnn.benchmark = True
##### StyleSpeech #####
from StyleSpeech.models.StyleSpeech import StyleSpeech
from StyleSpeech.models.Loss import StyleSpeechLoss
from StyleSpeech.optimizer import ScheduledOptim
from StyleSpeech.evaluate import evaluate
import StyleSpeech.utils as utils_ss
torch.backends.cudnn.enabled = True
##### E2E_TTS #####
from model import D_step, G_step, SS_step, parse_batch
from dataloader import prepare_dataloader
from torch.cuda.amp import autocast, GradScaler
#--------------------------------------------------------------------#

def train(rank, a, h, c, gpu_ids):
    # # if h.num_gpus > 1:
    # if len(gpu_ids) > 1:
    #     init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
    #                        world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    # torch.cuda.manual_seed(h.seed)
    # device = torch.device('cuda:{:d}'.format(rank))
    device = torch.device('cuda:{:d}'.format(gpu_ids[0]))

    # Added ------------------------------------------------------ #
    print("Defining the Model")
    # model = E2E_TTS(h, c, device).to(device)
    # model = torch.nn.DataParallel(model, gpu_ids)

    generator = Generator(h).to(device)
    stylespeech = StyleSpeech(c).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    generator = torch.nn.DataParallel(generator, gpu_ids)
    # stylespeech = torch.nn.DataParallel(stylespeech, gpu_ids)
    mpd = torch.nn.DataParallel(mpd, gpu_ids)
    msd = torch.nn.DataParallel(msd, gpu_ids)
    print("Model Defined\n")
    # -------------------------------------------------------------- #
    # generator = Generator(h).to(device)
    # mpd = MultiPeriodDiscriminator().to(device)
    # msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_ss = scan_checkpoint(a.checkpoint_path, 'ss_')
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    # Add cp_ss & loading code
    steps = 0
    if cp_g is None or (cp_do is None or cp_ss is None):
        state_dict_do = None
        # load dongchan's pre-trained model!: Should unblock this code block b4 running
        # checkpoint_dict_ss = load_checkpoint(checkpoint_path_ss, device)
        # stylespeech.load_state_dict(checkpoint_dict_ss['model'])
        # checkpoint_dict_g = load_checkpoint(checkpoint_path_g, device)
        # generator.load_state_dict(checkpoint_dict_g['generator'])
        last_epoch = -1
    else:
        state_dict_ss = load_checkpoint(cp_ss, device)
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        stylespeech.load_state_dict(state_dict_ss['stylespeech'])
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # # if h.num_gpus > 1:
    # if len(gpu_ids) > 1:
    #     # model.generator = DistributedDataParallel(model.generator, device_ids=gpu_ids).to(device)
    #     # trainer.mpd = DistributedDataParallel(trainer.mpd, device_ids=gpu_ids).to(device)
    #     # trainer.msd = DistributedDataParallel(trainer.msd, device_ids=gpu_ids).to(device)
    #     model = DistributedDataParallel(model, device_ids=[gpu_ids[0]]).to(device)
    #     trainer = DistributedDataParallel(trainer, device_ids=[gpu_ids[0]]).to(device)

    print("Defining Optimizer and Loss Function")
    # Add ss's optimizer - done
    # optim_g = torch.optim.AdamW(model.generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    # optim_d = torch.optim.AdamW(itertools.chain(trainer.msd.parameters(), trainer.mpd.parameters()),
    #                             h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    # Added ------------------------------------------------------ #
    optim_ss = torch.optim.Adam(stylespeech.parameters(), betas=c.betas, eps=c.eps)
    loss_ss = StyleSpeechLoss()
    print("Optimizer and Loss Function Defined.")
    # -------------------------------------------------------------- #

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        optim_ss.load_state_dict(state_dict_do['optim_ss'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    # Added & Changed ------------------------------------------------------ #
    scheduled_optim = ScheduledOptim(optim_ss, c.decoder_hidden, c.n_warm_up_step, steps)

    train_loader = prepare_dataloader(a.data_path, "train.txt", shuffle=True, batch_size=c.batch_size) 
    if rank == 0:        
        validation_loader = prepare_dataloader(a.data_path, "val.txt", shuffle=True, batch_size=c.batch_size) 
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    print("Data Loader is Prepared.")
    # model.train()
    stylespeech.train()
    generator.train()
    mpd.train()
    msd.train()
    # -------------------------------------------------------------- #

    # AutoCast #
    scaler = GradScaler()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        # if h.num_gpus > 1:
        #     train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            # Added ------------------------------------------------------ #
            print("\n---Start One Epoch Training---\n")
            # if (i == 1):
            #     break
            # ------------------------------------------------------ #
            if rank == 0:
                start_b = time.time()
            
            # Changed ------------------------------------------------------ #
            # sid, text, mel_target, mel_start_idx, wav, \
            #         D, log_D, f0, energy, \
            #         src_len, mel_len, max_src_len, max_mel_len = model.parse_batch(batch)
            # wav_output, mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _  = model(
            #         text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len, mel_start_idx)
            sid, text, mel_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
            # print("mel_len shape: ", mel_len.shape)
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = stylespeech(
                    device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            
            indices = [[mel_start_idx[i]+j for j in range(32)] for i in range(c.batch_size)]
            # print(indices)
            indices = torch.Tensor(indices).type(torch.int64)
            indices = torch.unsqueeze(indices, 2).expand(-1, -1, 80).to(device)
            # print(indices.shape)
            # wav_output = generator(mel_output[:, mel_start_idx : mel_start_idx+32])
            wav_output = generator(torch.transpose(torch.gather(mel_output, 1, indices), 1, 2))
            
            wav_output_mel = mel_spectrogram(wav_output.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)
            # -------------------------------------------------------------- #
            # print("wav shape: ", wav.shape)
            # print("wav_output shape: ", wav_output.shape)
            # wav_crop = wav[:, mel_start_idx * h.hop_size:(mel_start_idx + 32) * h.hop_size]
            wav_crop = torch.unsqueeze(wav, 1)
            # mel_crop = mel_target[:, mel_start_idx:mel_start_idx + 32]
            mel_crop = torch.transpose(torch.gather(mel_target, 1, indices), 1, 2)
            print("Optimizing Step\n")

            # GAN D&G step (optimize) ------------------------------------------------------ #
            loss_disc_all = D_step(mpd, msd, optim_d, wav_crop, wav_output.detach(), scaler)
            loss_gen_all = G_step(mpd, msd, optim_g, wav_crop, mel_crop, wav_output, wav_output_mel, scaler)

            # StyleSpeech optimize ------------------------------------------------------ #
            scheduled_optim.zero_grad()
            loss_ss_all = SS_step(loss_ss, optim_ss, mel_output.detach(), mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len, scaler)
            # Clipping gradients to avoid gradient explosion
            scaler.unscale_(scheduled_optim._optimizer)
            torch.nn.utils.clip_grad_norm_(stylespeech.parameters(), c.grad_clip_thresh)
            # Update weights
            scheduled_optim.step_and_update_lr(scaler=scaler)
            # scheduled_optim.step(scaler=scaler)
            scaler.update()
            # scheduled_optim.update_lr()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(mel_crop, wav_output_mel).item()

                    print('Steps : {:d}, SS Loss Total : {:4.3f}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_ss_all.item(), loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    print('Checkpointing')
                    checkpoint_path = "{}/ss_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'stylespeech': (stylespeech.module if h.num_gpus > 1 else stylespeech.generator).state_dict()})
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 
                                     'optim_ss': optim_ss.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    print('Tensorboard summary logging')
                    sw.add_scalar("training/loss_ss_all", loss_ss_all, steps)
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    print('Validation')
                    stylespeech.eval()
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            sid, text, mel_target, mel_start_idx, wav, \
                                    D, log_D, f0, energy, \
                                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
                            
                            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = stylespeech(
                                    device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
                            
                            indices = [[mel_start_idx[i]+j for j in range(32)] for i in range(c.batch_size)]
                            indices = torch.Tensor(indices).type(torch.int64)
                            indices = torch.unsqueeze(indices, 2).expand(-1, -1, 80).to(device)
                            wav_output = generator(torch.transpose(torch.gather(mel_output, 1, indices), 1, 2))
                            wav_output_mel = mel_spectrogram(wav_output.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                                        h.fmin, h.fmax_for_loss)
                            mel_crop = torch.transpose(torch.gather(mel_target, 1, indices), 1, 2)
                            
                            val_err_tot += F.l1_loss(mel_crop, wav_output_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), wav[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(mel_crop[0].cpu()), steps)
                                    # sw.add_text('gt/y_txt_{}'.format(j), text[0], steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), wav_output[0], steps, h.sampling_rate)
                                wav_output_spec = mel_spectrogram(wav_output[0].squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(wav_output_spec.squeeze(0).cpu().numpy()), steps)
                                # sw.add_text('generated/y_hat_txt_{}'.format(j), text[0], steps)

                            # new
                            break
                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    stylespeech.train()
                    generator.train()
                    # model.train()
                
            steps += 1
            break

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    # # parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    # # parser.add_argument('--input_mels_dir', default='ft_dataset')
    # # parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    # # parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    # # Changed ------------------------------------------------------ #
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    # parser.add_argument('--input_wavs_dir', default='/v9/dongchan/TTS/dataset/LibriTTS/wav22')
    # parser.add_argument('--input_mels_dir', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed_22/mel')
    # parser.add_argument('--input_training_file', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed_22/train.txt')
    # parser.add_argument('--input_validation_file', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed_22/val.txt')
    # # -------------------------------------------------------------- #
    parser.add_argument('--checkpoint_path', default='cp_E2E_TTS')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    # Added
    parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    # Added ------------------------------------------------------ #
    with open(a.config_ss) as f_ss:
        data_ss = f_ss.read()
    json_config_ss = json.loads(data_ss)
    config = utils_ss.AttrDict(json_config_ss)
    utils_ss.build_env(a.config_ss, 'config_ss.json', a.checkpoint_path)
    # -------------------------------------------------------------- #

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        # print('Batch size per GPU :', h.batch_size)
        print('Batch size per GPU :', config.batch_size)
    else:
        pass

    # if h.num_gpus > 1:
    #     mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    # else:
    #     train(0, a, h)
    # Changed ------------------------------------------------------ #
    # if h.num_gpus > 1:
    #     mp.spawn(train, nprocs=h.num_gpus, args=(a, h, config,))
    # else:
    #     train(0, a, h, config)
    gpu_ids = [4, 5, 6]
    train(0, a, h, config, gpu_ids)
    # -------------------------------------------------------------- #

if __name__ == '__main__':
    main()
