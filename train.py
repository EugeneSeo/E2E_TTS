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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from hifi_gan.env import AttrDict, build_env
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
from model import D_step, G_step, SS_step, parse_batch, WBLogger
from dataloader import prepare_dataloader
from torch.cuda.amp import autocast, GradScaler
from utils import plot_data
#--------------------------------------------------------------------#

def train(rank, a, h, c, gpu_ids):
    # torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(gpu_ids[0]))
    print("Start training\n")

    # Define model
    generator = Generator(h).to(device)
    stylespeech = StyleSpeech(c).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    loss_ss = StyleSpeechLoss()

    if a.freeze_ss:
        for param in stylespeech.parameters():
            param.requires_grad = False
    
    num_param = utils_ss.get_param_num(generator) + utils_ss.get_param_num(stylespeech) \
                + utils_ss.get_param_num(msd) + utils_ss.get_param_num(mpd)
    print("Model Defined\n")

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_ss = scan_checkpoint(a.checkpoint_path, 'ss_')
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    # Add cp_ss & loading code
    steps = 0
    if cp_g is None or (cp_do is None or cp_ss is None):
    # if True:
        state_dict_do = None
        stylespeech.load_state_dict(torch.load("./cp_StyleSpeech/stylespeech.pth.tar")['model'])
        state_dict_g = load_checkpoint("./cp_hifigan/g_02500000", device)
        generator.load_state_dict(state_dict_g['generator'])
        state_dict_do_ = load_checkpoint("./cp_hifigan/do_02500000", device)
        mpd.load_state_dict(state_dict_do_['mpd'])
        msd.load_state_dict(state_dict_do_['msd'])
        last_epoch = -1

        generator = torch.nn.DataParallel(generator, gpu_ids)
        # stylespeech = torch.nn.DataParallel(stylespeech, gpu_ids)
        mpd = torch.nn.DataParallel(mpd, gpu_ids)
        msd = torch.nn.DataParallel(msd, gpu_ids)
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

        generator = torch.nn.DataParallel(generator, gpu_ids)
        # stylespeech = torch.nn.DataParallel(stylespeech, gpu_ids)
        mpd = torch.nn.DataParallel(mpd, gpu_ids)
        msd = torch.nn.DataParallel(msd, gpu_ids)

    # Optimizers
    if (a.optim_g == "G_only"):
        optim_g = torch.optim.AdamW(generator.parameters(), a.lr_g, betas=[h.adam_b1, h.adam_b2])
        optim_ss = torch.optim.Adam(stylespeech.parameters(), a.lr_ss, betas=c.betas, eps=c.eps)
    else: # a.optim_g = "G_and_SS"
        optim_g = torch.optim.AdamW(itertools.chain(generator.parameters(), stylespeech.parameters()), a.lr_g, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                a.lr_d, betas=[h.adam_b1, h.adam_b2])
    print("Optimizer and Loss Function Defined.")

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        optim_ss.load_state_dict(state_dict_do['optim_ss'])
    else:
        if (a.optim_g == "G_only"):
            optim_g.load_state_dict(state_dict_do_['optim_g'])
        optim_d.load_state_dict(state_dict_do_['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    if (a.optim_g == "G_only"):
        scheduled_optim = ScheduledOptim(optim_ss, c.decoder_hidden, c.n_warm_up_step, steps)

    train_loader = prepare_dataloader(a.data_path, "train.txt", shuffle=True, batch_size=c.batch_size) 
    if rank == 0:        
        validation_loader = prepare_dataloader(a.data_path, "val.txt", shuffle=True, batch_size=1, val=True) 
        sw = SummaryWriter(os.path.join(a.save_path, 'logs'))
        # Init logger
        log_path = os.path.join(a.save_path, 'log.txt')
        with open(log_path, "a") as f_log:
            f_log.write("Dataset :{}\n Number of Parameters: {}\n".format(c.dataset, num_param))

    # Init synthesis directory
    synth_path = os.path.join(a.save_path, 'synth')
    os.makedirs(synth_path, exist_ok=True)

    print("Data Loader is Prepared.")
    
    stylespeech.train()
    generator.train()
    mpd.train()
    msd.train()
    # -------------------------------------------------------------- #

    # AutoCast #
    if a.use_scaler:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        
        for i, batch in enumerate(train_loader):
            # print("\n---Start One Epoch Training---\n")
            
            if rank == 0:
                start_b = time.time()
            
            # Get Data
            sid, text, mel_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch, device)
            
            # Forwards
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = stylespeech(
                    device, text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            indices = [[mel_start_idx[i]+j for j in range(32)] for i in range(c.batch_size)]
            indices = torch.Tensor(indices).type(torch.int64)
            indices = torch.unsqueeze(indices, 2).expand(-1, -1, 80).to(device)
            wav_output = generator(torch.transpose(torch.gather(mel_output, 1, indices), 1, 2))
            wav_output_mel = mel_spectrogram(wav_output.squeeze(1), h.n_fft, h.num_mels, c.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)
            
            wav_crop = torch.unsqueeze(wav, 1)
            mel_crop = torch.transpose(torch.gather(mel_target, 1, indices), 1, 2)
            # mel_crop = torch.transpose(torch.gather(mel_output, 1, indices), 1, 2)

            # Optimizing Step
            # GAN D step
            mpd.requires_grad_(True)
            msd.requires_grad_(True)
            generator.requires_grad_(False)
            loss_disc_all = D_step(mpd, msd, optim_d, wav_crop, wav_output.detach())

            # GAN G & SS step
            mpd.requires_grad_(False)
            msd.requires_grad_(False)
            generator.requires_grad_(True)
            if (a.optim_g == "G_and_SS"):
                loss_gen_all = G_step(mpd, msd, optim_g, wav_crop, mel_crop, wav_output, wav_output_mel, 
                                    loss_ss, mel_output, mel_target, 
                                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len,
                                    scaler=scaler, retain_graph=False, G_only=False, lmel_hifi=a.lmel_hifi, lmel_ss=a.lmel_ss)
                if scaler != None:
                    scaler.update()
            else: 
                loss_gen_all = G_step(mpd, msd, optim_g, wav_crop, mel_crop, wav_output, wav_output_mel, 
                                    loss_ss, mel_output, mel_target, 
                                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len,
                                    scaler=scaler, retain_graph=True, G_only=True, lmel_hifi=a.lmel_hifi, lmel_ss=a.lmel_ss)
                # StyleSpeech optimize
                scheduled_optim.zero_grad()
                loss_ss_all = SS_step(loss_ss, mel_output.detach(), mel_target, 
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len, scaler=scaler)
                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(stylespeech.parameters(), c.grad_clip_thresh)
                    scheduled_optim.step_and_update_lr(scaler=None)
                else:
                    scaler.unscale_(scheduled_optim._optimizer)
                    torch.nn.utils.clip_grad_norm_(stylespeech.parameters(), c.grad_clip_thresh)
                    scheduled_optim.step_and_update_lr(scaler=None)
                    # scheduled_optim.step(scaler=scaler)
                    scaler.update()
                    # scheduled_optim.update_lr()
            
            if rank == 0:
                # STDOUT & log.txt logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(mel_crop, wav_output_mel).item()

                    if (a.optim_g == "G_only"):
                        str1 = 'Steps : {:d}, SS Loss Total : {:4.3f}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                                    steps, loss_ss_all.item(), loss_gen_all, mel_error, time.time() - start_b)
                    else:
                        str1 = 'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                                    steps, loss_gen_all, mel_error, time.time() - start_b)
                    str2 = str1 + "\n"
                    with open(log_path, "a") as f_log:
                        f_log.write(str2)

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    print('Checkpointing')
                    checkpoint_path = "{}/ss_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'stylespeech': stylespeech.state_dict()})
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_dict = {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 
                                     'steps': steps, 'epoch': epoch}
                    if (a.optim_g == "G_only"):
                        save_dict['optim_ss'] = optim_ss.state_dict()
                    save_checkpoint(checkpoint_path, save_dict)
                    

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    print('Tensorboard summary logging')
                    if (a.optim_g == "G_only"):
                        sw.add_scalar("training/loss_ss_all", loss_ss_all, steps)
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0: # and steps != 0:
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
                            
                            wav_output = generator(torch.transpose(mel_output, 1, 2))
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
                            plot_data([mel.numpy(), wav_mel.numpy(), mel_target.numpy(), wav_target_mel.numpy()], 
                                ['Synthesized Spectrogram', 'Swav', 'Ground-Truth Spectrogram', 'GTwav'], 
                                filename=os.path.join(synth_path, 'step_{}.jpg'.format(steps)))
                            print("Synth spectrograms at step {}...\n".format(steps))
                            wav_output_val_path = os.path.join(synth_path, 'step_{}_synth.wav'.format(steps))
                            wav_val_path = os.path.join(synth_path, 'step_{}_gt.wav'.format(steps))
                            sf.write(wav_output_val_path, wav_output.squeeze(1)[0].cpu(), c.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
                            sf.write(wav_val_path, wav[0].cpu(), c.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
                            break

                    stylespeech.train()
                    generator.train()
                   
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='exp_default')
    parser.add_argument('--checkpoint_path', default='cp_default')
    parser.add_argument('--config', default='./hifi_gan/config_v1.json')
    parser.add_argument('--training_epochs', default=1, type=int)
    parser.add_argument('--stdout_interval', default=100, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    
    parser.add_argument('--config_ss', default='./StyleSpeech/configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--optim_g', default='G_and_SS') # "G_and_SS" or "G_only"
    parser.add_argument('--use_scaler', default=False, type=bool)
    parser.add_argument('--freeze_ss', default=False, type=bool)

    parser.add_argument('--lmel_hifi', default=45, type=int)
    parser.add_argument('--lmel_ss', default=1, type=int)
    parser.add_argument('--lr_g', default=0.0002, type=float)
    parser.add_argument('--lr_d', default=0.0002, type=float)
    parser.add_argument('--lr_ss', default=0.001, type=float)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    with open(a.config_ss) as f_ss:
        data_ss = f_ss.read()
    json_config_ss = json.loads(data_ss)
    config = utils_ss.AttrDict(json_config_ss)
    utils_ss.build_env(a.config_ss, 'config_ss.json', a.checkpoint_path)
    
    gpu_ids = [0]
    h.num_gpus = len(gpu_ids)
    train(0, a, h, config, gpu_ids)

if __name__ == '__main__':
    main()
