import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

from models.Hifigan import MultiPeriodDiscriminator_dwt as MPD, MultiScaleDiscriminator_dwt as MSD, Generator_intpol
from models.StyleSpeech import StyleSpeech_attn
from models.Loss import StyleSpeechLoss as StyleSpeechLoss, CVCLoss
from models.Optimizer import *

from dataloader_lin import prepare_dataloader, parse_batch
import utils


def cleanup():
    dist.destroy_process_group()

def load_checkpoint(checkpoint_path, model, name, rank, distributed=False):
    assert os.path.isfile(checkpoint_path)
    print("Starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(rank))
    if name in checkpoint_dict:
        # if distributed:
        if False:
            state_dict = {}
            for k,v in checkpoint_dict[name].items():
                state_dict['module.{}'.format(k)] = v
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint_dict[name])
        model.load_state_dict(checkpoint_dict[name])
        print('Model is loaded!') 
    return model

def train(rank, args, config, gpu_ids):

    print('Use GPU: {} for training'.format(rank))
    ngpus = args.ngpus
    if args.distributed:
        torch.cuda.set_device(rank % ngpus)
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url, world_size=config.world_size, rank=rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Define model
    generator = Generator_intpol(config).cuda()
    stylespeech = StyleSpeech_attn(config).cuda()
    mpd = MPD().cuda()
    msd = MSD().cuda()
    loss_ss = StyleSpeechLoss()
    loss_cvc = CVCLoss()
    
    num_param = utils.get_param_num(generator) + utils.get_param_num(stylespeech) \
                + utils.get_param_num(msd) + utils.get_param_num(mpd)

    if rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", args.checkpoint_path)
        print('Number of E2E StyleSpeech Parameters:', num_param)
        print("Model Has Been Defined")

    if os.path.isdir(args.checkpoint_path):
        cp_ss = utils.scan_checkpoint(args.checkpoint_path, 'ss_')
        cp_g = utils.scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = utils.scan_checkpoint(args.checkpoint_path, 'do_')

    generator_without_ddp = generator
    stylespeech_without_ddp = stylespeech
    mpd_without_ddp = mpd
    msd_without_ddp = msd

    # Add cp_ss & loading code
    steps = 0
    if cp_g is None or (cp_do is None or cp_ss is None):
        state_dict_do = None
        last_epoch = -1
    else:
        load_checkpoint(cp_ss, stylespeech, "stylespeech", rank, args.distributed)
        load_checkpoint(cp_g, generator, "generator", rank, args.distributed)
        load_checkpoint(cp_do, mpd, "mpd", rank, args.distributed)
        load_checkpoint(cp_do, msd, "msd", rank, args.distributed)

        state_dict_do = torch.load(cp_do, map_location='cuda:{}'.format(rank))
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if args.distributed:
        config.batch_size = config.batch_size // ngpus
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank])
        generator_without_ddp = generator.module
        stylespeech = torch.nn.parallel.DistributedDataParallel(stylespeech, device_ids=[rank])
        stylespeech_without_ddp = stylespeech.module
        mpd = torch.nn.parallel.DistributedDataParallel(mpd, device_ids=[rank])
        mpd_without_ddp = mpd.module
        msd = torch.nn.parallel.DistributedDataParallel(msd, device_ids=[rank])
        msd_without_ddp = msd.module

    # Optimizers
    optim_g = torch.optim.AdamW(itertools.chain(generator.parameters(), stylespeech.parameters()), config.lr_g, betas=[config.adam_b1, config.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                config.lr_d, betas=[config.adam_b1, config.adam_b2])
    print("Optimizer and Loss Function Defined.")

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay, last_epoch=last_epoch)
    
    train_loader = prepare_dataloader(args.data_path, "train.txt", shuffle=True, batch_size=config.batch_size) 
    if rank == 0:        
        validation_loader = prepare_dataloader(args.data_path, "val.txt", shuffle=True, batch_size=1, val=True) 
        log_path = os.path.join(args.checkpoint_path, 'log.txt')
        with open(log_path, "a") as f_log:
            f_log.write("Dataset :{}\n Number of Parameters: {}\n".format(config.dataset, num_param))

    # Init synthesis directory
    synth_path = os.path.join(args.save_path, 'synth')
    os.makedirs(synth_path, exist_ok=True)

    print("Data Loader is Prepared.")
    
    stylespeech.train()
    generator.train()
    mpd.train()
    msd.train()
    # -------------------------------------------------------------- #

    for epoch in range(max(0, last_epoch), args.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        
        for i, batch in enumerate(train_loader):
            # print("\n---Start One Epoch Training---\n")
            
            if rank == 0:
                start_b = time.time()
            
            # Get Data
            sid, text, mel_target, spec_target, mel_start_idx, wav, \
                    D, log_D, f0, energy, \
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
            
            # Forwards
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = stylespeech(
                    text, src_len, spec_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            indices = [[mel_start_idx[i]+j for j in range(32)] for i in range(config.batch_size)]
            indices = torch.Tensor(indices).type(torch.int64)
            indices_hidden = torch.unsqueeze(indices, 2).expand(-1, -1, config.encoder_hidden).cuda()

            wav_output = generator(acoustic_adaptor_output, hidden_output, indices=indices_hidden, mel_start_idx=mel_start_idx)
            wav_output_spec = utils.lin_spectrogram(wav_output.squeeze(1), config.n_fft, config.sampling_rate, config.hop_size, config.win_size)
            wav_output_mel = utils.mel_spectrogram(wav_output.squeeze(1), config.n_fft, 80, config.sampling_rate, config.hop_size, config.win_size,
                                                        config.fmin, config.fmax_for_loss)
            wav_crop = torch.unsqueeze(wav, 1)

            indices_spec = torch.unsqueeze(indices, 2).expand(-1, -1, config.n_mel_channels).cuda()
            indices_mel = torch.unsqueeze(indices, 2).expand(-1, -1, 80).cuda()

            spec_crop = torch.transpose(torch.gather(spec_target, 1, indices_spec), 1, 2)
            mel_crop = torch.transpose(torch.gather(mel_target, 1, indices_mel), 1, 2)

            # Optimizing Step
            # GAN D step
            mpd.requires_grad_(True)
            msd.requires_grad_(True)
            generator.requires_grad_(False)
            loss_disc_all, loss_disc_list = D_step(mpd, msd, optim_d, wav_crop, wav_output.detach())

            # GAN G & SS step
            mpd.requires_grad_(False)
            msd.requires_grad_(False)
            generator.requires_grad_(True)
            loss_gen_all, loss_gen_list = G_step(mpd, msd, loss_cvc, optim_g, wav_crop, spec_crop, wav_output, wav_output_spec,
                                            loss_ss, mel_output, spec_target, 
                                            log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len,
                                            retain_graph=False, G_only=False, lmel_hifi=config.lmel_hifi, lmel_ss=config.lmel_ss)
            
            if rank == 0:
                # STDOUT & log.txt logging
                if steps % args.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(spec_crop, wav_output_spec).item()

                    str1 = 'Steps: {:d}, G Loss: {:4.3f} ({:4.3f} + {:4.3f} + {:4.3f} + {:4.3f} + {:4.3f} + {:4.3f}), D Loss: {:4.3f} ({:4.3f} + {:4.3f} + {:4.3f} + {:4.3f})'.format(
                                steps, loss_gen_list[0], loss_gen_list[1], loss_gen_list[2],
                                loss_gen_list[3], loss_gen_list[4], loss_gen_list[5],
                                loss_gen_list[6], 
                                loss_disc_list[0], loss_disc_list[1], loss_disc_list[2], 
                                loss_disc_list[3], loss_disc_list[4])
                    str2 = str1 + "\n"
                    with open(log_path, "a") as f_log:
                        f_log.write(str2)

                # checkpointing
                if steps % args.checkpoint_interval == 0 and steps != 0:
                    print('Checkpointing')
                    checkpoint_path = "{}/ss_{:08d}".format(args.checkpoint_path, steps)
                    utils.save_checkpoint(checkpoint_path,
                                    {'stylespeech': stylespeech_without_ddp.state_dict()})
                    checkpoint_path = "{}/g_{:08d}".format(args.checkpoint_path, steps)
                    utils.save_checkpoint(checkpoint_path,
                                    {'generator': generator_without_ddp.state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(args.checkpoint_path, steps)
                    save_dict = {'mpd': mpd_without_ddp.state_dict(),
                                     'msd': msd_without_ddp.state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 
                                     'steps': steps, 'epoch': epoch}
                    utils.save_checkpoint(checkpoint_path, save_dict)

                # Validation
                if steps % args.validation_interval == 0: # and steps != 0:
                    print('Validation')
                    stylespeech.eval()
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            sid, text, mel_target, spec_target, mel_start_idx, wav, \
                                    D, log_D, f0, energy, \
                                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
                            
                            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = stylespeech(
                                    text, src_len, spec_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
                            
                            wav_output = generator(acoustic_adaptor_output, hidden_output)
                            wav_output_mel = utils.mel_spectrogram(wav_output.squeeze(1), config.n_fft, 80, config.sampling_rate, config.hop_size, config.win_size,
                                                        config.fmin, config.fmax_for_loss)
                            mel_crop = torch.transpose(mel_target, 1, 2)
                            spec_crop = torch.transpose(spec_target, 1, 2)
                            
                            length = mel_len[0].item()
                            
                            mel = utils.lin_to_mel(mel_output[0, :length].transpose(0,1).unsqueeze(0), config.n_fft, 80, config.sampling_rate, config.hop_size, config.win_size,
                                                        config.fmin, config.fmax_for_loss)
                            mel = mel.squeeze().detach().cpu()
                            wav_target_mel = utils.lin_to_mel(spec_crop[0].unsqueeze(0), config.n_fft, 80, config.sampling_rate, config.hop_size, config.win_size,
                                                        config.fmin, config.fmax_for_loss)
                            wav_target_mel = wav_target_mel.squeeze().detach().cpu()
                            
                            mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                            
                            wav_mel = wav_output_mel[0].detach().cpu()
                            # plotting
                            utils.plot_data([mel.numpy(), wav_mel.numpy(), mel_target.numpy(), wav_target_mel.numpy()], 
                                ['Synthesized Spectrogram', 'Swav', 'Ground-Truth Spectrogram', 'GTwav'], 
                                filename=os.path.join(synth_path, 'step_{}.jpg'.format(steps)))
                            print("Synth spectrograms at step {}...\n".format(steps))
                            wav_output_val_path = os.path.join(synth_path, 'step_{}_synth.wav'.format(steps))
                            wav_val_path = os.path.join(synth_path, 'step_{}_gt.wav'.format(steps))
                            sf.write(wav_output_val_path, wav_output.squeeze(1)[0].cpu(), config.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
                            sf.write(wav_val_path, wav[0].cpu(), config.sampling_rate, format='WAV', endian='LITTLE', subtype='PCM_16')
                            break

                    stylespeech.train()
                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    cleanup()
    return

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/mnt/aitrics_ext/ext01/kevin/dataset_en/LibriTTS_ss/preprocessed16/')
    parser.add_argument('--exp_code', default='default')
    parser.add_argument('--training_epochs', default=1, type=int)
    parser.add_argument('--stdout_interval', default=100, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--config', default='./configs/config_lin.json') # Configurations for StyleSpeech model

    args = parser.parse_args()
    args.save_path = os.path.join("/mnt/aitrics_ext/ext01/eugene/Exp_results/", "exp_{}".format(args.exp_code))
    args.checkpoint_path = os.path.join("/mnt/aitrics_ext/ext01/eugene/Exp_results/", "cp_{}".format(args.exp_code))

    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = utils.AttrDict(config)
    if not os.path.exists('{}/config.json'.format(args.checkpoint_path)):
        utils.build_env(args.config, 'config.json', args.checkpoint_path)
    
    gpu_ids = [0,1]
    ngpus = len(gpu_ids)
    args.ngpus = ngpus
    args.distributed = ngpus > 1

    if args.distributed:
        args.world_size = ngpus
        mp.spawn(train, nprocs=ngpus, args=(args, config, gpu_ids))
    else:
        train(0, args, config, gpu_ids)

if __name__ == '__main__':
    main()
