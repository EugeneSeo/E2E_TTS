import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import soundfile as sf
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

from models.Hifigan import *
from models.StyleSpeech import StyleSpeech
from models.Loss import StyleSpeechLoss2 as StyleSpeechLoss, CVCLoss
# from models.Optimizer import ScheduledOptim, D_step, G_step, SS_step
from models.Optimizer import *
# from models.Feature import *

from dataloader import prepare_dataloader, parse_batch
from torch.cuda.amp import autocast, GradScaler
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
    # generator = Generator_intpol4(config).cuda()
    generator = Generator_FastSpeech2s(config).cuda()
    stylespeech = StyleSpeech(config).cuda()
    mpd = MultiPeriodDiscriminator().cuda()
    msd = MultiScaleDiscriminator().cuda()
    loss_ss = StyleSpeechLoss()
    loss_cvc = CVCLoss()
    # feature = PatchSampleF()
    
    if args.freeze_ss:
        for param in stylespeech.parameters():
            param.requires_grad = False
    
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
    # if True:
        state_dict_do = None
        # stylespeech.load_state_dict(torch.load("./cp_StyleSpeech/stylespeech.pth.tar")['model'])
        # # state_dict_g = load_checkpoint("./cp_hifigan/g_02500000", device)
        # # generator.load_state_dict(state_dict_g['generator'])
        # state_dict_do_ = load_checkpoint("./cp_hifigan/do_02500000", device)
        # mpd.load_state_dict(state_dict_do_['mpd'])
        # msd.load_state_dict(state_dict_do_['msd'])
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
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True)
        generator_without_ddp = generator.module
        stylespeech = nn.parallel.DistributedDataParallel(stylespeech, device_ids=[rank], find_unused_parameters=True)
        stylespeech_without_ddp = stylespeech.module
        mpd = nn.parallel.DistributedDataParallel(mpd, device_ids=[rank])
        mpd_without_ddp = mpd.module
        msd = nn.parallel.DistributedDataParallel(msd, device_ids=[rank])
        msd_without_ddp = msd.module

    # Optimizers
    if (args.optim_g == "G_only"):
        optim_g = torch.optim.AdamW(generator.parameters(), config.lr_g, betas=[config.adam_b1, config.adam_b2])
        optim_ss = torch.optim.Adam(stylespeech.parameters(), config.lr_ss, betas=config.betas, eps=config.eps)
    else: # args.optim_g = "G_and_SS"
        optim_g = torch.optim.AdamW(itertools.chain(generator.parameters(), stylespeech.parameters()), config.lr_g, betas=[config.adam_b1, config.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                config.lr_d, betas=[config.adam_b1, config.adam_b2])
    # optim_d = torch.optim.AdamW(mpd.parameters(), config.lr_d, betas=[h.adam_b1, h.adam_b2])
    print("Optimizer and Loss Function Defined.")

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        if (args.optim_g == "G_only"):
            optim_ss.load_state_dict(state_dict_do['optim_ss'])
    # else:
    #     if (args.optim_g == "G_only"):
    #         optim_g.load_state_dict(state_dict_do_['optim_g'])
    #     optim_d.load_state_dict(state_dict_do_['optim_d'])

    # h.lr_decay = 0.8
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay, last_epoch=last_epoch)
    if (args.optim_g == "G_only"):
        scheduled_optim = ScheduledOptim(optim_ss, config.decoder_hidden, config.n_warm_up_step, steps)
    
    train_loader = prepare_dataloader(args.data_path, "train.txt", shuffle=True, batch_size=config.batch_size) 
    if rank == 0:        
        validation_loader = prepare_dataloader(args.data_path, "val.txt", shuffle=True, batch_size=1, val=True) 
        sw = SummaryWriter(os.path.join(args.save_path, 'logs'))
        # Init logger
        # log_path = os.path.join(args.save_path, 'log.txt')
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

    # AutoCast #
    if args.use_scaler:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(max(0, last_epoch), args.training_epochs):
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
                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
            
            # Forwards
            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = stylespeech(
                    text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            # mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, _, hidden_output = stylespeech(
            #         text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            indices = [[mel_start_idx[i]+j for j in range(32)] for i in range(config.batch_size)]
            indices = torch.Tensor(indices).type(torch.int64)
            indices = torch.unsqueeze(indices, 2).expand(-1, -1, 256).cuda()
            
            # wav_output = generator(acoustic_adaptor_output, hidden_output, indices=indices)
            # print("acoustic shape: ", acoustic_adaptor_output.shape)
            # print("hidden shape: ", hidden_output[1].shape)
            # wav_output = generator(hidden_output[1], hidden_output[2:], indices=indices)
            wav_output = generator(hidden_output[2], hidden_output[3:], indices=indices)
            wav_output_mel = utils.mel_spectrogram(wav_output.squeeze(1), config.n_fft, config.n_mel_channels, config.sampling_rate, config.hop_size, config.win_size,
                                          config.fmin, config.fmax_for_loss)
            
            wav_crop = torch.unsqueeze(wav, 1)

            indices2 = [[mel_start_idx[i]+j for j in range(32)] for i in range(config.batch_size)]
            indices2 = torch.Tensor(indices2).type(torch.int64)
            indices2 = torch.unsqueeze(indices2, 2).expand(-1, -1, 80).cuda()

            mel_crop = torch.transpose(torch.gather(mel_target, 1, indices2), 1, 2)
            # mel_crop = torch.transpose(torch.gather(mel_output, 1, indices2), 1, 2)

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
            if (args.optim_g == "G_and_SS"):
                loss_gen_all, loss_gen_list = G_step(mpd, msd, loss_cvc, optim_g, wav_crop, mel_crop, wav_output, wav_output_mel, 
                                                loss_ss, mel_output, mel_target, 
                                                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len,
                                                scaler=scaler, retain_graph=False, G_only=False, lmel_hifi=config.lmel_hifi, lmel_ss=config.lmel_ss)
                if scaler != None:
                    scaler.update()
            else: 
                loss_gen_all, loss_gen_list = G_step(mpd, msd, loss_cvc, optim_g, wav_crop, mel_crop, wav_output, wav_output_mel, 
                                                loss_ss, mel_output, mel_target, 
                                                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len,
                                                scaler=scaler, retain_graph=True, G_only=True, lmel_hifi=config.lmel_hifi, lmel_ss=config.lmel_ss)
                # StyleSpeech optimize
                scheduled_optim.zero_grad()
                loss_ss_all = SS_step(loss_ss, mel_output, mel_target, 
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len, scaler=scaler)
                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(stylespeech.parameters(), config.grad_clip_thresh)
                    scheduled_optim.step_and_update_lr(scaler=None)
                else:
                    scaler.unscale_(scheduled_optim._optimizer)
                    torch.nn.utils.clip_grad_norm_(stylespeech.parameters(), config.grad_clip_thresh)
                    # scheduled_optim.step_and_update_lr(scaler=None)
                    scheduled_optim.step(scaler=scaler)
                    scaler.update()
                    scheduled_optim.update_lr()
            # return
            if rank == 0:
                # STDOUT & log.txt logging
                if steps % args.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(mel_crop, wav_output_mel).item()

                    if (args.optim_g == "G_only"):
                        str1 = 'Steps : {:d}, SS Loss Total : {:4.3f}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                                    steps, loss_ss_all.item(), loss_gen_all, mel_error, time.time() - start_b)
                    else:
                        # str1 = 'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                        #             steps, loss_gen_all, mel_error, time.time() - start_b)
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
                    if (args.optim_g == "G_only"):
                        save_dict['optim_ss'] = optim_ss.state_dict()
                    utils.save_checkpoint(checkpoint_path, save_dict)
                    

                # Tensorboard summary logging
                if steps % args.summary_interval == 0:
                    print('Tensorboard summary logging')
                    if (args.optim_g == "G_only"):
                        sw.add_scalar("training/loss_ss_all", loss_ss_all, steps)
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % args.validation_interval == 0: # and steps != 0:
                    print('Validation')
                    stylespeech.eval()
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            sid, text, mel_target, mel_start_idx, wav, \
                                    D, log_D, f0, energy, \
                                    src_len, mel_len, max_src_len, max_mel_len = parse_batch(batch)
                            
                            mel_output, src_output, style_vector, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, acoustic_adaptor_output, hidden_output = stylespeech(
                                    text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
                            
                            # wav_output = generator(torch.transpose(acoustic_adaptor_output.detach(), 1, 2), hidden_output)
                            # wav_output = generator(acoustic_adaptor_output, hidden_output)
                            # wav_output = generator(hidden_output[1], hidden_output[2:])
                            wav_output = generator(hidden_output[2], hidden_output[3:])
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
            
            # cleanup()
            # return

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

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--data_path', default='/v9/dongchan/TTS/dataset/LibriTTS/preprocessed')
    parser.add_argument('--save_path', default='exp_default')
    parser.add_argument('--checkpoint_path', default='cp_default')
    parser.add_argument('--training_epochs', default=1, type=int)
    parser.add_argument('--stdout_interval', default=100, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    
    parser.add_argument('--config', default='./configs/config.json') # Configurations for StyleSpeech model
    parser.add_argument('--optim_g', default='G_and_SS') # "G_and_SS" or "G_only"
    parser.add_argument('--use_scaler', default=False)
    parser.add_argument('--freeze_ss', default=False)


    args = parser.parse_args()
    args.use_scaler = bool(args.use_scaler)
    args.freeze_ss = bool(args.freeze_ss)
    
    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = utils.AttrDict(config)
    if not os.path.exists('{}/config.json'.format(args.checkpoint_path)):
        utils.build_env(args.config, 'config.json', args.checkpoint_path)
    
    # ngpus = torch.cuda.device_count()
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
