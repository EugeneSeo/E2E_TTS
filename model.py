import datetime
import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hifi_gan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from StyleSpeech.models.StyleSpeech import StyleSpeech
from StyleSpeech.models.Loss import StyleSpeechLoss
from torch.cuda.amp import autocast, GradScaler

def parse_batch(batch, device):
    sid = torch.from_numpy(batch["sid"]).long().to(device)
    text = torch.from_numpy(batch["text"]).long().to(device)
    mel_target = torch.from_numpy(batch["mel_target"]).float().to(device)
    D = torch.from_numpy(batch["D"]).long().to(device)
    log_D = torch.from_numpy(batch["log_D"]).float().to(device)
    f0 = torch.from_numpy(batch["f0"]).float().to(device)
    energy = torch.from_numpy(batch["energy"]).float().to(device)
    src_len = torch.from_numpy(batch["src_len"]).long().to(device)
    mel_len = torch.from_numpy(batch["mel_len"]).long().to(device)
    max_src_len = np.max(batch["src_len"]).astype(np.int32)
    max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
    ##############################################################
    mel_start_idx = torch.Tensor(batch["mel_start_idx"]).int().to(device)
    wav = torch.from_numpy(np.array(batch["wav"])).to(device)
    ##############################################################
    return sid, text, mel_target, mel_start_idx, wav, \
            D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len

def D_step(mpd, msd, optim_d, y, y_g_hat, scaler=None, retain_graph=True):
    optim_d.zero_grad()

    if scaler is None:
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward(retain_graph=retain_graph)
        optim_d.step()
        return loss_disc_all
    
    with autocast():
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        
    scaler.scale(loss_disc_all).backward(retain_graph=retain_graph)
    scaler.step(optim_d)  

    return loss_disc_all

def G_step(mpd, msd, optim_g, y, y_mel, y_g_hat, y_g_hat_mel, \
            loss_ss, mel_output, mel_target, log_duration_output, log_D, \
            f0_output, f0, energy_output, energy, src_len, mel_len,
            scaler=None, retain_graph=True, G_only=True, lmel_hifi=45, lmel_ss=1):
    optim_g.zero_grad()

    if scaler is None:
        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        
        if G_only:
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi
        else:
            mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
            # Total loss
            total_loss = mel_loss + d_loss + f_loss + e_loss
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + total_loss * lmel_ss 
        
        loss_gen_all.backward(retain_graph=retain_graph)
        optim_g.step()
        return loss_gen_all
    
    with autocast():
        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        
        if G_only:
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi
        else:
            mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
            # Total loss
            total_loss = mel_loss + d_loss + f_loss + e_loss    
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + total_loss * lmel_ss

    scaler.scale(loss_gen_all).backward(retain_graph=retain_graph)
    scaler.step(optim_g)   

    return loss_gen_all

def SS_step(loss_ss, mel_output, mel_target, log_duration_output, log_D, \
            f0_output, f0, energy_output, energy, src_len, mel_len, scaler=None):
    # optim_ss.zero_grad()
    if scaler is None:
        mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
        # Total loss
        total_loss = mel_loss + d_loss + f_loss + e_loss

        total_loss.backward()
        return total_loss
    
    with autocast():
        mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
        # Total loss
        total_loss = mel_loss + d_loss + f_loss + e_loss
    
    scaler.scale(total_loss).backward()
    return total_loss


class WBLogger:
    def __init__(self, opts, project_name):
        wandb_run_name = opts.name
        wandb_id = wandb.util.generate_id() if opts.wandb_id is None else opts.wandb_id
        wandb.init(project=project_name, entity='stylelip', config=opts, name=wandb_run_name, id=wandb_id, resume=wandb_id)
    
    @staticmethod
    def finish():
        wandb.finish()

    @staticmethod
    def log_best_model():
        wandb.run.summary["best-model-save-time"] = datetime.datetime.now()

    @staticmethod
    def log(prefix, metrics_dict, commit=False):
        log_dict = {f'{prefix}.{key}': value for key, value in metrics_dict.items()}
        wandb.log(log_dict, commit=commit)

    @staticmethod
    def log_images_to_wandb(prefix, imgs_data, commit=False):
        imgs_data = [wandb.Image(img) for img in imgs_data]
        wandb.log({f"{prefix}.images": imgs_data}, commit=commit)