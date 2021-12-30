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
    # "mel_start_idx": mel_start_idxes, "wav": wavs,
    mel_start_idx = torch.Tensor(batch["mel_start_idx"]).int().to(device)
    # mel_start_idx = batch["mel_start_idx"]
    wav = torch.Tensor(batch["wav"]).to(device)
    ##############################################################
    return sid, text, mel_target, mel_start_idx, wav, \
            D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len

def D_step(mpd, msd, optim_d, y, y_g_hat, scaler=None):
    optim_d.zero_grad()

    with autocast():
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        
    if scaler is None:
        loss_disc_all.backward(retain_graph=True)
        optim_d.step()
    else:
        scaler.scale(loss_disc_all).backward(retain_graph=True)
        scaler.step(optim_d)  

    return loss_disc_all

def G_step(mpd, msd, optim_g, y, y_mel, y_g_hat, y_g_hat_mel, scaler=None):
    optim_g.zero_grad()

    with autocast():
        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        
    if scaler is None:
        loss_gen_all.backward(retain_graph=True)
        optim_g.step()
    else:
        scaler.scale(loss_gen_all).backward(retain_graph=True)
        scaler.step(optim_g)    

    return loss_gen_all
'''
class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.ss_loss = StyleSpeechLoss()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        # self.l1_loss = F.l1_loss
        self.mpd.train()
        self.msd.train()

    def D_step(optim_d, y, y_g_hat):
        optim_d.zero_grad()
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        optim_d.step()

        return loss_disc_all

    def G_step(optim_g, y, y_mel, y_g_hat, y_g_hat_mel):
        optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        optim_g.step()

        return loss_gen_all
'''

def SS_step(loss_ss, optim_ss, mel_output, mel_target, log_duration_output, log_D, \
            f0_output, f0, energy_output, energy, src_len, mel_len, scaler=None):
    with autocast():
        # optim_ss.zero_grad()
        mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
        # Total loss
        total_loss = mel_loss + d_loss + f_loss + e_loss
    # Backward
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    return total_loss


