import torch
import torch.nn.functional as F
import numpy as np
from models.Hifigan import feature_loss, generator_loss, discriminator_loss

def D_step(mpd, msd, optim_d, y, y_g_hat, retain_graph=True):
    optim_d.zero_grad()

    y = y.requires_grad_(True)
    # MPD
    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

    r1_grads_mpd = torch.autograd.grad(outputs=[temp.sum() for temp in y_df_hat_r], inputs=[y], 
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    r1_penalty_mpd = (r1_grads_mpd.reshape(r1_grads_mpd.shape[0], -1).norm(2, dim=1) ** 2).sum() * 0.5
    
    # MSD
    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
    r1_grads_msd = torch.autograd.grad(outputs=[temp.sum() for temp in y_ds_hat_r], inputs=[y], 
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    r1_penalty_msd = (r1_grads_msd.reshape(r1_grads_msd.shape[0], -1).norm(2, dim=1) ** 2).sum() * 0.5

    loss_disc_all = loss_disc_s + loss_disc_f + (r1_penalty_mpd + r1_penalty_msd) * 1

    # new line
    rtn_list = [loss_disc_all.item(), loss_disc_s.item(), loss_disc_f.item(), r1_penalty_mpd.item(), r1_penalty_msd.item()]
    loss_disc_all.backward(retain_graph=retain_graph)
    optim_d.step()
    return loss_disc_all, rtn_list


def G_step(mpd, msd, loss_cvc, optim_g, y, y_mel, y_g_hat, y_g_hat_mel, \
            loss_ss, mel_output, mel_target, log_duration_output, log_D, \
            f0_output, f0, energy_output, energy, src_len, mel_len,
            retain_graph=True, G_only=True, lmel_hifi=45, lmel_ss=1):
    optim_g.zero_grad()

    # L1 Mel-Spectrogram Loss
    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 

    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    
    # loss_cvc_v = loss_cvc(y_g_hat, y)
    loss_cvc_v = 0

    if G_only:
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + loss_cvc_v
        total_loss = None
    else:
        mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
        # Total loss
        total_loss = mel_loss + d_loss + f_loss + e_loss
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + loss_cvc_v + total_loss * lmel_ss 
    
    # new line
    rtn_list = [loss_gen_all.item(), loss_gen_s.item(), loss_gen_f.item(), loss_fm_s.item(), 
                loss_fm_f.item(), loss_mel.item() * lmel_hifi, total_loss.item() * lmel_ss if total_loss else 0, loss_cvc_v]
    loss_gen_all.backward(retain_graph=retain_graph)
    optim_g.step()
    return loss_gen_all, rtn_list


def G_step_ali(mpd, msd, loss_cvc, loss_fwdsum, optim_g, optim_ali, y, y_mel, y_g_hat, y_g_hat_mel, \
            loss_ss, mel_output, mel_target, log_duration_output, log_D, log_D_attn, \
            f0_output, f0, energy_output, energy, src_len, mel_len,
            retain_graph=True, G_only=True, lmel_hifi=45, lmel_ss=1, alin_out=None):
    optim_g.zero_grad()

    # Alignment Loss
    loss_align = loss_fwdsum(alin_out[3], src_len, mel_len)
    log_D = alin_out[2]
    log_D_attn = alin_out[3]
    # loss_align = 0

    # L1 Mel-Spectrogram Loss
    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 

    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    
    # loss_cvc_v = loss_cvc(y_g_hat, y)
    loss_cvc_v = 0

    if G_only:
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + loss_cvc_v
        total_loss = None
    else:
        mel_loss, d_loss, f_loss, e_loss = loss_ss(mel_output, mel_target, 
                log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)
        # Total loss
        total_loss = mel_loss + d_loss + f_loss + e_loss
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * lmel_hifi + loss_cvc_v + total_loss * lmel_ss 
    
    # new line
    rtn_list = [loss_gen_all.item(), loss_gen_s.item(), loss_gen_f.item(), loss_fm_s.item(), 
                loss_fm_f.item(), loss_mel.item() * lmel_hifi, total_loss.item() * lmel_ss if total_loss else 0, loss_cvc_v]
    
    loss_gen_all += loss_align
    loss_gen_all.backward(retain_graph=retain_graph)
    optim_g.step()
    return loss_gen_all, rtn_list