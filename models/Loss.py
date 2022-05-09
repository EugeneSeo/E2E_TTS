import torch
import torch.nn as nn


class StyleSpeechLoss(nn.Module):
    """ StyleSpeech Loss """
    def __init__(self):
        super(StyleSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, mel, mel_target, log_d_predicted, log_d_target, 
                        p_predicted, p_target, e_predicted, e_target, src_len, mel_len):
        B = mel_target.shape[0]
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        mel_loss = 0.
        d_loss = 0.
        p_loss = 0.
        e_loss = 0.

        for b, (mel_l, src_l) in enumerate(zip(mel_len, src_len)):
            mel_loss += self.mae_loss(mel[b, :mel_l, :], mel_target[b, :mel_l, :])
            d_loss += self.mse_loss(log_d_predicted[b, :src_l], log_d_target[b, :src_l])
            p_loss += self.mse_loss(p_predicted[b, :src_l], p_target[b, :src_l])
            e_loss += self.mse_loss(e_predicted[b, :src_l], e_target[b, :src_l])

        mel_loss = mel_loss / B
        d_loss = d_loss / B
        p_loss = p_loss / B
        e_loss = e_loss / B

        return mel_loss, d_loss, p_loss, e_loss

class StyleSpeechLoss2(nn.Module):
    """ StyleSpeech Loss """
    def __init__(self):
        super(StyleSpeechLoss2, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, mel, mel_target, log_d_predicted, log_d_target, 
                        p_predicted, p_target, e_predicted, e_target, src_len, mel_len):
        B = mel_target.shape[0]
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        mel_loss = 0.
        d_loss = 0.
        p_loss = 0.
        e_loss = 0.

        for b, (mel_l, src_l) in enumerate(zip(mel_len, src_len)):
            # mel_loss += self.mae_loss(mel[b, :mel_l, :], mel_target[b, :mel_l, :])
            d_loss += self.mse_loss(log_d_predicted[b, :src_l], log_d_target[b, :src_l])
            p_loss += self.mse_loss(p_predicted[b, :src_l], p_target[b, :src_l])
            e_loss += self.mse_loss(e_predicted[b, :src_l], e_target[b, :src_l])

        # mel_loss = mel_loss / B
        d_loss = d_loss / B
        p_loss = p_loss / B
        e_loss = e_loss / B

        return 0, d_loss, p_loss, e_loss

class LSGANLoss(nn.Module):
    """ LSGAN Loss """
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, r, is_real):
        if is_real: 
            ones = torch.ones(r.size(), requires_grad=False).to(r.device)
            loss = self.criterion(r, ones)
        else:
            zeros = torch.zeros(r.size(), requires_grad=False).to(r.device)
            loss = self.criterion(r, zeros)
        return loss


class CVCLoss(nn.Module):
    def __init__(self):
        super(CVCLoss, self).__init__()
        self.nce_T = 15
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, wav_pred, wav_orig):
        batch_size = wav_pred.shape[0]
        dim = 32

        # wav_pred = wav_pred.view(batch_size, -1, dim)
        # wav_orig = wav_orig.view(batch_size, -1, dim)
        wav_pred = wav_pred.view(batch_size, dim, -1).permute(0, 2, 1)
        wav_orig = wav_orig.view(batch_size, dim, -1).permute(0, 2, 1)

        wav_pred_norm = torch.norm(wav_pred, dim=2)
        wav_orig_norm = torch.norm(wav_orig, dim=2)

        wav_pred = wav_pred / wav_pred_norm.view(batch_size, -1, 1)
        wav_orig = wav_orig / wav_orig_norm.view(batch_size, -1, 1)

        # # positive samples: each predictions should be close to the corresponding wav-blocks
        l_pos = torch.bmm(wav_pred.contiguous().view(batch_size, 1, -1), wav_orig.contiguous().view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1) # (B, 1)
        
        # negativa samples: reshape features to batch size
        melbin_no = wav_orig.shape[1]
        l_neg_curbin = torch.bmm(wav_pred, wav_orig.transpose(2, 1)) # (B, 32, 32)

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(melbin_no, device=wav_pred.device, dtype=torch.bool)[None, :, :]
        l_neg_curbin.masked_fill_(diagonal, -10.0)
        
        l_neg = l_neg_curbin.view(batch_size, -1)

        # New
        l_neg2_curbin = torch.bmm(wav_pred, wav_pred.transpose(2, 1)) # (B, 32, 32)
        diagonal2 = torch.eye(melbin_no, device=wav_pred.device, dtype=torch.bool)[None, :, :]
        l_neg2_curbin.masked_fill_(diagonal, -10.0)
        l_neg_internal = l_neg2_curbin.view(batch_size, -1)

        out = torch.cat((l_pos, l_neg, l_neg_internal), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=wav_pred.device))
        
        return loss.mean()