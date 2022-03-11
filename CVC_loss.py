import torch.nn as nn
import torch
import torch.nn.functional as F

# original code: CVC interspeech 2021
class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(self.opt.batch_size, -1, dim)
        feat_k = feat_k.view(self.opt.batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

# My code! 
class CVCLoss(nn.Module):
    def __init__(self):
        super(CVCLoss, self).__init__()
        self.nce_T = 10
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, wav_pred, wav_orig):
        batch_size = wav_pred.shape[0]
        dim = 32

        wav_pred = wav_pred.view(batch_size, -1, dim)
        wav_orig = wav_orig.view(batch_size, -1, dim)

        wav_pred_norm = torch.norm(wav_pred, dim=2)
        wav_orig_norm = torch.norm(wav_orig, dim=2)

        wav_pred = wav_pred / wav_pred_norm.view(batch_size, -1, 1)
        wav_orig = wav_orig / wav_orig_norm.view(batch_size, -1, 1)

        # # positive samples: each predictions should be close to the corresponding wav-blocks
        l_pos = torch.bmm(wav_pred.view(batch_size, 1, -1), wav_orig.view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1) # (B, 1)
        
        # negativa samples: reshape features to batch size
        melbin_no = wav_orig.shape[1]
        l_neg_curbin = torch.bmm(wav_pred, wav_orig.transpose(2, 1)) # (B, 256, 256)

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(melbin_no, device=wav_pred.device, dtype=torch.bool)[None, :, :]
        l_neg_curbin.masked_fill_(diagonal, -10.0)
        
        l_neg = l_neg_curbin.view(batch_size, -1)

        # New
        l_neg2_curbin = torch.bmm(wav_pred, wav_pred.transpose(2, 1)) # (B, 256, 256)
        diagonal2 = torch.eye(melbin_no, device=wav_pred.device, dtype=torch.bool)[None, :, :]
        l_neg2_curbin.masked_fill_(diagonal, -10.0)
        l_neg_internal = l_neg2_curbin.view(batch_size, -1)

        out = torch.cat((l_pos, l_neg, l_neg_internal), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=wav_pred.device))
        
        return loss.mean()
