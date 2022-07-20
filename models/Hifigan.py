import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
from models.Modules import Mish, get_sinusoid_encoding_table
import numpy as np
LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator_E2E(torch.nn.Module):
    def __init__(self, h):
        super(Generator3, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(Conv1d(h.decoder_hidden, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        #################################################
        self.ss_hidden = nn.ModuleList()
        k_size = [1, 16, 128, 256, 512]
        s_size = [1,  8,  64, 128, 256]
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**i)
            self.ss_hidden.append(ConvTranspose1d(h.decoder_hidden, ch, k_size[i], s_size[i], padding=(k_size[i]-s_size[i])//2))
        
        # new!
        # variance adaptor part
        self.ss_hidden2 = nn.ModuleList()
        self.lns = nn.ModuleList()
        k_size = [3, 5, 7, 11, 13]
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**i)
            self.ss_hidden2.append(Conv1d(ch, ch, k_size[i], 1, padding=(k_size[i]-1)//2))
            self.lns.append(nn.LayerNorm(ch))
            self.lns.append(nn.LayerNorm(ch))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # up to here

        self.max_seq_len = 1000
        self.d_model = h.decoder_hidden
        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)
        #################################################

    def forward(self, x, hidden, indices=None):
        #################################################
        batch_size, max_len = x.shape[0], x.shape[1]
        # poistion encoding
        if x.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(x.shape[1], self.d_model)[:x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        x = x + position_embedded
        
        if (indices == None):
            x = torch.transpose(x, 1, 2)
        else:
            x = torch.transpose(torch.gather(x, 1, indices), 1, 2)
        #################################################

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            #################################################   
            # if (i < 2):
            # position encoding (NEW)
            h = hidden[i] + position_embedded

            if (indices != None):
                h = torch.gather(h, 1, indices)

            h = self.ss_hidden[i](torch.transpose(h, 1, 2))
            # new!
            h = self.relu(h).transpose(1,2)
            h = self.dropout(self.lns[2*i](h)).transpose(1,2)
            h = self.ss_hidden2[i](h)
            h = self.relu(h).transpose(1,2)
            h = self.dropout(self.lns[2*i+1](h)).transpose(1,2)
            # up to here
            
            x = x + h
            #################################################
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# Current Best Model w/ Interpolation part changed (original: hidden state division -> interpolation, current: interpolation -> division)
class Generator_intpol(torch.nn.Module):
    def __init__(self, h):
        super(Generator_intpol, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.decoder_hidden, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = nn.Sequential(
            weight_norm(Conv1d(ch, 4, 7, 1, padding=3)),
            weight_norm(Conv1d(4, 1, 5, 1, padding=2)))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.expand_model = ExpandFrame()
        self.up_cum = [1]
        for i, u in enumerate(h.upsample_rates):
            self.up_cum.append(self.up_cum[-1] * u)

        # variance adaptor part
        self.ss_hidden = nn.ModuleList()
        self.lns = nn.ModuleList()
        k_size = [3, 5, 7, 11]
        for i in range(len(k_size)):
            ch = h.upsample_initial_channel//(2**(i + h.itp_start_idx))
            self.ss_hidden.append(Conv1d(h.decoder_hidden, ch, k_size[i], 1, padding=(k_size[i]-1)//2))
            self.lns.append(nn.LayerNorm(ch))
        
        self.relu = nn.ReLU()
        self.mish = Mish()
        self.dropout = nn.Dropout(0.1)

        self.max_seq_len = 1000
        self.d_model = h.decoder_hidden
        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.itp_start = h.itp_start_idx
        self.itp_end = h.itp_end_idx

    def forward(self, x, hidden, indices=None, mel_start_idx=None):
        batch_size, max_len, encoder_hidden = x.shape
        # poistion encoding
        if x.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(x.shape[1], self.d_model)[:x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        x = x + position_embedded
        
        if (indices == None): # inference
            x = torch.transpose(x, 1, 2)
        else:
            x = torch.transpose(torch.gather(x, 1, indices), 1, 2)
            # max_len = 32

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            
            if (self.itp_start <= i <= self.itp_end): # interpolation indice
                # position encoding
                h = hidden[i - self.itp_start] + position_embedded
                l = torch.tensor([[self.up_cum[i] for _ in range(max_len)] for _ in range(batch_size)], device=x.device)
                
                indices = None
                if mel_start_idx != None: # train
                    indices = [[mel_start_idx[b]*self.up_cum[i]+j for j in range(32*self.up_cum[i])] for b in range(batch_size)]
                    indices = torch.tensor(indices, device=x.device, dtype=torch.int64).unsqueeze(1)
                h = self.expand_model(h, l, indices=indices)

                h = self.ss_hidden[i - self.itp_start](h)
                h = self.mish(h).transpose(1,2)
                h = self.dropout(self.lns[i - self.itp_start](h)).transpose(1,2)            
                x = x + h
            
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# Current Best Model
class Generator_intpol_conv(torch.nn.Module):
    def __init__(self, h):
        super(Generator_intpol_conv, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.decoder_hidden, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 4, 7, 1, padding=3))
        self.conv_post_2 = weight_norm(Conv1d(4, 1, 5, 1, padding=2))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.expand_model = ExpandFrame()
        self.up_cum = [1]
        for i, u in enumerate(h.upsample_rates):
            self.up_cum.append(self.up_cum[-1] * u)

        # variance adaptor part
        self.ss_hidden = nn.ModuleList()
        self.lns = nn.ModuleList()
        k_size = [3, 5, 7, 11]
        for i in range(len(k_size)):
            ch = h.upsample_initial_channel//(2**(i + h.itp_start_idx))
            self.ss_hidden.append(Conv1d(h.decoder_hidden, ch, k_size[i], 1, padding=(k_size[i]-1)//2))
            self.lns.append(nn.LayerNorm(ch))
        
        self.relu = nn.ReLU()
        self.mish = Mish()
        self.dropout = nn.Dropout(0.1)

        self.max_seq_len = 1000
        self.d_model = h.decoder_hidden
        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.itp_start = h.itp_start_idx
        self.itp_end = h.itp_end_idx

    def forward(self, x, hidden, indices=None):
        batch_size, max_len = x.shape[0], x.shape[1]
        # poistion encoding
        if x.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(x.shape[1], self.d_model)[:x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        x = x + position_embedded
        
        if (indices == None): # inference
            x = torch.transpose(x, 1, 2)
        else:
            x = torch.transpose(torch.gather(x, 1, indices), 1, 2)
            max_len = 32

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            
            if (self.itp_start <= i <= self.itp_end): # interpolation indice
                # position encoding
                h = hidden[i - self.itp_start] + position_embedded
                if (indices != None): # train
                    h = torch.gather(h, 1, indices)

                #################################################   
                l = torch.Tensor([[self.up_cum[i] for _ in range(max_len)] for _ in range(batch_size)])
                # print(l)
                l = l.to(x.device)
                h = self.expand_model(h, l)
                #################################################   
                h = self.ss_hidden[i - self.itp_start](h)
                h = self.mish(h).transpose(1,2)
                h = self.dropout(self.lns[i - self.itp_start](h)).transpose(1,2)            
                x = x + h
            
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = self.conv_post_2(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ConstantExpandFrame(nn.Module):
    def __init__(self):
        super(ConstantExpandFrame, self).__init__()
        pass

    def forward(self, hidden, duration):
        copy_n = int(duration[0][0].item())
        B, D, N = hidden.shape
        out = hidden.transpose(1,2).view(B, N, D, 1).repeat(1, 1, 1, copy_n).view(B, N, -1)
        return out

# Gaussian Interpolation (inspired by EATS)
class ExpandFrame(nn.Module):
    def __init__(self):
        super(ExpandFrame, self).__init__()
        pass

    def forward(self, hidden, duration, indices=None):
        t = torch.round(torch.sum(duration, dim=-1, keepdim=True, dtype=torch.float32)) #[B, 1]: (batch, total duration)
        e = torch.cumsum(duration, dim=-1).float() #[B, L]: (batch, cumulative summation of duration)
        c = e - 0.5 * torch.round(duration.type(torch.float32)) #[B, L]: token center positions

        if indices != None:
            t = indices #[B, 1, T]
        else:
            t = torch.arange(0, torch.max(t)) #[0, 1, 2, ..., max_length-1]
            t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2) #[B, L, 1]
        
        t = t.to(hidden.device)
        c = c.to(hidden.device)
        
        # temp = -0.1
        temp = -1.0 / (5 * np.sqrt(duration.cpu()[0][0].item()))
        w_1 = torch.exp(temp * (t - c) ** 2)  # [B, L, T]
        w_2 = torch.sum(torch.exp(temp * (t - c) ** 2), dim=1, keepdim=True)  # [B, 1, T]
        w = w_1 / w_2 # [B, L, T]
        out = torch.matmul(w.transpose(1, 2), hidden).transpose(1,2)
        return out

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


from dwt import DWT_1D

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # self.convs = nn.ModuleList([
        #     norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        # ])
        self.convs = nn.ModuleList([
            Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        # self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.conv_post = Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# DWT discriminator proposed in Fre-GAN
class DiscriminatorP_dwt(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP_dwt, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        ##################
        # From the implementation of the paper Fre-GAN
        # https://github.com/rishikksh20/Fre-GAN-pytorch/blob/master/discriminator.py
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_proj1 = norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.dwt_proj2 = norm_f(Conv2d(1, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv3 = norm_f(Conv1d(8, 1, 1))
        self.dwt_proj3 = norm_f(Conv2d(1, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        ##################

        # self.convs = nn.ModuleList([
        #     norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
        #     norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        # ])
        self.convs = nn.ModuleList([
            Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        # self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.conv_post = Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        #################
        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        # 1d to 2d
        b, c, t = x_d1.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d1 = F.pad(x_d1, (0, n_pad), "reflect")
            t = t + n_pad
        x_d1 = x_d1.view(b, c, t // self.period, self.period)

        x_d1 = self.dwt_proj1(x_d1)

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        # 1d to 2d
        b, c, t = x_d2.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d2 = F.pad(x_d2, (0, n_pad), "reflect")
            t = t + n_pad
        x_d2 = x_d2.view(b, c, t // self.period, self.period)

        x_d2 = self.dwt_proj2(x_d2)

        # DWT 3

        x_d3_high1, x_d3_low1 = self.dwt1d(x_d2_high1)
        x_d3_high2, x_d3_low2 = self.dwt1d(x_d2_low1)
        x_d3_high3, x_d3_low3 = self.dwt1d(x_d2_high2)
        x_d3_high4, x_d3_low4 = self.dwt1d(x_d2_low2)
        x_d3 = self.dwt_conv3(
            torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4],
                      dim=1))
        # 1d to 2d
        b, c, t = x_d3.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d3 = F.pad(x_d3, (0, n_pad), "reflect")
            t = t + n_pad
        x_d3 = x_d3.view(b, c, t // self.period, self.period)

        x_d3 = self.dwt_proj3(x_d3)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        i = 0
        #################

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            ########################
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            elif i == 1:
                x = torch.cat([x, x_d2], dim=2)
            elif i == 2:
                x = torch.cat([x, x_d3], dim=2)
            else:
                x = x
            i = i + 1
            ########################
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# DWT discriminator proposed in Fre-GAN
class MultiPeriodDiscriminator_dwt(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator_dwt, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP_dwt(2),
            DiscriminatorP_dwt(3),
            DiscriminatorP_dwt(5),
            DiscriminatorP_dwt(7),
            DiscriminatorP_dwt(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # self.convs = nn.ModuleList([
        #     norm_f(Conv1d(1, 128, 15, 1, padding=7)),
        #     norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
        #     norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
        #     norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
        #     norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        # ])
        self.convs = nn.ModuleList([
            Conv1d(1, 128, 15, 1, padding=7),
            Conv1d(128, 128, 41, 2, groups=4, padding=20),
            Conv1d(128, 256, 41, 2, groups=16, padding=20),
            Conv1d(256, 512, 41, 4, groups=16, padding=20),
            Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        # self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))
        self.conv_post = Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# DWT discriminator proposed in Fre-GAN
class DiscriminatorS_dwt(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS_dwt, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        ##################
        # From the implementation of the paper Fre-GAN
        # https://github.com/rishikksh20/Fre-GAN-pytorch/blob/master/discriminator.py
        self.dwt1d = DWT_1D()
        # self.dwt_conv1 = norm_f(Conv1d(2, 128, 15, 1, padding=7))
        # self.dwt_conv2 = norm_f(Conv1d(4, 128, 41, 2, padding=20))
        self.dwt_conv1 = Conv1d(2, 128, 15, 1, padding=7)
        self.dwt_conv2 = Conv1d(4, 128, 41, 2, padding=20)
        ##################
        # self.convs = nn.ModuleList([
        #     norm_f(Conv1d(1, 128, 15, 1, padding=7)),
        #     norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
        #     norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
        #     norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
        #     norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        # ])
        self.convs = nn.ModuleList([
            Conv1d(1, 128, 15, 1, padding=7),
            Conv1d(128, 128, 41, 2, groups=4, padding=20),
            Conv1d(128, 256, 41, 2, groups=16, padding=20),
            Conv1d(256, 512, 41, 4, groups=16, padding=20),
            Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        # self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))
        self.conv_post = Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []

        ##################
        # From the implementation of the paper Fre-GAN
        # https://github.com/rishikksh20/Fre-GAN-pytorch/blob/master/discriminator.py
        
        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        i = 0
        ##################
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            #############################
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            if i == 1:
                x = torch.cat([x, x_d2], dim=2)
            i = i + 1
            #############################
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# DWT discriminator proposed in Fre-GAN
class MultiScaleDiscriminator_dwt(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiScaleDiscriminator_dwt, self).__init__()
        # norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        ##################
        # From the implementation of the paper Fre-GAN
        # https://github.com/rishikksh20/Fre-GAN-pytorch/blob/master/discriminator.py
        self.dwt1d = DWT_1D()
        # self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        # self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.dwt_conv1 = Conv1d(2, 1, 1)
        self.dwt_conv2 = Conv1d(4, 1, 1)
        ##################
        self.discriminators = nn.ModuleList([
            DiscriminatorS_dwt(use_spectral_norm=True),
            DiscriminatorS_dwt(),
            DiscriminatorS_dwt(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        ##################
        # From the implementation of the paper Fre-GAN
        # https://github.com/rishikksh20/Fre-GAN-pytorch/blob/master/discriminator.py
        
        # DWT 1
        y_hi, y_lo = self.dwt1d(y)
        y_1 = self.dwt_conv1(torch.cat([y_hi, y_lo], dim=1))
        x_d1_high1, x_d1_low1 = self.dwt1d(y_hat)
        y_hat_1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(y_hi)
        x_d2_high2, x_d2_low2 = self.dwt1d(y_lo)
        y_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        y_hat_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        ##################

        for i, d in enumerate(self.discriminators):
            # if i != 0:
            #     y = self.meanpools[i-1](y)
            #     y_hat = self.meanpools[i-1](y_hat)
            if i == 1:
                y = y_1
                y_hat = y_hat_1
            if i == 2:
                y = y_2
                y_hat = y_hat_2

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class Generator_FastSpeech2s(torch.nn.Module):
    def __init__(self, h):
        super(Generator_FastSpeech2s, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(Conv1d(256, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        #################################################
        self.expand_model = ExpandFrame()
        self.up_cum = [1]
        for i, u in enumerate(h.upsample_rates):
            self.up_cum.append(self.up_cum[-1] * u)

        # variance adaptor part
        self.ss_hidden = nn.ModuleList()
        self.lns = nn.ModuleList()
        k_size = [3, 5, 7, 11, 13]
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**i)
            self.ss_hidden.append(Conv1d(256, ch, k_size[i], 1, padding=(k_size[i]-1)//2))
            self.lns.append(nn.LayerNorm(ch))
        
        self.relu = nn.ReLU()
        self.mish = Mish()
        self.dropout = nn.Dropout(0.1)

        self.max_seq_len = 1000
        self.d_model = 256
        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)
        #################################################

    def forward(self, x, hidden, indices=None):
        starting_idx = 3 # 0(Vanila), 1, ...
        # x = hidden[0]

        batch_size, max_len = x.shape[0], x.shape[1]
        # poistion encoding
        if x.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(x.shape[1], self.d_model)[:x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        x = x + position_embedded
        
        # print("position_em: ", position_embedded.shape)
        if (indices == None): # inference
            x = torch.transpose(x, 1, 2)
        else:
            x = torch.transpose(torch.gather(x, 1, indices), 1, 2)
            max_len = 32

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            
            # position encoding
            if (i > 0) and (i < (self.num_upsamples - starting_idx + 1)):
                h = hidden[i-1] + position_embedded
                if (indices != None): # train
                    h = torch.gather(h, 1, indices)

                #################################################   
                l = torch.Tensor([[self.up_cum[i] for _ in range(max_len)] for _ in range(batch_size)])
                # print(l)
                l = l.to(x.device)
                h = self.expand_model(h, l)
                #################################################   
                h = self.ss_hidden[i](h)
                # h = self.relu(h).transpose(1,2)
                h = self.mish(h).transpose(1,2)
                h = self.dropout(self.lns[i](h)).transpose(1,2)            
                x = x + h
            
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)