import torch
import torch.nn as nn
import numpy as np

from StyleSpeech.text.symbols import symbols
import StyleSpeech.models.Constants as Constants
from StyleSpeech.models.Modules import Mish, LinearNorm, ConvNorm, Conv1dGLU, \
                    MultiHeadAttention, StyleAdaptiveLayerNorm, get_sinusoid_encoding_table
from StyleSpeech.models.VarianceAdaptor import VarianceAdaptor
from StyleSpeech.models.Loss import StyleSpeechLoss
from utils import get_mask_from_lengths


class Speech(nn.Module):
    ''' StyleSpeech '''
    def __init__(self, config):
        super(Speech, self).__init__()
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        
    def parse_batch(self, batch):
        sid = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        D = torch.from_numpy(batch["D"]).long().cuda()
        log_D = torch.from_numpy(batch["log_D"]).float().cuda()
        f0 = torch.from_numpy(batch["f0"]).float().cuda()
        energy = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        return sid, text, mel_target, D, log_D, f0, energy, src_len, mel_len, max_src_len, max_mel_len

    def forward(self, device, src_seq, src_len, mel_target, mel_len=None, 
                    d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None):
        src_mask = get_mask_from_lengths(src_len, device, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, device, max_mel_len) if mel_len is not None else None
        
        # Encoding
        encoder_output, src_embedded, _ = self.encoder(src_seq, src_mask)
        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                device, encoder_output, src_mask, mel_len, mel_mask, 
                        d_target, p_target, e_target, max_mel_len)
        
        mel_prediction, _, hidden_output = self.decoder(acoustic_adaptor_output, mel_mask)
        return mel_prediction, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len, acoustic_adaptor_output, hidden_output

    def inference(self, device, src_seq, src_len=None, max_src_len=None, return_attn=False):
        src_mask = get_mask_from_lengths(src_len, device, max_src_len)
        
        # Encoding
        encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, src_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)

        # Deocoding
        mel_output, dec_slf_attn, _ = self.decoder(acoustic_adaptor_output, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_output, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

    def get_criterion(self):
        return StyleSpeechLoss()

class Encoder(nn.Module):
    ''' Encoder '''
    def __init__(self, config, n_src_vocab=len(symbols)+1):
        super(Encoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.encoder_layer
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.decoder_hidden
        self.dropout = config.dropout

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_model, padding_idx=Constants.PAD)
        self.prenet = Prenet(self.d_model, self.d_model, self.dropout)

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)

    def forward(self, src_seq, mask):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # word embedding
        src_embedded = self.src_word_emb(src_seq)
        # prenet
        src_seq = self.prenet(src_embedded, mask)
        # position encoding
        if src_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = src_seq + position_embedded
        # fft blocks
        slf_attn = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, 
                mask=mask, 
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(enc_slf_attn)
        # last fc
        enc_output = self.fc_out(enc_output)
        return enc_output, src_embedded, slf_attn


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.decoder_layer
        self.d_model = config.decoder_hidden
        self.n_head = config.decoder_head
        self.d_k = config.decoder_hidden // config.decoder_head
        self.d_v = config.decoder_hidden // config.decoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.n_mel_channels
        self.dropout = config.dropout

        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, self.d_model)
        )

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_out)
        )

    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # prenet
        dec_embedded = self.prenet(enc_seq)
        # poistion encoding
        if enc_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        dec_output = dec_embedded + position_embedded
        # fft blocks
        slf_attn = []
        output = []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(dec_slf_attn)
            output.append(dec_output)
            
        # last fc
        dec_output = self.fc_out(dec_output)
        return dec_output, slf_attn, output

class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)

    def forward(self, input, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 =  ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output


class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output