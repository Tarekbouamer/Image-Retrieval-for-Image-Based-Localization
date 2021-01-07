import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# pylint: disable=arguments-differ


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


def create_pad_mask(t, pad):
    mask = (t[:, :, 0] == pad).unsqueeze(-3)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)

        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]

        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output, t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)


class Transformer(nn.Module):
    def __init__(self,
                 i_vocab_size,
                 t_vocab_size,
                 n_layers=6,
                 hidden_size=256,
                 filter_size=2048,
                 dropout_rate=0.1,
                 share_target_embedding=True,
                 has_inputs=True,
                 src_pad_idx=1,
                 trg_pad_idx=1,
                 size=16):
        super(Transformer, self).__init__()

        torch.set_printoptions(precision=4)

        self.hidden_size = hidden_size

        self.filter_size = filter_size

        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.size = size

        self.t_emb_dropout = nn.Dropout(dropout_rate)

        self.decoder = Decoder(hidden_size, filter_size, dropout_rate, n_layers)

        patch_size = self.size * self.size

        self.t_vocab_embedding = nn.Linear(patch_size, self.hidden_size)
        nn.init.normal_(self.t_vocab_embedding.weight, mean=0, std=hidden_size ** -0.5)

        if has_inputs:

            if not share_target_embedding:
                self.i_vocab_embedding = nn.Linear(patch_size, self.hidden_size)
                nn.init.normal_(self.i_vocab_embedding.weight, mean=0, std=hidden_size ** -0.5)
            else:
                self.i_vocab_embedding = self.t_vocab_embedding

            self.i_emb_dropout = nn.Dropout(dropout_rate)

            self.encoder = Encoder(hidden_size, filter_size, dropout_rate, n_layers)

        # spatial positional encoding
        inv_spatial_timescales = self.get_inv_timescales(self.hidden_size)

        # channel positional encoding
        inv_channel_timescales = self.get_inv_timescales(self.filter_size)

        self.register_buffer('inv_spatial_timescales', inv_spatial_timescales)
        self.register_buffer('inv_channel_timescales', inv_channel_timescales)

    def get_inv_timescales(self, dim):

        num_timescales = dim // 2

        print("num_timescales", num_timescales)

        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))

        print("log_timescale_increment", log_timescale_increment)

        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        print("inv_timescales", inv_timescales)

        return inv_timescales

    def get_position_encoding(self, x):

        print("x", x.size())

        max_length = x.size()[1]
        print("max_length", max_length)

        position = torch.arange(max_length, dtype=torch.float32, device=x.device)

        print("position", position.unsqueeze(1).size())
        print("self.inv_timescales", self.inv_spatial_timescales.unsqueeze(0).size())

        scaled_time = position.unsqueeze(1) * self.inv_spatial_timescales.unsqueeze(0)
        print("scaled_time", scaled_time.size())

        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        print("signal", signal.size())

        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        print("signal", signal.size())
        import numpy as np
        np.savetxt('signal.np', signal.cpu().numpy(), fmt='%1.5e')

        signal = signal.view(1, max_length, self.hidden_size)
        print("signal", signal.size())

        print("signal", signal)

        return signal

    def encode(self, inputs, i_mask):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)
        print("input_embedded", input_embedded[0])

        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        print("input_embedded", input_embedded[0])

        print("emb_scale", self.emb_scale)

        input_embedded *= self.emb_scale
        print("input_embedded", input_embedded[0])
        print("input_embedded", input_embedded.size())

        input_embedded += self.get_position_encoding(inputs)
        print("input_embedded", input_embedded[0])


        #input()

        input_embedded = self.i_emb_dropout(input_embedded)
        print("input_embedded", input_embedded.size())

        return self.encoder(input_embedded, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask, cache=None):

        # Target embedding
        target_embedded = self.t_vocab_embedding(targets)
        print("target_embedded", target_embedded.size())

        print("t_mask", t_mask.size())

        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)
        print("target_embedded", target_embedded.size())

        # Shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask,
                                      t_self_mask, cache)
        # linear
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))

        return output

    def forward(self, inputs, targets):


        enc_output, i_mask = None, None

        # Size zero padding
        inputs = dynamic_padding(inputs, self.size)

        size = inputs.size()[0:2]
        inputs = inputs.view(size[0], size[1], -1)
        print("after padding inputs = ", inputs.size())

        # Channels --> Words
        i_mask = create_pad_mask(inputs, self.src_pad_idx)
        print("i_mask = ", i_mask.size())

        enc_output = self.encode(inputs, i_mask)
        print("enc_output = ", enc_output.size())

        return enc_output

        # Size zero padding
        #targets = dynamic_padding(targets, self.size)

        #size = targets.size()[0:2]
        #targets = targets.view(size[0], size[1], -1)
        #print("after padding target = ", targets.size())

        # Channels --> Words
        #t_mask = create_pad_mask(targets, self.trg_pad_idx)
        #print("t_mask = ", t_mask.size())

        #target_size = targets.size()[1]
        #print("target_size = ", target_size)

        #t_self_mask = create_trg_self_mask(target_size, device=targets.device)
        #print("t_self_mask_size = ", t_self_mask.size())
        #self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)


def dynamic_padding(inp, min_size):

    pad_size_W = min_size - inp.size()[-1]
    pad_size_H = min_size - inp.size()[-2]

    pad = (0, pad_size_W, 0, pad_size_H)

    padding = nn.ConstantPad2d(pad, 0)

    return padding(inp)