import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transMOT.pygcn import GCN
from typing import Optional, Any, Union, Callable
from torch import Tensor


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, head_num):
        super(GraphMultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.head_size = input_dim // head_num
        self.input_dim = input_dim
        self.out_dim = self.input_dim
        assert self.head_size * head_num == self.input_dim, "embed_dim must be divisible by num_heads"
        self.WQ = nn.Linear(self.input_dim, self.out_dim)
        self.WK = nn.Linear(self.input_dim, self.out_dim)
        self.WV = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, seq_input, w_X=None, mask=None):
        batch_size, seq_len, embedding_size = seq_input.size()
        Q = self.WQ(seq_input).view(batch_size, seq_len, self.head_num, self.head_size)
        Q = Q.permute(0, 2, 1, 3)  # bs, hn, sl, hs
        K = self.WK(seq_input).reshape(batch_size, seq_len, self.head_num, self.head_size)
        K = K.permute(0, 2, 1, 3)
        V = self.WV(seq_input).reshape(batch_size, seq_len, self.head_num, self.head_size)
        V = V.permute(0, 2, 1, 3)
        sim = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_size ** 0.5  # bs, hn, sl, sl
        # print(sim)
        sim = F.softmax(sim, dim=3)
        if w_X is not None:
            w_X = w_X.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            sim = sim * w_X
        if mask is not None:
            temp_mask = mask.unsqueeze(1).repeat(1, self.head_num, 1)
            sim = (sim.permute(3, 0, 1, 2) * temp_mask)
            # sim = sim / torch.sum(sim, dim=0)
            sim = sim.permute(1, 2, 3, 0)
        new_V = torch.matmul(sim, V)  # bs, hn, sl, input_dim

        new_V = new_V.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.input_dim)
        return new_V

    """
    def forward(self, seq_input, w_X=None, mask=None):
        batch_size, seq_len, embedding_size = seq_input.size()
        Q = self.WQ(seq_input).view(batch_size, seq_len, self.head_num, self.head_size)
        Q = Q.permute(0, 2, 1, 3)  # bs, hn, sl, hs
        K = self.WK(seq_input).reshape(batch_size, seq_len, self.head_num, self.head_size)
        K = K.permute(0, 2, 1, 3)
        V = seq_input.unsqueeze(1).repeat(1, self.head_num, 1, 1)  # bs, hn, sl, input_dim
        sim = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_size ** 0.5  # bs, hn, sl, sl
        if w_X is not None:
            w_X = w_X.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            sim = sim * w_X
        sim = F.softmax(sim, dim=3)
        if mask is not None:
            temp_mask = mask.unsqueeze(1).repeat(1, self.head_num, 1)
            sim = (sim.permute(3, 0, 1, 2) * temp_mask)
            #sim = sim / torch.sum(sim, dim=0)
            sim = sim.permute(1, 2, 3, 0)
        new_V = torch.matmul(sim, V)  # bs, hn, sl, input_dim

        new_V = torch.mean(new_V, dim=1) # bs, sl, input_dim
        #new_V = new_V.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.input_dim)
        return new_V
    """


class SpatialEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpatialEncoderLayer, self).__init__()
        # self.self_attn = GraphMultiHeadAttention(d_model, nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SpatialEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    """
    def forward(self, src: Tensor, x_W: Tensor, mask: Tensor) -> Tensor:

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src

        x = self.norm1(x + self._sa_block(x, x_W, mask))
        x = self.norm2(x + self._ff_block(x))

        return x
    """

    # self-attention block
    # def _sa_block(self, x: Tensor, w_X: Tensor, mask: Tensor) -> Tensor:
    #   x = self.self_attn(x, w_X, mask)
    #  return self.dropout1(x)
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SpatialEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SpatialEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    """
    def forward(self, src: Tensor, x_W: Tensor, mask: Tensor = None) -> Tensor:
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(output, x_W, mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    """


class TemporalEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TemporalEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # print(x.shape, key_padding_mask.shape)
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TemporalEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TemporalEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DecoderLayer, self).__init__()
        # self.self_attn = GraphMultiHeadAttention(d_model, nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    """
    def forward(self, tgt: Tensor, memory: Tensor, w_X: Tensor, first_layer = False,
                repeat_num = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, debug = False) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        # print("(DecoderLayer) tgt shape:", x.shape)
        # print("(DecoderLayer) enc_output shape:", memory.shape)
        if not first_layer:
            x = x.transpose(0, 1)
        x = self.norm1(x + self._sa_block(x, w_X, tgt_key_padding_mask))
        if first_layer:
            x = x.repeat(repeat_num, 1, 1)
        x = x.transpose(0, 1)
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x
    """

    # self-attention block
    # def _sa_block(self, x: Tensor, w_X: Tensor, mask: Tensor) -> Tensor:
    #   x = self.self_attn(x, w_X, mask)
    #  return self.dropout1(x)
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # print(x.shape, mem.shape)
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, d_model, norm=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.projection = nn.Linear(self.d_model, 1, bias=False)  # vocab_size

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, w_X: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, debug=False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt  # .transpose(0, 1)

        for i, mod in enumerate(self.layers):
            # print("output:", output.shape)
            repeat_num = memory.shape[1]
            output = mod(output, memory, w_X, i == 0, repeat_num, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            # if debug:
            #   print(f"decoder_output_{i}:", output)

        if self.norm is not None:
            output = self.norm(output)

        output = self.projection(output)
        output = torch.squeeze(output, axis=2)
        # output = F.softmax(output, dim=1)
        # print("(TransformerDecoder) output shape after softmax:", output.shape)
        return output


class TransMOT(nn.Module):
    def __init__(self, input_size: int = 4100, T: int = 5, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransMOT, self).__init__()
        self.input_size = input_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Linear(input_size, d_model)
        spatial_encoder_layer = SpatialEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.spatial_encoder = SpatialEncoder(spatial_encoder_layer, num_encoder_layers)
        temporal_encoder_layer = TemporalEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.temporal_encoder = TemporalEncoder(temporal_encoder_layer, num_encoder_layers)
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, d_model)

        self.d_model = d_model
        self.nhead = nhead

    def reformulate_feature(self, feature):
        bbox = feature[:, :, -4:]
        result_size = self.input_size - 4
        if result_size == 0:
            return bbox
        image_feature = feature[:, :, :-4]
        T, N, M = image_feature.shape
        image_feature = image_feature.view(T, N, M // result_size, result_size)
        image_feature = torch.mean(image_feature, dim=2)
        return torch.cat((bbox, image_feature), dim=2)

    def forward(self, enc_feature, enc_edge_weight, enc_mask, dec_feature, dec_edge_weight, dec_mask, debug=False):
        # print("==================================================")
        # if debug:
        #   print("enc_feature_before", enc_feature)
        enc_feature = self.reformulate_feature(enc_feature)
        enc_feature = F.relu(self.embedding(enc_feature))
        enc_feature = (enc_feature.permute(2, 0, 1) * enc_mask).permute(1, 2, 0)
        # print("enc_feature", enc_feature)
        # if debug:
        # print("enc_feature", enc_feature)
        spatial_enc_output = self.spatial_encoder(enc_feature, enc_edge_weight, )
        # if debug:
        #   print("spatial_enc_output", spatial_enc_output)
        #  print("spatial_enc_output_shape", spatial_enc_output.shape)
        # print("spatial_enc_output", spatial_enc_output)
        enc_mask = enc_mask.to(torch.bool).transpose(0, 1)
        spatial_enc_output = spatial_enc_output.transpose(0, 1)
        new_enc_mask = enc_mask[torch.sum(enc_mask, dim=1) != 0]
        new_enc_mask = ~new_enc_mask
        spatial_enc_output = spatial_enc_output[torch.sum(enc_mask, dim=1) != 0]

        # print(new_enc_mask)

        temporal_enc_output = self.temporal_encoder(spatial_enc_output, src_key_padding_mask=new_enc_mask)
        # temporal_enc_output = temporal_enc_output.masked_fill(torch.isnan(temporal_enc_output), 0)
        # print(enc_mask)
        # print(temporal_enc_output)
        # if debug:
        #   print("temporal_enc_output_shape", temporal_enc_output.shape)
        temporal_enc_output = temporal_enc_output.transpose(0, 1)
        # padding the virtual source
        temporal_enc_output = F.pad(temporal_enc_output, (0, 0, 1, 0), 'constant', value=1)
        # print(temporal_enc_output.shape)
        new_enc_mask = F.pad(new_enc_mask, (0, 0, 1, 0), 'constant', value=False)
        # print(enc_mask.shape)
        # print(enc_mask)
        # if debug:
        #   print("temporal_enc_output", temporal_enc_output)
        dec_feature = self.reformulate_feature(dec_feature)
        # print("dec_feature0", dec_feature)
        # print("dec_mask", dec_mask)
        dec_feature = F.relu(self.embedding(dec_feature))
        dec_feature = (dec_feature.permute(2, 0, 1) * dec_mask).permute(1, 2, 0)
        # print("dec_feature", dec_feature)
        # dec_feature = dec_feature.transpose(0, 1)
        # if debug:
        #   print("dec_feature", dec_feature)
        # print(dec_mask.shape)
        out = self.decoder(dec_feature, temporal_enc_output, dec_edge_weight, memory_key_padding_mask=new_enc_mask,
                           tgt_key_padding_mask=dec_mask, debug=debug)
        # if debug:
        # print("decoder_out", out)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))