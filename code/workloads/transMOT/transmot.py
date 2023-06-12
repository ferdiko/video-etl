import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transMOT.pygcn import GCN
from typing import Optional, Any, Union, Callable
from torch import Tensor


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

    
    def forward(self, tgt: Tensor, memory: Tensor, first_layer = False,
                repeat_num = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, debug = False) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        # print("(DecoderLayer) tgt shape:", x.shape)
        # print("(DecoderLayer) enc_output shape:", memory.shape)
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        if first_layer:
            x = x.repeat(1, repeat_num, 1)
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x
    

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

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
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
            #print("output:", output.shape)
            repeat_num = memory.shape[1]
            output = mod(output, memory, i == 0, repeat_num, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            if i == 0:
                if tgt_key_padding_mask is not None:
                    tgt_key_padding_mask = tgt_key_padding_mask.repeat(repeat_num, 1)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.repeat(repeat_num, 1, 1)
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
    def __init__(self, input_size: int = 4100, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransMOT, self).__init__()
        self.input_size = input_size
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
        self.embedding = nn.Linear(input_size, d_model)
        spatial_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, 
                                                           batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_encoder_layer, num_encoder_layers)
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                           batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_encoder_layers)
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
        #enc_mask = enc_mask.to(torch.bool)
        #new_enc_mask = enc_mask[torch.sum(enc_mask, dim=1) != 0]
        #new_enc_mask = ~new_enc_mask
        #enc_feature = enc_feature[torch.sum(enc_mask, dim=1) != 0]
        #enc_edge_weight = enc_edge_weight[torch.sum(enc_mask, dim=1) != 0]
        enc_edge_weight = enc_edge_weight.repeat(self.nhead, 1, 1)
        spatial_enc_output = self.spatial_encoder(enc_feature, mask=enc_edge_weight, src_key_padding_mask=~enc_mask)  # , enc_edge_weight)
        #print("spatial_enc_output", spatial_enc_output)
        #  print("spatial_enc_output_shape", spatial_enc_output.shape)
        # print("spatial_enc_output", spatial_enc_output)
        enc_mask = enc_mask.transpose(0, 1)
        spatial_enc_output = spatial_enc_output.transpose(0, 1)
        new_enc_mask = enc_mask[torch.sum(enc_mask, dim=1) != 0]
        new_enc_mask = ~new_enc_mask
        spatial_enc_output = spatial_enc_output[torch.sum(enc_mask, dim=1) != 0]

        # print(new_enc_mask)

        temporal_enc_output = self.temporal_encoder(spatial_enc_output, src_key_padding_mask=new_enc_mask)
        # temporal_enc_output = temporal_enc_output.masked_fill(torch.isnan(temporal_enc_output), 0)
        # print(enc_mask)
        #print("temporal_enc_output", temporal_enc_output.shape)
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
        #print("dec_mask", dec_mask)
        dec_feature = F.relu(self.embedding(dec_feature))
        dec_feature = (dec_feature.permute(2, 0, 1) * dec_mask).permute(1, 2, 0)
        # print("dec_feature", dec_feature)
        # dec_feature = dec_feature.transpose(0, 1)
        # if debug:
        #   print("dec_feature", dec_feature)
        # print(dec_mask.shape)
        #print(dec_feature.shape, temporal_enc_output.shape)
        dec_mask = dec_mask
        #spatial_enc_output = spatial_enc_output.transpose(0, 1)
        new_dec_mask = dec_mask[:, dec_mask[0] != 0]
        dec_feature = dec_feature[:, dec_mask[0] != 0]
        dec_edge_weight = dec_edge_weight[:, :int(torch.sum(dec_mask[0] != 0)), :int(torch.sum(dec_mask[0] != 0))]
        new_dec_mask = ~new_dec_mask
        
        dec_edge_weight = dec_edge_weight.repeat(self.nhead, 1, 1)
        output = self.decoder(dec_feature.transpose(0, 1), temporal_enc_output, tgt_mask=dec_edge_weight, memory_key_padding_mask=new_enc_mask,
                           tgt_key_padding_mask=new_dec_mask)
        # if debug:
        #print("decoder_out", output.shape)
        return output

    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

