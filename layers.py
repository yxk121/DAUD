import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ==================== General ====================
class MLP(nn.Module):
    def __init__(self, input_size, output_size, mode="encode", dropout=0.0, n_layers=1):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        layers = []

        if mode == "encode": 
            if n_layers == 1:
                hidden_size = int(input_size / 2)
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                prev_size = input_size
                for _ in range(n_layers):
                    next_size = max(int(prev_size / 4), output_size)
                    layers.append(nn.Linear(prev_size, next_size))
                    layers.append(nn.LeakyReLU())
                    layers.append(nn.Dropout(p=dropout))
                    prev_size = next_size
                layers.append(nn.Linear(prev_size, output_size))
        elif mode == "project": 
            if n_layers == 1:
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.Dropout(p=dropout))
            else:
                print("MLP mode project only supports 1 layer!")
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.Dropout(p=dropout))
        elif mode == "norm_project": 
            if n_layers == 1:
                layers.append(nn.LayerNorm(input_size))
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.Dropout(p=dropout))
            else:
                print("MLP mode fusion only supports 1 layer!")
                layers.append(nn.LayerNorm(input_size))
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.Dropout(p=dropout))

        self.network = nn.Sequential(*layers)

        for lyr in self.network:
            if isinstance(lyr, nn.Linear):
                nn.init.xavier_normal_(lyr.weight)
                nn.init.zeros_(lyr.bias)

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        S, N, E = x.size()
        if E != self.d_model:
            raise ValueError(f"d_model mismatch: x.size(-1)={E}, expected {self.d_model}")
        x = x + self.pe[:S, :, :] 
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor, padding_mask: Tensor) -> Tensor:
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        if padding_mask is not None:
            if padding_mask.size(0) != padding_mask.size(1):
                if padding_mask.size(0) == src.size(0) and padding_mask.size(1) == src.size(1):
                    raise ValueError(
                        f"[Mask shape error] Detected padding_mask of shape {padding_mask.shape}, "
                        f"which looks like (seq_len, batch). "
                        f"Expected shape is (batch, seq_len). "
                        f"Please fix dimension order (transpose has been deleted in Transformer)."
                    )
            output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        else:
            output = self.transformer_encoder(src)
        output = output.transpose(0, 1)
        return output


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):   
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn


class SelfAttentionFeatureExtract(nn.Module):
    def __init__(self, multi_head_num, input_size, output_size):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
        self.out_layer = nn.Linear(input_size, output_size)

    def forward(self, inputs, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=inputs,
                                       value=inputs,
                                       key=inputs,
                                       mask=mask
                                       )
        feature = feature.contiguous().view([-1, feature.size(-1)])
        out = self.out_layer(feature)
        return out, attn


class CoAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, out_dim,
                 attn_dropout=0.0, out_dropout=0.0, layer_norm=False):
        super(CoAttention, self).__init__()
        self.Wq = nn.Parameter(torch.empty(query_dim, hidden_dim))
        self.Wk = nn.Parameter(torch.empty(key_dim,  hidden_dim))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        self.out_q = nn.Linear(2 * hidden_dim, out_dim)
        self.out_k = nn.Linear(2 * hidden_dim, out_dim)
        self.out_proj = nn.Linear(2 * out_dim, out_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(query_dim)   

    def forward(self, query, key, query_mask=None, key_mask=None):
        key_pad, query_pad = None, None
        B, Q_len, _ = query.shape
        _, K_len, _ = key.shape

        if self.layer_norm:
            query = self.ln(query)
            key = self.ln(key)

        Qh = torch.matmul(query, self.Wq)
        Kh = torch.matmul(key, self.Wk)
        scores = torch.matmul(Qh, Kh.transpose(1, 2)) / (Qh.size(-1) ** 0.5)
        if key_mask is not None:
            key_pad = (key_mask == 0).to(torch.bool).to(scores.device)
        if query_mask is not None:
            query_pad = (query_mask == 0).to(torch.bool).to(scores.device)

        if key_pad is not None:
            mask_value = -1e9
            scores = scores.masked_fill(key_pad.unsqueeze(1), mask_value)
            all_keys_pad = key_pad.all(dim=1)
            if all_keys_pad.any():
                scores[all_keys_pad] = 0.0
        A_q = F.softmax(scores, dim=-1)
        A_q = self.attn_dropout(A_q)
        if query_pad is not None:
            A_q = A_q.masked_fill(query_pad.unsqueeze(2), 0.0)
        C_q = torch.matmul(A_q, Kh)
        if query_pad is not None:
            C_q = C_q.masked_fill(query_pad.unsqueeze(2), 0.0)
        scores_T = scores.transpose(1, 2)
        if query_pad is not None:
            scores_T = scores_T.masked_fill(query_pad.unsqueeze(1), -1e9)
            all_queries_pad = query_pad.all(dim=1)
            if all_queries_pad.any():
                scores_T[all_queries_pad] = 0.0
        A_k = F.softmax(scores_T, dim=-1)
        A_k = self.attn_dropout(A_k)
        if key_pad is not None:
            A_k = A_k.masked_fill(key_pad.unsqueeze(2), 0.0)
        C_k = torch.matmul(A_k, Qh)
        if key_pad is not None:
            C_k = C_k.masked_fill(key_pad.unsqueeze(2), 0.0)
        Q_out = self.out_q(torch.cat([Qh, C_q], dim=-1))
        K_out = self.out_k(torch.cat([Kh, C_k], dim=-1))

        attended_output = torch.cat([Q_out, K_out], dim=-1)
        output = self.out_proj(attended_output)
        output = self.out_dropout(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, out_dim,
                 attn_dropout=0.0, out_dropout=0.0, layer_norm=False):
        super(CrossAttention, self).__init__()
        # Linear projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(query_dim)  

    def forward(self, query, key, query_mask=None, key_mask=None):
        key_pad, query_pad = None, None
        if self.layer_norm:
            query = self.ln(query)
            key = self.ln(key)

        Q = self.query_proj(query)   # (B, Q_len, hidden)
        K = self.key_proj(key)       # (B, K_len, hidden)
        V = self.value_proj(key)   # (B, K_len, hidden)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # (B, Q_len, K_len)
        if key_mask is not None:
            key_pad = (key_mask == 0).to(torch.bool).to(scores.device)  # (B, K_len)
        if query_mask is not None:
            query_pad = (query_mask == 0).to(torch.bool).to(scores.device)  # (B, Q_len)

        if key_pad is not None:
            mask_value = -1e9
            scores = scores.masked_fill(key_pad.unsqueeze(1), mask_value)
            all_keys_pad = key_pad.all(dim=1)
            if all_keys_pad.any():
                scores[all_keys_pad] = 0.0
        attn_weights = F.softmax(scores, dim=-1)  # (B, Q_len, K_len)
        if query_pad is not None:
            attn_weights = attn_weights.masked_fill(query_pad.unsqueeze(2), 0.0)
        attn_weights = self.attn_dropout(attn_weights)
        attended_output = torch.bmm(attn_weights, V)  # (B, Q_len, hidden)
        output = self.out_proj(attended_output)  # (B, Q_len, out_dim)
        output = self.out_dropout(output)
        return output


# ==================== Disentangling ====================
class Disentangler(nn.Module):
    def __init__(self, viewed_input_dim, mask1_hidden_dim, mask2_hidden_dim):
        super(Disentangler, self).__init__()
        self.query_layer_vr = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)
        self.key_layer_vr = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)
        self.value_layer_vr = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)
        self.query_layer_vi = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)
        self.key_layer_vi = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)
        self.value_layer_vi = torch.nn.Linear(viewed_input_dim, mask1_hidden_dim)

        self.output_layer_vr = torch.nn.Linear(mask1_hidden_dim, viewed_input_dim)
        self.output_layer_vi = torch.nn.Linear(mask1_hidden_dim, viewed_input_dim)

        self.query_layer_dsf = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)
        self.key_layer_dsf = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)
        self.value_layer_dsf = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)
        self.query_layer_dsh = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)
        self.key_layer_dsh = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)
        self.value_layer_dsh = torch.nn.Linear(viewed_input_dim, mask2_hidden_dim)

        self.output_layer_dsf = torch.nn.Linear(mask2_hidden_dim, viewed_input_dim)
        self.output_layer_dsh = torch.nn.Linear(mask2_hidden_dim, viewed_input_dim)

        self.relu = torch.nn.LeakyReLU()
        self.softplus = torch.nn.Softplus()

        self.fc_before_v_masking = nn.Linear(viewed_input_dim, viewed_input_dim)
        self.fc_before_d_masking = nn.Linear(viewed_input_dim, viewed_input_dim)

        nn.init.xavier_uniform_(self.query_layer_vr.weight)
        nn.init.zeros_(self.query_layer_vr.bias)
        nn.init.xavier_uniform_(self.key_layer_vr.weight)
        nn.init.zeros_(self.key_layer_vr.bias)
        nn.init.xavier_uniform_(self.value_layer_vr.weight)
        nn.init.zeros_(self.value_layer_vr.bias)
        nn.init.xavier_uniform_(self.query_layer_vi.weight)
        nn.init.zeros_(self.query_layer_vi.bias)
        nn.init.xavier_uniform_(self.key_layer_vi.weight)
        nn.init.zeros_(self.key_layer_vi.bias)
        nn.init.xavier_uniform_(self.value_layer_vi.weight)
        nn.init.zeros_(self.value_layer_vi.bias)

        nn.init.xavier_uniform_(self.output_layer_vi.weight)
        nn.init.zeros_(self.output_layer_vi.bias)
        nn.init.xavier_uniform_(self.output_layer_vr.weight)
        nn.init.zeros_(self.output_layer_vr.bias)

        nn.init.xavier_uniform_(self.query_layer_dsf.weight)
        nn.init.zeros_(self.query_layer_dsf.bias)
        nn.init.xavier_uniform_(self.key_layer_dsf.weight)
        nn.init.zeros_(self.key_layer_dsf.bias)
        nn.init.xavier_uniform_(self.value_layer_dsf.weight)
        nn.init.zeros_(self.value_layer_dsf.bias)

        nn.init.xavier_uniform_(self.query_layer_dsh.weight)
        nn.init.zeros_(self.query_layer_dsh.bias)
        nn.init.xavier_uniform_(self.key_layer_dsh.weight)
        nn.init.zeros_(self.key_layer_dsh.bias)
        nn.init.xavier_uniform_(self.value_layer_dsh.weight)
        nn.init.zeros_(self.value_layer_dsh.bias)

        nn.init.xavier_uniform_(self.output_layer_dsf.weight)
        nn.init.zeros_(self.output_layer_dsf.bias)
        nn.init.xavier_uniform_(self.output_layer_dsh.weight)
        nn.init.zeros_(self.output_layer_dsh.bias)

        nn.init.xavier_uniform_(self.fc_before_v_masking.weight)
        nn.init.zeros_(self.fc_before_v_masking.bias)
        nn.init.xavier_uniform_(self.fc_before_d_masking.weight)
        nn.init.zeros_(self.fc_before_d_masking.bias)

    def forward(self, x, sentence_num_padding_mask):
        co_attn_feature = x.squeeze(1)
        vr_mask, vi_mask = self.veracity_attn_mask(co_attn_feature, sentence_num_padding_mask)
        mapped_co_attn_feature = self.relu(self.fc_before_v_masking(co_attn_feature))
        veracity_relevant_feature = torch.multiply(mapped_co_attn_feature, vr_mask)
        veracity_irrelevant_feature = torch.multiply(mapped_co_attn_feature, vi_mask)

        dsh_mask, dsf_mask = self.domain_attn_mask(veracity_relevant_feature, sentence_num_padding_mask)
        mapped_veracity_relevant_feature = self.relu(self.fc_before_d_masking(veracity_relevant_feature))
        domain_shared_feature = torch.multiply(mapped_veracity_relevant_feature, dsh_mask)
        domain_specific_feature = torch.multiply(mapped_veracity_relevant_feature, dsf_mask)

        return veracity_relevant_feature, veracity_irrelevant_feature, domain_shared_feature, domain_specific_feature

    def veracity_attn_mask(self, co_attn_feature, sentence_num_padding_mask):
        query1 = self.query_layer_vr(co_attn_feature)
        key1 = self.key_layer_vr(co_attn_feature)
        value1 = self.value_layer_vr(co_attn_feature)
        attn_output1 = self.attn_layer(query1, key1, value1, mask=sentence_num_padding_mask)
        x1 = self.relu(self.output_layer_vr(attn_output1))

        query2 = self.query_layer_vi(co_attn_feature)
        key2 = self.key_layer_vi(co_attn_feature)
        value2 = self.value_layer_vi(co_attn_feature)
        attn_output2 = self.attn_layer(query2, key2, value2, mask=sentence_num_padding_mask)
        x2 = self.relu(self.output_layer_vi(attn_output2))

        if sentence_num_padding_mask is not None:
            msk1 = torch.sigmoid(x1) * sentence_num_padding_mask
            msk2 = torch.sigmoid(x2) * sentence_num_padding_mask
        else:
            msk1 = torch.sigmoid(x1)
            msk2 = torch.sigmoid(x2)
        return msk1, msk2

    def domain_attn_mask(self, vr_feature, sentence_num_padding_mask):
        query1 = self.query_layer_dsh(vr_feature)
        key1 = self.key_layer_dsh(vr_feature)
        value1 = self.value_layer_dsh(vr_feature)
        attn_output1 = self.attn_layer(query1, key1, value1, mask=sentence_num_padding_mask)
        x1 = self.relu(self.output_layer_dsh(attn_output1))

        query2 = self.query_layer_dsf(vr_feature)
        key2 = self.key_layer_dsf(vr_feature)
        value2 = self.value_layer_dsf(vr_feature)
        attn_output2 = self.attn_layer(query2, key2, value2, mask=sentence_num_padding_mask)
        x2 = self.relu(self.output_layer_dsf(attn_output2))

        if sentence_num_padding_mask is not None:
            msk1 = torch.sigmoid(x1) * sentence_num_padding_mask
            msk2 = torch.sigmoid(x2) * sentence_num_padding_mask
        else:
            msk1 = torch.sigmoid(x1)
            msk2 = torch.sigmoid(x2)
        return msk1, msk2

    def attn_layer(self, query, key, value, mask=None):
        # Scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        if mask is not None:
            attn_weights = F.softmax(scores, dim=-1) * mask
        else:
            attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super(Classifier, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.zeros_(self.hidden_layer1.bias)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.zeros_(self.hidden_layer2.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, in_x):
        x = self.hidden_layer1(in_x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x