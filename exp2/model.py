import torch
import torch.nn as nn
class FeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_hidden = config['d_hidden']
        dropout = config['dropout']
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        n_head = config['n_head']
        dropout = config['dropout']
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNet(config)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x, x, x, need_weights=False)[0]  # (B, d_model , C)
        x = x + self.dropout1(attn_out)
        x = self.ln1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.ln2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_features = config['d_features']
        d_model = config['d_model']
        seq_len = config['seq_len']
        n_layers = config['n_layers']

        self.linear = nn.Linear(in_features=d_features, out_features=d_model)
        self.position_embedding_table = nn.Embedding(seq_len, d_model)
        self.blocks = nn.Sequential(*[BlockEncoder(config) for _ in range(n_layers)])

    def forward(self, tar):
        B, L, C = tar.shape
        tar = self.linear(tar)  # (B, L, d_model)
        # print(tar.shape)
        pos_ids = torch.arange(L, device=tar.device)
        pos_embd = self.position_embedding_table(pos_ids)
        # print(pos_embd.shape)
        tar = tar + pos_embd
        tar = self.blocks(tar)

        return tar

class BlockDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        d_model = config['d_model']
        n_head = config['n_head']
        dropout = config['dropout']

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNet(config)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tar, output_ecd, reason_mask, nan_mask):
        self_attn_out = self.self_attn(tar, tar, tar,
                                       attn_mask=reason_mask,
                                       key_padding_mask=nan_mask,
                                       need_weights=False)[0]
        tar = tar + self.dropout1(self_attn_out)
        tar = self.ln1(tar)

        cross_attn_out = self.cross_attn(tar, output_ecd, output_ecd, need_weights=False)[0]
        tar = tar + self.dropout2(cross_attn_out)
        tar = self.ln2(tar)

        ffn_out = self.ffn(tar)
        tar = tar + self.dropout3(ffn_out)
        tar = self.ln3(tar)

        return tar


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        n_layers = config['n_layers']
        seq_len = config['seq_len']

        self.linear_in = nn.Linear(in_features=1, out_features=d_model)
        self.position_embedding_table = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([BlockDecoder(config) for _ in range(n_layers)])
        self.linear_out = nn.Linear(in_features=d_model, out_features=1)
        self.register_buffer('reason_mask', nn.Transformer.generate_square_subsequent_mask(seq_len).isinf())

    def forward(self, tar, output_ecd):
        B, L, C = tar.shape
        nan_mask = torch.isnan(tar).squeeze(-1)
        tar = torch.nan_to_num(tar, nan=0.0)

        tar = self.linear_in(tar)
        # print(tar.shape)
        pos_embd = self.position_embedding_table(torch.arange(L).to(tar.device))
        # print(pos_embd.shape)
        tar = tar + pos_embd

        for block in self.blocks:
            tar = block(tar, output_ecd,
                        reason_mask=self.reason_mask,
                        nan_mask=nan_mask)
        tar = self.linear_out(tar)
        return tar


class RR_Former(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.tar_len = config['tar_len']

    def forward(self, input_ecd, input_dcd):
        output_ecd = self.encoder(input_ecd)
        output_dcd = self.decoder(tar=input_dcd, output_ecd=output_ecd)

        return output_dcd[:, -self.tar_len:]
