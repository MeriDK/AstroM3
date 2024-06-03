import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Informer import DataEmbedding, EncoderLayer, AttentionLayer, ProbAttention, Encoder


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    def __init__(self, enc_in=144, d_model=512, dropout=0.1, factor=1, output_attention=False, n_heads=8, d_ff=2048,
                 activation='gelu', e_layers=2):
        super(Informer, self).__init__()

        self.enc_embedding = DataEmbedding(enc_in, d_model)

        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                    d_model,
                    n_heads
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation
            ) for _ in range(e_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=torch.nn.LayerNorm(d_model))

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        enc_out = self.dropout(enc_out)
        output = enc_out * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        return output


class GalSpecNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(GalSpecNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3), nn.ReLU())
        self.mp1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3), nn.ReLU())
        self.mp2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=3), nn.ReLU())
        self.mp3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)

        return x


class MetaModel(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, dropout=0.5):
        super(MetaModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)

        return x


class ModelV0(nn.Module):
    def __init__(self, config):
        super(ModelV0, self).__init__()

        self.photometry_encoder = Informer(
            enc_in=config['p_feature_size'], d_model=config['p_d_model'], dropout=config['p_dropout'], factor=1,
            output_attention=False, n_heads=config['p_n_heads'], d_ff=config['p_d_ff'],
            activation='gelu', e_layers=config['p_encoder_layers']
        )
        self.spectra_encoder = GalSpecNet(dropout=config['s_dropout'])
        self.metadata_encoder = MetaModel(hidden_dim=config['s_hidden_dim'], dropout=config['s_dropout'])

        self.photometry_proj = nn.Linear(config['seq_len'] * config['p_d_model'], config['hidden_dim'])
        self.spectra_proj = nn.Linear(1184, config['hidden_dim'])
        self.metadata_proj = nn.Linear(config['m_hidden_dim'], config['hidden_dim'])

        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, el1, el2):
        photometry1, photometry_mask1, spectra1, metadata1 = el1
        photometry2, photometry_mask2, spectra2, metadata2 = el2

        p_emb1 = self.photometry_proj(self.photometry_encoder(photometry1, photometry_mask1))
        p_emb2 = self.photometry_proj(self.photometry_encoder(photometry2, photometry_mask2))
        s_emb1 = self.spectra_proj(self.spectra_encoder(spectra1))
        s_emb2 = self.spectra_proj(self.spectra_encoder(spectra2))
        m_emb1 = self.metadata_proj(self.metadata_encoder(metadata1))
        m_emb2 = self.metadata_proj(self.metadata_encoder(metadata2))

        ps_sim = self.cos(p_emb1, s_emb2)
        mp_sim = self.cos(m_emb1, p_emb2)
        sm_sim = self.cos(s_emb1, m_emb2)

        return ps_sim, mp_sim, sm_sim
