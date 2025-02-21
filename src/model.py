import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from informer import DataEmbedding, EncoderLayer, AttentionLayer, ProbAttention, Encoder


class Informer(nn.Module, PyTorchModelHubMixin):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    def __init__(self, classification, num_classes, seq_len, enc_in, d_model, dropout, factor,
                 output_attention, n_heads, d_ff, activation, e_layers):
        super(Informer, self).__init__()

        self.classification = classification
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout
        self.factor = factor
        self.output_attention = output_attention
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers

        # Embedding Layer
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model)

        # Encoder with ProbSparse attention
        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                  output_attention=self.output_attention),
                    self.d_model,
                    self.n_heads
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(self.e_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=torch.nn.LayerNorm(self.d_model))
        self.dropout = nn.Dropout(self.dropout)

        # Classification head
        if self.classification:
            self.fc = nn.Linear(seq_len * d_model, num_classes)

    def forward(self, x_enc, x_mark_enc):
        """
        Forward pass for the Informer encoder.

        Args:
            x_enc (torch.Tensor): Input sequence.
            x_mark_enc (torch.Tensor): Mask for padding.

        Returns:
            torch.Tensor: Encoded features or classification logits.
        """
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out = self.dropout(enc_out)
        output = enc_out * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        if self.classification:
            output = self.fc(output)

        return output


class GalSpecNet(nn.Module, PyTorchModelHubMixin):
    """
    Spectra encoder using a 1D CNN architecture.
    Paper link: https://academic.oup.com/mnras/article/527/1/1163/7283157
    """
    def __init__(self, classification, num_classes, dropout_rate, conv_channels, kernel_size, mp_kernel_size):
        super(GalSpecNet, self).__init__()

        self.classification = classification
        self.layers = nn.ModuleList([])

        # Build 1D Convolutional layers with ReLU activation
        for i in range(len(conv_channels) - 1):
            self.layers.append(nn.Conv1d(conv_channels[i], conv_channels[i + 1], kernel_size=kernel_size))
            self.layers.append(nn.ReLU())

            # Apply MaxPooling to all except the last conv layer
            if i < len(conv_channels) - 2:
                self.layers.append(nn.MaxPool1d(kernel_size=mp_kernel_size))

        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        if self.classification:
            self.fc = nn.Linear(1184, num_classes)

    def forward(self, x):
        """
        Forward pass for the GalSpecNet encoder.

        Args:
            x (torch.Tensor): Input spectra.

        Returns:
            torch.Tensor: Encoded features or classification logits.
        """

        for layer in self.layers:
            x = layer(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)

        if self.classification:
            x = self.fc(x)

        return x


class MetaModel(nn.Module, PyTorchModelHubMixin):
    """
    Fully connected MLP encoder for metadata features.
    """
    def __init__(self, classification, num_classes, input_dim, hidden_dim, dropout):
        super(MetaModel, self).__init__()

        self.classification = classification
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Fully connected layers with ReLU activation and dropout
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Classification head
        if self.classification:
            self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for the MetaModel encoder.

        Args:
            x (torch.Tensor): Metadata features.

        Returns:
            torch.Tensor: Encoded features or classification logits.
        """
        x = self.model(x)

        if self.classification:
            x = self.fc(x)

        return x


class AstroM3(nn.Module, PyTorchModelHubMixin):
    """
    Multi-modal model for Astronomy data fusion.

    The model integrates three modalities:
    - Photometry (light curves) processed by Informer
    - Spectra processed by GalSpecNet (1D CNN)
    - Metadata processed by MetaModel (MLP)

    It supports both classification and contrastive learning.
    """
    def __init__(self, classification, num_classes, hidden_dim, fusion, seq_len, p_enc_in, p_d_model, p_dropout,
                 p_factor, p_output_attention, p_n_heads, p_d_ff, p_activation, p_e_layers, s_dropout, s_conv_channels,
                 s_kernel_size, s_mp_kernel_size, m_input_dim, m_hidden_dim, m_dropout):
        super(AstroM3, self).__init__()

        self.classification = classification

        # Encoders
        self.photometry_encoder = Informer(
            classification=False,
            num_classes=None,
            seq_len=seq_len,
            enc_in=p_enc_in,
            d_model=p_d_model,
            dropout=p_dropout,
            factor=p_factor,
            output_attention=p_output_attention,
            n_heads=p_n_heads,
            d_ff=p_d_ff,
            activation=p_activation,
            e_layers=p_e_layers
        )
        self.spectra_encoder = GalSpecNet(
            classification=False,
            num_classes=None,
            dropout_rate=s_dropout,
            conv_channels=s_conv_channels,
            kernel_size=s_kernel_size,
            mp_kernel_size=s_mp_kernel_size
        )
        self.metadata_encoder = MetaModel(
            classification=False,
            num_classes=None,
            input_dim=m_input_dim,
            hidden_dim=m_hidden_dim,
            dropout=m_dropout
        )

        # Projection layers
        self.photometry_proj = nn.Linear(seq_len * p_d_model, hidden_dim)
        self.spectra_proj = nn.Linear(1184, hidden_dim)
        self.metadata_proj = nn.Linear(m_hidden_dim, hidden_dim)

        # Scaling factors for contrastive loss
        self.logit_scale_ps = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_sm = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_mp = nn.Parameter(torch.log(torch.ones([]) * 100))

        # Classification head
        if self.classification:
            self.fusion = fusion
            in_features = hidden_dim * 3 if self.fusion == 'concat' else hidden_dim
            self.fc = nn.Linear(in_features, num_classes)

    def get_embeddings(self, photometry, photometry_mask, spectra, metadata):
        """
        Computes embeddings for photometry, spectra, and metadata.

        Args:
            photometry, photometry_mask, spectra, metadata: Input features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Normalized embeddings.
        """
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
        m_emb = self.metadata_proj(self.metadata_encoder(metadata))

        # Normalize embeddings
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)

        return p_emb, s_emb, m_emb

    def forward(self, photometry, photometry_mask, spectra, metadata):
        """
        Forward pass for classification or contrastive learning.

        Returns:
            - Classification: logits for class prediction.
            - Contrastive: similarity scores between modalities.
        """
        p_emb, s_emb, m_emb = self.get_embeddings(photometry, photometry_mask, spectra, metadata)

        if self.classification:
            if self.fusion == 'concat':
                emb = torch.cat((p_emb, s_emb, m_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + s_emb + m_emb) / 3
            else:
                raise NotImplementedError
            return self.fc(emb)
        else:
            logit_scale_ps = torch.clamp(self.logit_scale_ps.exp(), min=1, max=100)
            logit_scale_sm = torch.clamp(self.logit_scale_sm.exp(), min=1, max=100)
            logit_scale_mp = torch.clamp(self.logit_scale_mp.exp(), min=1, max=100)

            logits_ps = logit_scale_ps * p_emb @ s_emb.T
            logits_sm = logit_scale_sm * s_emb @ m_emb.T
            logits_mp = logit_scale_mp * m_emb @ p_emb.T

            return logits_ps, logits_sm, logits_mp
