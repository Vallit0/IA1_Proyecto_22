import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(input_dim, embed_dim)
        self.decoder_embedding = nn.Embedding(output_dim, embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src, tgt):
        src_emb = self.encoder_embedding(src)
        tgt_emb = self.decoder_embedding(tgt)

        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)

        return self.fc_out(output)
