import torch
import torch.nn as nn
from models.attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, attention_dim, encoder_dim, decoder_dim, dropout=0.5):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, attention_dim, decoder_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, image_features):
        mean_features = image_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_features))
        c = torch.tanh(self.init_c(mean_features))
        return h, c

    def forward(self, image_features, captions):
        batch_size = image_features.shape[0]
        vocab_size = self.vocab_size

        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(image_features)
        caption_length = captions.shape[1] - 1

        predictions = torch.zeros(batch_size, caption_length, vocab_size).to(image_features.device)
        alphas = torch.zeros(batch_size, caption_length, 49).to(image_features.device)

        for t in range(caption_length):
            context, alpha = self.attention(image_features, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            pred = self.fc(self.dropout(h))
            predictions[:, t, :] = pred
            alphas[:, t, :] = alpha

        return predictions, alphas