import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        #Project features down to attention_dim
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        #Project decoder hidden state down to attention_dim
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        #Get a score for each image location
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        #Use softmax to get attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, hidden):
        #image_features shape: (batch, num_pixels, encoder_dim)
        #hidden shape: (batch, decoder_dim)
        att1 = self.encoder_att(image_features)
        att2 = self.decoder_att(hidden).unsqueeze(1)
        combined = self.relu(att1 + att2)
        att = self.full_att(combined).squeeze(2)
        alpha = self.softmax(att)
        context = (alpha.unsqueeze(2) * image_features).sum(dim=1)
        return context, alpha