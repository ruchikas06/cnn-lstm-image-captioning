import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import load_image_list, parse_captions, build_vocabulary
from utils.dataset import FlickrDataset, collate_fn, transform

image_dir = 'data/images'
caption_file = 'data/captions/Flickr8k.token.txt'
train_file = 'data/captions/Flickr_8k.trainImages.txt'
val_file = 'data/captions/Flickr_8k.devImages.txt'

embed_dim = 256
attention_dim = 512
decoder_dim = 512
encoder_dim = 2048
dropout = 0.5
batch_size = 32
epochs = 20
learning_rate = 4e-4
lambda_reg = 1.0

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_images = load_image_list(train_file)
val_images = load_image_list(val_file)
result = parse_captions(caption_file)
vocab, reverse_vocab = build_vocabulary(result, train_images)

train_dataset = FlickrDataset(image_dir, result, train_images, vocab, transform)
val_dataset = FlickrDataset(image_dir, result, val_images, vocab, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

encoder = Encoder().to(device)
decoder = Decoder(
    vocab_size=len(vocab),
    embed_dim=embed_dim,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim,
    dropout=dropout
).to(device)

optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])


def train_epoch(encoder, decoder, loader, optimizer, criterion):
    decoder.train()
    encoder.eval()
    total_loss = 0

    for images, captions in tqdm(loader, desc="training"):
        images = images.to(device)
        captions = captions.to(device)

        image_features = encoder(images)
        predictions, alphas = decoder(image_features, captions)

        targets = captions[:, 1:]
        predictions = predictions.reshape(-1, len(vocab))
        targets = targets.reshape(-1)

        loss = criterion(predictions, targets)
        loss += lambda_reg * ((1 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(encoder, decoder, loader, criterion):
    decoder.eval()
    encoder.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="validation"):
            images = images.to(device)
            captions = captions.to(device)

            image_features = encoder(images)
            predictions, alphas = decoder(image_features, captions)

            targets = captions[:, 1:]
            predictions = predictions.reshape(-1, len(vocab))
            targets = targets.reshape(-1)

            loss = criterion(predictions, targets)
            loss += lambda_reg * ((1 - alphas.sum(dim=1)) ** 2).mean()

            total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss = float('inf')
print("starting training...")

for epoch in range(epochs):
    print(f"\nepoch {epoch+1}/{epochs}")
    train_loss = train_epoch(encoder, decoder, train_loader, optimizer, criterion)
    print(f"train loss: {train_loss:.4f}")
    val_loss = val_epoch(encoder, decoder, val_loader, criterion)
    print(f"val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'vocab': vocab,
            'reverse_vocab': reverse_vocab
        }, 'best_model.pt')
        print("saved best model!")