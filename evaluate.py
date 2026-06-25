import torch
import os
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import load_image_list, parse_captions, build_vocabulary
from utils.dataset import transform

image_dir = 'data/images'
caption_file = 'data/captions/Flickr8k.token.txt'
train_file = 'data/captions/Flickr_8k.trainImages.txt'
test_file = 'data/captions/Flickr_8k.testImages.txt'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_images = load_image_list(train_file)
test_images = load_image_list(test_file)
result = parse_captions(caption_file)
vocab, reverse_vocab = build_vocabulary(result, train_images)

checkpoint = torch.load('best_model.pt', map_location=device)

encoder = Encoder().to(device)
decoder = Decoder(
    vocab_size=len(vocab),
    embed_dim=256,
    attention_dim=512,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()


def generate_caption(encoder, decoder, image, vocab, reverse_vocab, max_length=50):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        image_features = encoder(image)
        h, c = decoder.init_hidden_state(image_features)
        word = torch.tensor([vocab['<start>']]).to(device)

        caption = []
        alphas = []

        for _ in range(max_length):
            embedding = decoder.embedding(word)
            context, alpha = decoder.attention(image_features, h)
            alphas.append(alpha.cpu())
            lstm_input = torch.cat([embedding, context], dim=1)
            h, c = decoder.lstm(lstm_input, (h, c))
            pred = decoder.fc(decoder.dropout(h))
            word_idx = pred.argmax(dim=1)
            word = word_idx
            caption.append(word_idx.item())
            if reverse_vocab[word_idx.item()] == '<end>':
                break

        words = [reverse_vocab[idx] for idx in caption
                 if reverse_vocab[idx] not in ['<end>', '<pad>', '<start>']]
        return words, alphas


matching = [name for name in test_images if name in result]
references = []
hypotheses = []

for image_name in tqdm(matching[:200], desc="evaluating"):
    try:
        refs = [cap.lower().split() for cap in result[image_name]]
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        hypothesis, _ = generate_caption(encoder, decoder, image, vocab, reverse_vocab)
        if len(hypothesis) > 0:
            references.append(refs)
            hypotheses.append(hypothesis)
    except Exception as e:
        continue

smoother = SmoothingFunction().method1
bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smoother)
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smoother)
bleu3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smoother)
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother)

print(f"\nbleu-1: {bleu1*100:.1f}")
print(f"bleu-2: {bleu2*100:.1f}")
print(f"bleu-3: {bleu3*100:.1f}")
print(f"bleu-4: {bleu4*100:.1f}")
print(f"\npaper reports bleu-4 of 19.5 for soft attention on flickr8k")