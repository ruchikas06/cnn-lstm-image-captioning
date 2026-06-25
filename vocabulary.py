from collections import Counter


def load_image_list(filepath):
    with open(filepath, 'r') as f:
        return set(f.read().strip().split('\n'))


def parse_captions(filepath):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id, caption = line.split('\t', 1)
            image_name = image_id.split('#')[0]
            if image_name not in result:
                result[image_name] = []
            result[image_name].append(caption)
    return result


def build_vocabulary(result, train_images, min_freq=5):
    word_counts = Counter()
    for image, caps in result.items():
        if image not in train_images:
            continue
        for cap in caps:
            for word in cap.lower().split():
                word_counts[word] += 1

    vocab_list = ['<pad>', '<start>', '<end>', '<unk>']
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab_list.append(word)

    vocab = {word: idx for idx, word in enumerate(vocab_list)}
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    return vocab, reverse_vocab