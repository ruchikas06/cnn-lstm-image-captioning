import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class FlickrDataset(Dataset):
    def __init__(self, image_dir, result, image_list, vocab, transform):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        self.pairs = []
        for image in image_list:
            if image not in result:
                continue
            for cap in result[image]:
                self.pairs.append((image, cap))

    def caption_to_tensor(self, caption):
        words = ['<start>'] + caption.lower().split() + ['<end>']
        indices = []
        for word in words:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab['<unk>'])
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, caption = self.pairs[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption_tensor = self.caption_to_tensor(caption)
        return image, caption_tensor


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
    return images, padded