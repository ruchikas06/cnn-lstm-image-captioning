import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        #Remove the last two layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        #Freeze pretrained Resnet weights
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        #Images shape : (batch, 3, 224, 224)
        features = self.resnet(images)
        #Features shape : (batch, 2048, 7, 7)
        batch = features.shape[0]
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, -1, 2048)
        return features