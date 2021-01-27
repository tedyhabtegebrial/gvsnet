import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEmbedding(nn.Module):
    def __init__(self,num_classes, embedding_size):
        # configs['num_classes'], configs['embedding_size']
        super(SemanticEmbedding, self).__init__()
        size_1 = num_classes
        size_2 = int(0.8*num_classes + 0.2*embedding_size)
        size_3 = int(0.5*num_classes + 0.5*embedding_size)
        size_4 = embedding_size

        self.encoder = nn.Sequential(nn.Conv2d(num_classes, size_1, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_1, out_channels=size_2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_2, out_channels=size_3, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_3, out_channels=size_4,  kernel_size=1),
            nn.ReLU(True))
        self.decoder =  nn.Sequential(nn.Conv2d(in_channels=size_4, out_channels=size_3,  kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_3, out_channels=size_2,  kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_2, out_channels=size_1,  kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=size_1, out_channels=num_classes, kernel_size=1),
            )
        self.weights_init()
    def encode(self, input_tensor):
        return self.encoder(input_tensor)

    def decode(self, input_tensor):
        out =  self.decoder(input_tensor)
        return out

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
