import torch
import torch.nn as nn

class CameraEncoder(nn.Module):
    def __init__(self, opt, nc=3, nk=5):
        super(CameraEncoder, self).__init__()
        self.name = 'camera_encoder'

        block1 = self.convblock(nc, 32, nk, stride=2, pad=2)
        block2 = self.convblock(32, 64, nk, stride=2, pad=2)
        block3 = self.convblock(64, 128, nk, stride=2, pad=2)
        block4 = self.convblock(128, 256, nk, stride=2, pad=2)
        block5 = self.convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = self.linearblock(512, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 14)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, \
        linear1, linear2 \

    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, x, camera_gt):
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        # cameras
        camera_output = self.linear3(x)

        # get the error
        error = torch.pow(camera_output - camera_gt.squeeze(1), 2).mean()

        return camera_output, error
