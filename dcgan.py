# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:41:22 2020

@author: JH
"""

""" Pytorch-GAN (DCGAN) """

import torch
import torch.nn as nn
import argparse, os, utils

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


os.makedirs("images/dcgan", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="")
parser.add_argument("--b1", type=float, default=0.5, help="")
parser.add_argument("--b2", type=float, default=0.999, help="")
parser.add_argument("--latent_dim", type=int, default=100, help="")
parser.add_argument("--img_size", type=int, default=32, help="")
parser.add_argument("--channels", type=int, default=1, help="")
parser.add_argument("--sample_interval", type=int, default=400, help="")
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1              [-1, 1, 2048]         206,848
       BatchNorm2d-2            [-1, 128, 4, 4]             256
   ConvTranspose2d-3            [-1, 128, 8, 8]         262,272
       BatchNorm2d-4            [-1, 128, 8, 8]             256
         LeakyReLU-5            [-1, 128, 8, 8]               0
   ConvTranspose2d-6           [-1, 64, 16, 16]         131,136
         LeakyReLU-7           [-1, 64, 16, 16]               0
   ConvTranspose2d-8            [-1, 1, 32, 32]           1,025
              Tanh-9            [-1, 1, 32, 32]               0
================================================================
Total params: 601,793
Trainable params: 601,793
Non-trainable params: 0
----------------------------------------------------------------
"""
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.init_size = args.img_size // 2 ** 3
        self.linear = nn.Linear(args.latent_dim, 128 * self.init_size ** 2)
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, args.channels, 4, 2, 1),
            nn.Tanh()
            )
    
    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        return img
    

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 16, 16]             272
         LeakyReLU-2           [-1, 16, 16, 16]               0
         Dropout2d-3           [-1, 16, 16, 16]               0
            Conv2d-4             [-1, 32, 8, 8]           8,224
         LeakyReLU-5             [-1, 32, 8, 8]               0
         Dropout2d-6             [-1, 32, 8, 8]               0
       BatchNorm2d-7             [-1, 32, 8, 8]              64
            Conv2d-8             [-1, 64, 4, 4]          32,832
         LeakyReLU-9             [-1, 64, 4, 4]               0
        Dropout2d-10             [-1, 64, 4, 4]               0
      BatchNorm2d-11             [-1, 64, 4, 4]             128
           Conv2d-12            [-1, 128, 2, 2]         131,200
        LeakyReLU-13            [-1, 128, 2, 2]               0
        Dropout2d-14            [-1, 128, 2, 2]               0
      BatchNorm2d-15            [-1, 128, 2, 2]             256
           Linear-16                    [-1, 1]             513
          Sigmoid-17                    [-1, 1]               0
================================================================
"""
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        def discriminator_block(in_features, out_features, bn=True):
            block = [nn.Conv2d(in_features, out_features, 4, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_features, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)                
            )
        
        self.down_size = args.img_size // 2 ** 4
        self.layer = nn.Sequential(nn.Linear(128 * self.down_size ** 2, 1),
                                   nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        val = self.layer(out)
        return val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = generator().to(device)
discriminator = discriminator().to(device)

adversarial_loss = torch.nn.BCELoss().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

generator.apply(utils.weights_init_normal)
discriminator.apply(utils.weights_init_normal)

os.makedirs('data', exist_ok=True)
dataloader = DataLoader(datasets.MNIST('data', train=True, download=True,
                                       transform=transforms.Compose([transforms.Resize(args.img_size),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(0.5, 0.5)])
                                       ),
                        batch_size = args.batch_size,
                        shuffle=True,
                        )

for epoch in range(args.epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        imgs = imgs.to(device)
        
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        z = torch.randn(imgs.shape[0], args.latent_dim).to(device)
        
        
        optimizer_G.zero_grad()
                
        gen_imgs = generator(z)
        
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        
        optimizer_G.step()
        
        
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        
        optimizer_D.step()
        
        print('[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]'
              % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/dcgan/%d.png' % batches_done, nrow=5, normalize=True)





    
