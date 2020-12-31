# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 08:24:14 2020

@author: JH
"""
""" Pytorch-GAN (CGAN) """

import torch
import torch.nn as nn
import argparse, os

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


os.makedirs("images/cgan", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.embedding = nn.Embedding(args.n_classes, args.n_classes)
        
        self.init_size = args.img_size // 2 ** 3
        self.linear = nn.Linear(args.latent_dim + args.n_classes, 128 * self.init_size ** 2)
        
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
    
    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        emb = self.embedding(label)
        out = self.linear(torch.cat((z, emb), 1))
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        return img
    
    
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.embedding = nn.Embedding(args.n_classes, args.img_size * args.img_size)

        def discriminator_block(in_features, out_features, bn=True):
            block = [nn.Conv2d(in_features, out_features, 4, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.4)]
            if bn:
                block.append(nn.BatchNorm2d(out_features, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(args.channels + 1, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)                
            )
        
        self.down_size = args.img_size // 2 ** 4
        self.layer = nn.Sequential(nn.Linear(128 * self.down_size ** 2, 1),
                                   nn.Sigmoid())

    def forward(self, img, label):
        emb = self.embedding(label)
        emb = emb.view(emb.shape[0], args.channels, args.img_size, args.img_size)
        out = self.model(torch.cat((img, emb), 1))
        out = out.view(out.shape[0], -1)
        val = self.layer(out)
        return val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = generator().to(device)
discriminator = discriminator().to(device)

adversarial_loss = torch.nn.BCELoss().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

'''
generator.apply(utils.weights_init_normal)
discriminator.apply(utils.weights_init_normal)
'''

os.makedirs('data', exist_ok=True)
dataloader = DataLoader(datasets.MNIST('data', train=True, download=True,
                                       transform=transforms.Compose([transforms.Resize(args.img_size),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5,), (0.5,))])
                                       ),
                        batch_size = args.batch_size,
                        shuffle=True,
                        )

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    
    z = torch.randn(n_row**2, args.latent_dim).to(device)    
    labels = torch.LongTensor([num for _ in range(n_row) for num in range(n_row)]).to(device)
    gen_imgs = generator(z, labels)
    
    save_image(gen_imgs.data, "images/cgan/%d.png" % batches_done, nrow=n_row, normalize=True)
    

for epoch in range(args.epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        z = torch.randn(imgs.shape[0], args.latent_dim).to(device)
        gen_labels = torch.randint(args.n_classes, (labels.shape[0],)).long().to(device)
        
        optimizer_G.zero_grad()
                
        gen_imgs = generator(z, gen_labels)
        
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------        
        
        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        
        optimizer_D.step()
        
        
        print('[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]'
              % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            #save_image(gen_imgs.data[:25], 'images/dcgan/%d.png' % batches_done, nrow=5, normalize=True)
            sample_image(n_row=10, batches_done=batches_done)
