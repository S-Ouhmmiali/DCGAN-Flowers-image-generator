import os
import torch.nn as nn
import torch


os.makedirs("images", exist_ok=True)

#---------------
# Genrator : 
#---------------
class Generator(nn.Module):
    def __init__(self,input_size,nb_channels):
        super(Generator, self).__init__()

        self.init_size = input_size// 2**5
        self.nb_channels = nb_channels
        self.f1 = nn.Sequential(nn.Linear(100, 128*8*self.init_size**2),nn.Tanh())
        self.conv_blocks = nn.Sequential(  
            nn.ConvTranspose2d(128*8, 128*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*4),
            nn.ReLU(),
            nn.ConvTranspose2d(128*4, 128*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*2, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.nb_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out =self.f1(z)
        out = out.view(out.shape[0],128*8,self.init_size,self.init_size)
        img = self.conv_blocks(out)
        return img


#-------------
# Discriminator : 
#-------------

class Discriminator(nn.Module):
    def __init__(self,input_size,nb_channels):
        super(Discriminator, self).__init__()

       
        self.nb_channels = nb_channels
        self.model = nn.Sequential(
            nn.Conv2d(self.nb_channels, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128*2, 128*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128*4, 128*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(128*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128*8, 1, 4, stride=2, padding=1),
        )
        self.sigmoid = nn.Sequential(nn.Linear((input_size//2**5)**2, 1), nn.Sigmoid())

        

    def forward(self, img):

      out = self.model(img)
      out = out.view(out.shape[0],-1)
      out = self.sigmoid(out)
      out=out.flatten()

      return out



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
