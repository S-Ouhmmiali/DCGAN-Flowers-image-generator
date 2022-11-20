import os
from model import dataloader, model
from torchvision.utils import save_image
from torch.autograd import Variable
import torch

# Create folder /images to store generated images
os.makedirs("images", exist_ok=True)

def train(generator,discriminator,dataset,criterion,G_optimizer, D_optimizer,dataloader,nb_epoches,Tensor,cuda):
    for epoch in range(nb_epoches):
        for i, imgs in enumerate(dataloader):

            if dataset=='MNIST':
              inputs,_ = imgs
              imgs = inputs

            # Labels for real and fake samples
            labels_valid = Variable(torch.ones(imgs.shape[0]).cuda() if cuda else torch.ones(imgs.shape[0]))
            labels_fake = Variable(torch.zeros(imgs.shape[0]).cuda() if cuda else torch.zeros(imgs.shape[0]))
            # Configure input
            real_inputs = Variable(imgs.cuda() if cuda else imgs)
            fake_inputs = Variable(torch.randn((imgs.shape[0], 100)).view(-1, 100).cuda() if cuda else torch.randn((imgs.shape[0], 100)).view(-1, 100))

            # -----------------
            #  Train Generator
            # -----------------
            # Generate images with fake_inputs
            gen_imgs = generator(fake_inputs)  
            # Measure generator's ability to generate valid samples        
            g_loss = criterion(discriminator(gen_imgs), labels_valid)


            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(real_inputs), labels_valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), labels_fake)
            d_loss = (real_loss + fake_loss) / 2


            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, nb_epoches, i+1, len(dataloader), d_loss.item(), g_loss.item())
            )
        #-------------------
        # Save generated images :
        #-------------------
        if epoch%2 == 0:
          save_image(gen_imgs.data[:4], "images/i__%d.png" % epoch, nrow=5, normalize=True)
if __name__ == '__main__':

    # Initial parameters : 
    batch_size = 32
    input_size = 128
    learning_rate = 0.0002
    nb_channels = 3
    nb_epoches = 26
    data_dir = 'flowers_dataset'


    # Dataset
    dataset = 'MNIST'
    if dataset == 'MNIST':
      nb_channels = 1
    

   
    
    # DataLoader
    dataloader = dataloader.getDataLoader(dataset,batch_size,data_dir,input_size)

    generator = model.Generator(input_size,nb_channels)
    discriminator = model.Discriminator(input_size,nb_channels)

    # Loss function : 
    criterion = torch.nn.BCELoss()

    #GPU if available : 
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if cuda:
        generator.cuda()
        discriminator.cuda()
        criterion.cuda()

    # Optimizers : 
    generator.apply(model.weights_init_normal)
    discriminator.apply(model.weights_init_normal)
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))



    train(generator,discriminator,dataset,criterion,G_optimizer, D_optimizer,dataloader,nb_epoches,Tensor,cuda)


