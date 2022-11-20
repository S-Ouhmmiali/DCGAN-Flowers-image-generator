
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch

# Flowers dataset class : 

class FlowersDataset(Dataset):
    def __init__(self,data_dir, transform=None):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames]
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open(self.filenames[index]) 
        image = self.transform(image)
        return image
    def __len__(self):
        return len(self.filenames)

def getDataLoader(dataset,batch_size,data_dir,input_size):

    # Transformer : 
    transformer = transforms.Compose([
                transforms.Resize((input_size,input_size)),        
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    if dataset== 'Flowers':
        dataset = FlowersDataset(data_dir,transform = transformer)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    elif dataset == "MNIST":

        dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('data_MNIST', train=True, download=True, transform=transformer),
        batch_size=batch_size, shuffle=True)
    
    
    return dataloader



