import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
 
img_list=glob.glob('cornell/01/*.png')
 
class CornellDataSet(Dataset):
  def __init__(self,image_list,transforms=None):
    self.image_list=image_list
    self.transforms=transforms
  def __len__(self):
    return len(self.image_list)
  def __getitem__(self,i):
    img=plt.imread(self.image_list[i])
    img=Image.fromarray(img).convert('RGB')
    img=np.array(img).astype(np.uint8)
 
    if self.transforms is not None:
      img=self.transforms(img)
    return torch.tensor(img,dtype=torch.float)
 
def show_img(img):
  plt.figure(figsize=(40,38))
  npimg=img.numpy()
  plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.show()

transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomRotation((0,360)),   
                              transforms.ToTensor(),
                              transforms.RandomErasing(),  
                              ])

Image_dataloader=DataLoader(CornellDataSet(img_list,transform),batch_size=8,shuffle=True)

data=iter(Image_dataloader)
show_img(torchvision.utils.make_grid(data.next()))