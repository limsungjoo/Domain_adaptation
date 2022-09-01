import cv2
# import tensorflow_datasets as tfds
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import itertools

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Dataset
def image_minmax(img):
    img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
    img_minmax = (img_minmax * 255).astype(np.uint8)
        
    return img_minmax

image = sorted(os.listdir('/home/vfuser/sungjoo/data/Lateral_deepnoid/spine'))
label = sorted(os.listdir('/home/vfuser/sungjoo/data/new_dataset/spine'))[:len(image)]


class CycleGanData(Dataset):
    def __init__(self,trainA_path,trainB_path,transform):
        self.trainA_path = trainA_path
        self.trainB_path = trainB_path
        self.transform = transform
        
    def __len__(self):
        return len(image)
    
    def trainA(self,trainA_path):
        trainA = sitk.ReadImage('/home/vfuser/sungjoo/data/Lateral_deepnoid/spine/'+trainA_path)
        # trainA = Image.open('/data/workspace/vfuser/VF/data/GE_all/'+trainA_path)
        
        trainA = sitk.GetArrayFromImage(trainA)
        # trainA = cv2.cvtColor(trainA, cv2.COLOR_BGR2GRAY)
        # print(np.mean(trainA))
        # trainA = image_minmax(trainA)
        # IMG_SIZE = 256

        # ori_size = trainA.shape

        # h,w = trainA.shape
        
        # bg_img = np.zeros((1024,512))

        # if w>h:
        #     x=512
        #     y=int(h/w *x)
        # else:
        #     y=1024
        #     x=int(w/h *y)

        #     if x >512:
        #         x =512
        #         y= int(h/w *x)
        
        # img_resize = cv2.resize(trainA, (x,y))

        # xs = int((512 - x)/2)
        # ys = int((1024-y)/2)
        # bg_img[ys:ys+y,xs:xs+x]=img_resize

        # trainA = bg_img
        trainA = cv2.resize(trainA,(800,1000))
        trainA = trainA / 255.
        
        trainA = self.transform(trainA)
        return trainA
    
    def trainB(self,trainB_path):
        trainB = sitk.ReadImage('/home/vfuser/sungjoo/data/new_dataset/spine/'+trainB_path)
        IMG_SIZE = 256
        # trainB = trainB.resize((296,420))

        trainB = sitk.GetArrayFromImage(trainB)
        # trainB = cv2.cvtColor(trainB, cv2.COLOR_BGR2GRAY)
        
        # trainB = image_minmax(trainB)
        

        # ori_size = trainB.shape

        # h,w = trainB.shape
        
        # bg_img = np.zeros((1024,512))

        # if w>h:
        #     x=512
        #     y=int(h/w *x)
        # else:
        #     y=1024
        #     x=int(w/h *y)

        #     if x >512:
        #         x =512
        #         y= int(h/w *x)
        
        # img_resize = cv2.resize(trainB, (x,y))

        # xs = int((512 - x)/2)
        # ys = int((1024-y)/2)
        # bg_img[ys:ys+y,xs:xs+x]=img_resize

        # trainB = bg_img
        trainB = cv2.resize(trainB,(800,1000))
        trainB = trainB / 255.
        trainB = self.transform(trainB)
        return trainB
    
    def __getitem__(self,index):
        trainA = self.trainA(self.trainA_path[index])
        trainB = self.trainB(self.trainB_path[index])
            
        return {'trainA': trainA,
                'trainB': trainB}


# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(256), 
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])
  
# DataSet, DataLoader
Dataset = CycleGanData(trainA_path=image,trainB_path=label,transform=transform)
# print(Dataset[2]['trainA'])
DataLoader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=False, num_workers=0,drop_last=True)

from PIL import Image

# print(Dataset[40]['trainA'].shape)
# print(Dataset[40]['trainB'].shape)


# Model Making
from CycleGAN import ResidualBlock
# from CycleGAN import Generator
from CycleGAN import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from Unetencoder import Generator

# import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# netG_A2B = Generator(3, 3).to(device)
# netG_B2A = Generator(3, 3).to(device)
# netD_A = Discriminator(3).to(device)
# netD_B = Discriminator(3).to(device)

netG_A2B = Generator(1, 1, 8).to(device)
netG_B2A = Generator(1, 1, 8).to(device)
netD_A = Discriminator(1).to(device)
netD_B = Discriminator(1).to(device)

# Pretrained
# netG_A2B.load_state_dict(torch.load('output/pkl/netG_A2B.pkl'))
# netG_B2A.load_state_dict(torch.load('output/pkl/netG_B2A.pkl'))
# netD_A.load_state_dict(torch.load('output/pkl/netD_A.pkl'))
# netD_B.load_state_dict(torch.load('output/pkl/netD_B.pkl'))


netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Loss Function 

criterion_GAN = torch.nn.MSELoss()
########################변경#######
# criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

LAMBDA = 10
loss_obj = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=2e-4, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

n_epochs = 200
decay_epoch = 25 # epoch to start linearly decaying the learning rate to 0

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(1, 1, 1000,800)


input_B = Tensor(1, 1, 1000,800)


from torch.autograd import Variable
target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)

target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)


from utils import ReplayBuffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Loss plot
logger = Logger(n_epochs, len(DataLoader))

save_dir = "./"
# Train Set Learning
loss_arr = []
for epoch in range(n_epochs):
    # for i,batch in tqdm(enumerate(DataLoader),total=len(DataLoader)):
    for i,batch in enumerate(DataLoader):
        real_A = batch['trainA'].to(torch.float).to(device)
        real_B = batch['trainB'].to(torch.float).to(device)
      
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        #######################################변경###############
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        # loss_GAN_A2B = criterion_GAN(pred_fake, real_B)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        # loss_GAN_B2A = criterion_GAN(pred_fake, real_A)


        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*5.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*5.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097) (python -m visdom.server)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
                    
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    pkl_save_dir = os.path.join('output', 'pkl')
    if not os.path.exists(pkl_save_dir):
        os.makedirs(pkl_save_dir)

    model_filename = '%s_loss_epoch_' % (epoch+1)

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), '/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netG_A2B/deepnoid/'+model_filename+'netG_A2B.pkl')
    torch.save(netG_B2A.state_dict(), '/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netG_B2A/deepnoid/'+model_filename+'netG_B2A.pkl')
    torch.save(netD_A.state_dict(), '/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netD_A/deepnoid/'+model_filename+'netD_A.pkl')
    torch.save(netD_B.state_dict(), '/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netD_B/deepnoid/'+model_filename+'netD_B.pkl')






