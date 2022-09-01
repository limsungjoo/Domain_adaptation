import argparse
import sys
import os
from PIL import Image
import cv2
from PIL import Image
import SimpleITK as sitk
import torchvision.transforms as transforms
import torchvision.utils as v_utils
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch
from Unetencoder import Generator
# from CycleGAN import Generator
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(1, 1).to(device)
netG_B2A = Generator(1, 1).to(device)
# netG_A2B = torch.nn.DataParallel(netG_A2B)
# netG_B2A=torch.nn.DataParallel(netG_B2A)

# Load state dicts
netG_A2B.load_state_dict(torch.load('/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netG_A2B/deepnoid/200_loss_epoch_netG_A2B.pkl'))
netG_B2A.load_state_dict(torch.load('/home/vfuser/sungjoo/Cycle-GAN/output/pkl/netG_B2A/deepnoid/200_loss_epoch_netG_B2A.pkl'))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(1, 1, 1000,800) # (batchsize, output channel, size, size)
input_B = Tensor(1, 1, 1000,800)

# Dataset loader

image = sorted(os.listdir('/home/vfuser/sungjoo/data/Lateral_deepnoid/spine'))
label = sorted(os.listdir('/home/vfuser/sungjoo/data/new_dataset/spine'))[:len(image)]

print(len(image),len(label))
# for i in range(0,10000):
#     random.shuffle(label)
#     if 'busan' == label[0]:
#         break
#     else :
#         continue
# label = label[:58]
# print(len(label))
# print(label)
class CycleGanData_test(Dataset):
    def __init__(self,testA_path,testB_path,transform):
        self.testA_path = testA_path
        self.testB_path = testB_path
        self.transform = transform
        
    def __len__(self):
        return len(image)
    
    def testA(self,testA_path):
        testA = sitk.ReadImage('/home/vfuser/sungjoo/data/Lateral_deepnoid/spine/'+testA_path)
        # testA = testA.resize((296,420))
        # testA = self.transform(testA)
        testA = sitk.GetArrayFromImage(testA)
        # testA = cv2.cvtColor(testA, cv2.COLOR_BGR2GRAY)
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

        testA = cv2.resize(testA,(800,1000))
        # cv2.imwrite('/home/vfuser/sungjoo/Cycle-GAN/output/testA/'+testA_path,testA)
        testA = testA / 255.
        
        trainA = self.transform(testA)
        return testA
    
    def testB(self,testB_path):
        testB = sitk.ReadImage('/home/vfuser/sungjoo/data/new_dataset/spine/'+testB_path)
        # testB = testB.resize((296,420))
        # testB = self.transform(testB)
        testB = sitk.GetArrayFromImage(testB)
        
        # testB = cv2.cvtColor(testB, cv2.COLOR_BGR2GRAY)
        # trainB = image_minmax(trainB)
        # IMG_SIZE = 256

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
        testB = cv2.resize(testB,(800,1000))
        
        testB = testB / 255.
        # trainB = sitk.ReadImage('dataset/cezanne2photo/trainB/'+trainB_path)
        # trainB = sitk.GetArrayFromImage(trainB)
        # trainB = cv2.resize(trainB,(IMG_SIZE,IMG_SIZE))
        # trainB = trainB.astype('float')
        testB = self.transform(testB)
        return testB
    
    def __getitem__(self,index):
        testA = self.testA(self.testA_path[index])
        testB = self.testB(self.testB_path[index])
        testA_name = self.testA_path[index]  
        testB_name = self.testB_path[index]   
        return {'testA': testA,
                'testB': testB,
                'testA_name' : testA_name,
                'testB_name' : testB_name}

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256,256), Image.BICUBIC),
    # transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# DataSet, DataLoader
Dataset = CycleGanData_test(testA_path=image,testB_path=label,transform=transform)
# print(Dataset[2]['trainA'])

dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=False, num_workers=0,drop_last=True)

###### Testing######

# Create output dirs if they don't exist
# if not os.path.exists('output/A_image'):
#     os.makedirs('output/A_image')
# if not os.path.exists('output/B_image'):
#     os.makedirs('output/B_image')

# for i, batch in enumerate(dataloader):
#     # Set model input
#     real_A = Variable(input_A.copy_(batch['testA']))
#     real_B = Variable(input_B.copy_(batch['testB']))

#     # Generate output
#     fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
#     fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

#     # Save image files
#     batch_tensorA = torch.cat((real_A, fake_B), dim=2)
#     batch_tensorB = torch.cat((real_B, fake_A), dim=2)
    
#     grid_imgA = v_utils.make_grid(batch_tensorA) # padding = 1, nrow = 4
#     grid_imgB = v_utils.make_grid(batch_tensorB)

#     v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/A_image/%04d.jpg' % (i+1))
#     v_utils.save_image(grid_imgB, '/home/vfuser/sungjoo/Cycle-GAN/output/B_image/%04d.jpg' % (i+1))

#     sys.stdout.write('\rGenerated images %04d of %04d'%(i+1, len(dataloader)))
# sys.stdout.write('\n')
#PSNR#
import numpy 
import math
def psnr(img1, img2):

    mse = numpy.mean( (img1 - img2) ** 2 ) #MSE 구하는 코드

    print("mse : ",mse)

    if mse == 0:

        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
k=0
total = 0
for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['testA']))
    real_B = Variable(input_B.copy_(batch['testB']))
    name_A = batch['testA_name']
    name_B = batch['testB_name']
    # print(name_A)
    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    # batch_tensorA = torch.cat((real_A, fake_B), dim=2)
    # batch_tensorB = torch.cat((real_B, fake_A), dim=2)
    batch_tensorA = real_A
    batch_tensorB = fake_B


    grid_imgA = v_utils.make_grid(batch_tensorA) # padding = 1, nrow = 4
    grid_imgB = v_utils.make_grid(batch_tensorB)
    grid_imgC = v_utils.make_grid(real_B)
    # v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/A_image/%04d.jpg' % (i+1))
    # v_utils.save_image(grid_imgA, '/home/vfuser/sungjoo/Cycle-GAN/output/deepnoid_A/'+ name_A[0])
    # v_utils.save_image(grid_imgB, '/home/vfuser/sungjoo/Cycle-GAN/output/deepnoid_B/'+name_A[0])
    # v_utils.save_image(grid_imgC, '/home/vfuser/sungjoo/Cycle-GAN/output/GE_Real/'+name_B[0])
    
    sys.stdout.write('\rGenerated images %04d of %04d'%(i+1, len(dataloader)))
    real = real_B.cpu().numpy()
    fake = fake_B.cpu().numpy()
    psnr_s = psnr(real,fake)
    print(psnr_s)
    total+=round(psnr_s,2)
    k+=1
sys.stdout.write('\n')
print(k)
avg = total/k
print(avg)
###################################
