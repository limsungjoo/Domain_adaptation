import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from glob import glob
import SimpleITK as sitk

# from utils.preprocessing import centercropping
# from utils.transforms import image_windowing, image_minmax, mask_binarization, augment_imgs_and_masks, center_crop
import matplotlib.pyplot as plt
def image_minmax(img):
    img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
    img_minmax = (img_minmax * 255).astype(np.uint8)
        
    return img_minmax
def mask_binarization(mask_array):
    threshold = np.max(mask_array) / 2
    mask_binarized = (mask_array > threshold).astype(np.uint8)
    
    return mask_binarized
class VertebralDataset(Dataset):
    def __init__(self, img_path,  augmentation=True):
        super(VertebralDataset, self).__init__()

        self.augmentation = augmentation
        self.img_path = img_path
        # self.is_Train = is_Train
        
        
        self.mask_list = sorted(glob(img_path))

        # print(self.mask_list)
        # self.mask_list = self.mask_list[:int(len(self.mask_list)*0.80)]
        # print(self.mask_list)
        print(len(self.mask_list),"dataset: Training")
        
       
                
        
        self.len = len(self.mask_list)

        # self.augmentation = augmentation
        # self.opt = opt

        # self.is_Train = is_Train


    def __getitem__(self, index):
        # Load Image and Mask
        mask_path = self.mask_list[index]
        
        
        # xray_path = mask_path.replace('/Label/', '/Dataset/').replace('_label.png', '.png')
        xray_path = mask_path.replace('/mask/', '/spine/').replace('.png', '.jpg')
        img = cv2.imread(xray_path,0)
        # print('img:',xray_path)
        # print('msk:',mask_path)
        mask = cv2.imread(mask_path,0)
        # img = cv2.equalizeHist(img)
        # img,mask = centercropping(img,mask)
        

        # HU Windowing
        # img = image_windowing(img, self.opt.w_min, self.opt.w_max)

        # Center Crop and MINMAX to [0, 255] and Resize
        # img = center_crop(img, self.opt.crop_size)
        # mask = center_crop(mask, self.opt.crop_size)
        
        img = image_minmax(img)
        
        ori_size = img.shape

        h,w = img.shape
        bg_img = np.zeros((1024,512))
        bg_msk = np.zeros((1024,512))

        if w>h:
            x=512
            y=int(h/w *x)
        else:
            y=1024
            x=int(w/h *y)

            if x >512:
                x =512
                y= int(h/w *x)
        
        img_resize = cv2.resize(img, (x,y))
        msk_resize = cv2.resize(mask, (x,y))

        xs = int((512 - x)/2)
        ys = int((1024-y)/2)
        bg_img[ys:ys+y,xs:xs+x]=img_resize
        bg_msk[ys:ys+y,xs:xs+x]=msk_resize

        img = bg_img
        mask = bg_msk

        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/img/'+str(index)+'.jpg',img)
        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/msk/'+str(index)+'.png',mask)
        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/img/'+str(index)+'.jpg',img)
        # cv2.imwrite('/home/vfuser/sungjoo/Resize_model/exp/image_check/msk/'+str(index)+'.png',mask)
        # Add channel axis
        # img = np.stack((img,)*3,axis=0)
        # mask = np.stack((mask,)*3,axis=0)
        # img = img[None, ...].astype(np.float32)
        # mask = mask[None, ...].astype(np.float32)
        # print(img.shape)
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
        # print(img.shape)
                
        # Augmentation
        # if self.augmentation:
        #     img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img, mask
        
    def __len__(self):
        return self.len
