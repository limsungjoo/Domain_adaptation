Domain adaptation
=====================
There are a variety of manufacturers of machines that take x-rays at each hospital. The Vertebra segmentation model was trained for X-ray images from one manufacturer. If the model is evaluated with images taken by other manufacturers, the performance of the model will be considerably reduced. I have been studied domain adaptation methodology with the aim of minimizing these domain shift problems. I have applied methods based GAN, Meta-learning, and fine-tuning for domain adaptation. 
* [Cycle_GAN](#cycle_gan)
* [Cycle_UNetGAN](#cycle_unetgan)
* [Few_shotGAN](#few_shotgan)
* [Meta_UNet](#meta_unet)
* [StarGAN_v1](#stargan_v1)
* [StarGAN_v2](#stargan_v2)
----------------------

## Cycle_GAN 
CycleGAN is a proposed model to convert the source domains into target domains without pairing.                     
The source domain of the model is set as training data of the segmentation model, and the target domain is set as external data taken by other manufacturers.       
Zhu et al. proposed the CycleGAN network, paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593?amp=1)   

#### Datasets           
Left is a target image and right is a source image.        
![image](https://user-images.githubusercontent.com/48985628/187868064-b6ed95e9-16af-4eb3-9efa-319c16c2bdd0.png)
          
#### Results          
Model-transformed images                      
![image](https://user-images.githubusercontent.com/48985628/187868525-75071c49-9673-4911-a580-e907ecd5545f.png)             
The results of the model show that the texture of the spine is not preserved and the quality of the image is much lower.         

## Cycle_UNetGAN
Most of the models specialized in medical images use the U-Net structure, and the reason for this is that medical images are most important to preserve their unique characteristics.So, to preserve the characteristics of the original image, we set the existing CycleGAN generator structure as U-Net and developed the Unet-CycleGAN using the cycle-consistency loss for domain conversion.       

## Few_shotGAN

## Meta_UNet


## StarGAN_v1


## StarGAN_v2

