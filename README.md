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
Zhu et al. proposed the CycleGAN network, [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593?amp=1)   

#### Datasets           
Left is a target image and right is a source image.        
![image](https://user-images.githubusercontent.com/48985628/187868064-b6ed95e9-16af-4eb3-9efa-319c16c2bdd0.png)
          
#### Results          
Model-transformed images                      
![image](https://user-images.githubusercontent.com/48985628/187868525-75071c49-9673-4911-a580-e907ecd5545f.png)             
The results of the model show that the texture of the spine is not preserved and the quality of the image is much lower.         

## Cycle_UNetGAN
Most of the models specialized in medical images use the U-Net structure, and it is that medical images are most important to preserve their unique characteristics. Therefore, to preserve the characteristics of the original image, I set the existing CycleGAN generator structure as U-Net and developed the Unet-CycleGAN using the cycle-consistency loss for domain conversion.       

#### Results       
Left is a source image and right is a transformed image. (same as input datasets of CycleGAN)                   
![image](https://user-images.githubusercontent.com/48985628/188071291-e1826349-c1ba-47bf-850d-9b573c24b67c.png)  
The fake data synthesized by Cycle-UNetGAN was evaluated by the segmentation model. However, the results of domain adaptation for the segmentation model were lower than those of no domain adaptation.

## Few_shotGAN
The normal domain adaptation method requires a large amount of data. The method of this model is to approach the unlabeled data of the source domain and to adapt it to the unconditional image generation.                    
Utkarsh et al. proposed the Few-shotGAN network, [Few-shot Image Generation via Cross-domain Correspondence](https://arxiv.org/abs/2104.06820)

## Meta_UNet
I studied methods that solves domain adaptation using meta-learning. Meta-learning is a method to learn new concepts and skills with less data. For the quality preservation of medical images, U-Net was used as a base model of meta-learning technique. The support set was all three domains, and 40,40,100 images are used for each domain.        
MAML method was used to optimize parameters of meta-learning. Paper: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)             

#### Results         
Example of query set (Image, Predicted mask)                
![image](https://user-images.githubusercontent.com/48985628/188077740-84259a1f-34f6-4a39-bd4b-96cac7d0aaf4.png)              
The proposed method was evaluated by source domain images and yielded an average Dice score of 0.7896. 

## StarGAN_v1
As CycleGAN is only 1:1 domain transformable, I used a StarGAN network to convert it into many domains using one neural network.          
Y. Choi et al. proposed the StarGAN_v1 network, [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)           
#### Results      
![image](https://user-images.githubusercontent.com/48985628/188081137-11df38a3-d423-44fe-a747-830a073451ae.png)

## StarGAN_v2

