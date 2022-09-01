import os
import higher
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from utils import sgd, crossentropyloss, fix_seed, write_log, compute_accuracy, binarycrossentropy, rmsprop
from dice_loss import GeneralizedDiceLoss, WeightedCrossEntropyLoss, compute_per_channel_dice,DiceLoss
import numpy as np
import torch.utils.data as data
import torch.optim as optim
import os.path
import torch
from utils3D import cross_entropy_dice
from data_reader_unet_spine import *
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import Unet3D_meta_learning
import scipy.misc
import sklearn as sk
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle
import torchvision
import sys
from visdom import Visdom
import time
import datetime
from datasets.dataset import VertebralDataset
from torch.utils.data import DataLoader
from modified_unet import Modified2DUNet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=torch.cuda.FloatTensor([self.epoch]), Y=torch.cuda.FloatTensor([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

# class SubsetRandomSampler(SubsetRandomSampler):
#     r"""Samples elements randomly, without replacement.

#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """

#     def __init__(self, data_source):
#         self.data_source=data_source

#     def __iter__(self):
#         cpu= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         return iter(self.data_source)

#     def __len__(self):
#         return len(self.data_source)
def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]

        return f_net

class ModelBaseline:
    def __init__(self):
        self.batch_size = 2
        self.num_classes = 2
        self.unseen_index = 0
        self.lr = 0.001
        self.inner_loops = 1200
        self.step_size = 20
        self.weight_decay = 0.00005
        self.momentum = 0.9
        self.state_dict = ''
        self.logs = 'logs'
        self.patch_size = 64
        self.test_every = 50
        self.test_unseen = 100
        self.epochs = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(comment='spine-baseline-train-june21')
        self.n_gradient_step = 1
        self.n_train_step = 10
        self.alpha = 0.01
        self.beta = 0.001
        self.model =  Modified2DUNet(1, 1, 8)
    #     torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    # in_channels=3, out_channels=1, init_features=32, pretrained=True)

        self.weights = list(self.model.parameters())
        self.meta_optimizer = optim.Adam(self.weights, lr=self.beta)
        log_dir = f'./runs/MAML_{"seg"}' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # torch.set_default_tensor_type('torch.cuda.DoubleTensor')

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

        # fix the random seed or not
        fix_seed()

        self.setup_path()
        self.network = Unet3D_meta_learning.Unet3D()  # load the vanilla 3D-Unet

        self.network = self.network.cuda()

        self.configure()

    def setup_path(self):
        modality = 'dummy'

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.test_path = ['/home/vfuser/sungjoo/data/meta_GE/test/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam_test/dong_test/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam_test/sam_test/mask/*']

        self.train_paths = ['/home/vfuser/sungjoo/data/meta_GE/train/train/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam/dong/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam/sam/mask/*']

        self.few_paths = ['/home/vfuser/sungjoo/data/dongsam/dong/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam/sam/mask/*']

        self.val_paths = ['/home/vfuser/sungjoo/data/meta_GE/val/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam/dong_val/mask/*',
                            '/home/vfuser/sungjoo/data/dongsam/sam_val/mask/*']


        for x in range(3):
            img_path = self.train_paths[x]
            dataset_img =VertebralDataset(img_path,  augmentation=False)
            # dataset_img = BatchImageGenerator(img_path, modality, transform=True,
            #                             patch_size=self.patch_size, n_patches_transform=30)

            self.TrainMetaData.append(dataset_img)

            val_path = self.val_paths[x]
            dataset_val = VertebralDataset(val_path,  augmentation=False)

            # dataset_val = BatchImageGenerator(val_path, modality, transform=False,
            #                             patch_size=self.patch_size, n_patches_transform=30)
            

            self.ValidMetaData.append(dataset_val)


        curr_test_modality = self.test_path[0]
        test_dataset = VertebralDataset(curr_test_modality,  augmentation=False)
        print(test_dataset)
        self.batImageGenTests = DataLoader(test_dataset,
                                  batch_size=self.batch_size,
                                  sampler=SequentialSampler(self.TestMetaData),
                                  shuffle=False,
                                  num_workers=0)
        # self.TestMetaData = BatchImageGenerator(curr_test_modality, modality, transform=False,
        #                                 patch_size=self.patch_size, n_patches_transform=30, is_test=True)

        # self.batImageGenTests = torch.utils.data.DataLoader(self.TestMetaData, batch_size=self.batch_size,
        #                                                sampler=SequentialSampler(self.TestMetaData),
        #                                                num_workers=0,
        #                                                pin_memory=False)

    def configure(self):
        
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size, gamma=0.1)
        self.loss_fn = DiceLoss()

    def train_step(self):
        logger = Logger(self.epochs, len(self.TrainMetaData))
        print('>>>>>>>>>TRAINING<<<<<<<<')
        PATH = '/home/vfuser/sungjoo/Meta_Unet/exp/100_MAML.pth'
        pretrained_dict = torch.load(PATH, map_location=torch.device('cuda'))
        self.model.load_state_dict(pretrained_dict)
        self.network.train()
        self.best_accuracy_val=-1

        self.batImageGenTrains=[]
        self.batImageGenVals=[]
        self.batImageGenFews=[]
        #self.batImageGenTests=[]

        for dataset in self.TrainMetaData:
            
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(dataset),
                                                       num_workers=0,
                                                       pin_memory=False)

            self.batImageGenTrains.append(train_loader)

        for dataset in self.ValidMetaData:

            val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(dataset),
                                                       num_workers=0,
                                                       pin_memory=False)

            self.batImageGenVals.append(val_loader)

        fewshot = False
        #fewshot training
        # curr_few_modality = self.few_paths[1]
        # if fewshot is True:
        #     dataset_few = BatchImageGenerator(curr_few_modality, modality, transform=True,
        #                                 patch_size=self.patch_size, n_patches_transform=30)
        #     self.TrainMetaData.append(dataset_few)


        # add a fourth modality if few shot
        d1 = iter(self.batImageGenTrains[0])
        d2 = iter(self.batImageGenTrains[1])
        d3 = iter(self.batImageGenTrains[2])
        t1 = iter(self.batImageGenVals[0])
        t2 = iter(self.batImageGenVals[1])
        t3 = iter(self.batImageGenVals[2])
        # d4 = iter(self.batImageGenTrains[3])

        ite = 100
        total_loss=0.0
        for epoch in range(self.epochs):
            print("<<<<<<<<< epoch >>>>>>>>", epoch)
            d1 = iter(self.batImageGenTrains[0])
            d2 = iter(self.batImageGenTrains[1])
            d3 = iter(self.batImageGenTrains[2])
            t1 = iter(self.batImageGenVals[0])
            t2 = iter(self.batImageGenVals[1])
            t3 = iter(self.batImageGenVals[2])
            # d4 = iter(self.batImageGenTrains[3])

            # for im in range(6):

            trains_d1, labels_d1 = next(d1)
            trains_d1 = trains_d1.squeeze(0)
            labels_d1 = labels_d1.squeeze(0)

            trains_d2, labels_d2 = next(d2)
            trains_d2 = trains_d2.squeeze(0)
            labels_d2 = labels_d2.squeeze(0)

            trains_d3, labels_d3 = next(d3)
            trains_d3 = trains_d3.squeeze(0)
            labels_d3 = labels_d3.squeeze(0)

            trains_t1, labels_t1 = next(t1)
            trains_t1 = trains_t1.squeeze(0)
            labels_t1 = labels_t1.squeeze(0)

            trains_t2, labels_t2 = next(t2)
            trains_t2 = trains_t2.squeeze(0)
            labels_t2 = labels_t2.squeeze(0)

            trains_t3, labels_t3 = next(t3)
            trains_t3 = trains_t3.squeeze(0)
            labels_t3 = labels_t3.squeeze(0)

            # trains_d4, labels_d4 = next(d4)
            # trains_d4 = trains_d4.squeeze(0)
            # labels_d4 = labels_d4.squeeze(0)

            # bs = 2
            # images_train_three_domains_shape = np.shape(trains_d1)
            # print(images_train_three_domains_shape)
            # total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            # # print(total_batches_per_image)
            # batch_index = 0

            
            ite += 1
            
            loss = 0.0
            total_train_acc = []
            meta_train_loss=0.0
            
            for index in range(3):
                
                if index==0:
                    trains, labels = trains_d1, labels_d1
                    vals, val_labels = trains_t1, labels_t1
                if index==1:
                    trains, labels = trains_d2, labels_d2
                    vals, val_labels = trains_t2, labels_t2
                if index==2:
                    trains, labels = trains_d3, labels_d3
                    vals, val_labels = trains_t3, labels_t3
                # if index==3:
                #     trains, labels = trains_d4, labels_d4
                torch.autograd.set_detect_anomaly(True)
                for i in range(self.batch_size):
                ############MAML###########
                    
                    images_train, labels_train = trains.float(), labels.float()
                    outputs = self.model(images_train)
                    train_loss = self.loss_fn(self.model(images_train), labels_train)
                    
                    grad = torch.autograd.grad(train_loss, self.weights)
                    temp_weights = [w - self.alpha * g for w, g in zip(self.weights, grad)]
                    y = outputs.sigmoid()
                    zeros = torch.zeros(y.size())
                    ones = torch.ones(y.size())
                    y = y.cpu()
                    # make mask
                    y = torch.where(y > 0.9, ones, zeros)
                    dice = compute_per_channel_dice(y, labels_train.cpu())

                    # Progress report (http://localhost:8097) (python -m visdom.server)
                    logger.log({'loss':train_loss}, 
                                images={'image': images_train, 'pred': y, 'GT':labels_train})
                    print('Dice:',dice)
                    

                    #validation#
                    # q_params = self.model.fast_params
                    # base_net_params =  self.model.forward()
                    images_test, labels_test = trains.float(), labels.float()
                    f_model = torch_module_to_functional(torch_net=self.model)
                    
                    logits = f_model.forward(images_test, params=temp_weights)
                    test_loss = self.loss_fn(logits,labels_test)
                    y = logits.sigmoid()
                    zeros = torch.zeros(y.size())
                    ones = torch.ones(y.size())
                    y = y.cpu()
                    # make mask
                    y = torch.where(y > 0.9, ones, zeros)
                    dice = compute_per_channel_dice(y, labels_test.cpu())
                    print("Testdice:",dice)
                    
                    meta_train_loss += test_loss
                    
                    
                    


            self.meta_optimizer.zero_grad()
            meta_train_loss.backward(retain_graph =True)
            self.meta_optimizer.step()
            torch.save(self.model.state_dict(), '/home/vfuser/sungjoo/Meta_Unet/exp/'+str(ite)+'_MAML.pth')
    
    def train(self):
        for i in range(self.n_train_step):
            self.train_step()




def unseen_fourth_mod():
    model = Modified2DUNet(1, 1, 8)
    test_path = ['/home/vfuser/sungjoo/data/meta_GE/test/mask/*',
                        '/home/vfuser/sungjoo/data/dongsam_test/dong_test/mask/*',
                        '/home/vfuser/sungjoo/data/dongsam_test/sam_test/mask/*']
    curr_test_modality = test_path[2]
    test_dataset = VertebralDataset(curr_test_modality,  augmentation=False)
    
    batImageGenTests = DataLoader(test_dataset,
                                batch_size=2,
                                sampler=SequentialSampler(test_dataset),
                                shuffle=False,
                                num_workers=0)
    
    ds = iter(batImageGenTests)
    print(" <<<<<< length of the test loader >>>>> ", len(ds))

    PATH = '/home/vfuser/sungjoo/Meta_Unet/exp/124_MAML.pth'
    pretrained_dict = torch.load(PATH, map_location=torch.device('cuda'))
    print(">>>> let us see the state pretrained_dict", PATH)
    
    
    model.load_state_dict(pretrained_dict)

    num_images = len(ds)

    for im in range(num_images):
        trains, labels = next(ds)
        trains = trains.squeeze(0)
        labels = labels.squeeze(0)
            

        images_test, labels_test = trains.float(), labels.float()
        outputs = model(images_test)
        
        
    
        y = outputs.sigmoid()
        zeros = torch.zeros(y.size())
        ones = torch.ones(y.size())
        y = y.cpu()
        # make mask
        y = torch.where(y > 0.9, ones, zeros)
        dice = compute_per_channel_dice(y, labels_test.cpu())
        print("dice:",dice)



            

            

            

                

               

            

       

###### call the model here
# baseline = ModelBaseline()
# baseline.train()

# baseline.unseen_fourth_mod()
unseen_fourth_mod()