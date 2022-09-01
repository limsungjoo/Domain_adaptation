import os
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
torch.autograd.set_detect_anomaly(True)
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
        self.epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(comment='spine-baseline-train-june21')

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

        self.test_path = ['/home/vfuser/sungjoo/data/dongsam_test/dong/*']

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

    def train(self):
        logger = Logger(self.epochs, len(self.TrainMetaData))
        print('>>>>>>>>>TRAINING<<<<<<<<')
        
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
        # d4 = iter(self.batImageGenTrains[3])

        ite = 0
        total_loss=0.0
        for epoch in range(self.epochs):
            print("<<<<<<<<< epoch >>>>>>>>", epoch)
            d1 = iter(self.batImageGenTrains[0])
            d2 = iter(self.batImageGenTrains[1])
            d3 = iter(self.batImageGenTrains[2])
            # d4 = iter(self.batImageGenTrains[3])

            for im in range(6):

                trains_d1, labels_d1 = next(d1)
                trains_d1 = trains_d1.squeeze(0)
                labels_d1 = labels_d1.squeeze(0)

                trains_d2, labels_d2 = next(d2)
                trains_d2 = trains_d2.squeeze(0)
                labels_d2 = labels_d2.squeeze(0)

                trains_d3, labels_d3 = next(d3)
                trains_d3 = trains_d3.squeeze(0)
                labels_d3 = labels_d3.squeeze(0)

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
                total_loss=0.0
                for index in range(3):
                    
                    if index==0:
                        trains, labels = trains_d1, labels_d1
                    if index==1:
                        trains, labels = trains_d2, labels_d2
                    if index==2:
                        trains, labels = trains_d3, labels_d3
                    # if index==3:
                    #     trains, labels = trains_d4, labels_d4
                    
                    
                    images_train, labels_train = trains.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                    # images_train, labels_train = Variable(images_train, requires_grad=False).float().cuda(), Variable(labels_train, requires_grad=False).float().cuda()
                    
                    # print(labels_train_2.shape) 
                    outputs, predictions_train = self.network(x=images_train, meta_step_size=0.001, meta_loss=None, stop_gradient=False)
                    
                    loss = self.loss_fn(outputs, labels_train)
                    total_loss +=loss
                    
                    y = outputs.sigmoid()
                    zeros = torch.zeros(y.size())
                    ones = torch.ones(y.size())
                    y = y.cpu()

                    # make mask
                    y = torch.where(y > 0.9, ones, zeros)
                    dice = compute_per_channel_dice(y, labels_train.cpu())

                    predictions_train = predictions_train.cpu().data.numpy()
                    predicted_classes = np.argmax(predictions_train, axis=1)

                    train_acc = compute_accuracy(predictions=predicted_classes, labels=labels_train.cpu().data.numpy())
                    total_train_acc.append(train_acc)

                    total_train_acc.append(train_acc)
                    # del loss, outputs
                    
                    print('Train/Accuracy', np.mean(total_train_acc))
                    print('Dice:',dice)
                    # self.writer.add_scalar('Train/Loss', total_loss.data, ite)
                    # self.writer.add_scalar('Train/Accuracy', np.mean(total_train_acc), ite)
                    # init the grad to zeros first
                    self.optimizer.zero_grad()
                    # backpropagate your network
                    loss.backward(retain_graph =True)
                    logger.log({'loss':loss}, 
                                images={'image': images_train, 'pred': y, 'GT':labels_train})
                    
                    # optimize the parameters
                    # self.scheduler.step()
                    self.optimizer.step()

                    print('ite:', ite, 'loss:', total_loss.cpu().data.numpy(), 'lr:', self.scheduler.get_lr()[0])
                    
                    # del total_loss
                    # torch.cuda.empty_cache()

                    if ite % self.test_every == 0:
                        print('>>>>>>>>> VALIDATION <<<<<<<<')
                        self.test(ite)

            self.scheduler.step()


    def test(self, ite):
        self.network.eval()

        d1 = iter(self.batImageGenVals[0])
        d2 = iter(self.batImageGenVals[1])
        d3 = iter(self.batImageGenVals[2])
        # d4 = iter(self.batImageGenVals[3])


        print("length of the three loaders >>>>> ", len(d1), len(d2),len(d3))
        accuracies = []
        val_losses = []

        for im in range(3):

            for index in range(3):
                if index == 0:
                    try:
                        trains, labels = next(d1)
                    except StopIteration:
                        d1 = iter(self.batImageGenVals[0])
                        trains, labels = next(d1)

                if index == 1:
                    try:
                        trains, labels = next(d2)
                    except StopIteration:
                        d2 = iter(self.batImageGenVals[1])
                        trains, labels = next(d2)
                if index == 2:
                    try:
                        trains, labels = next(d3)
                    except StopIteration:
                        d3 = iter(self.batImageGenVals[2])
                        trains, labels = next(d3)

                


                
                trains = trains.squeeze(0)
                labels = labels.squeeze(0)
                

                

                images_test, labels_test = trains, labels

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                # images_test = Variable(images_test).cuda()
                # labels_test = Variable(labels_test).float().cuda()
                try : 
                    outputs, predictions = self.network(x=images_test, meta_step_size=0.001, meta_loss=None,
                                                        stop_gradient=False)
                except:
                    print('e')
                val_loss = self.loss_fn(outputs, labels_test)
                val_loss_data = val_loss.cpu().data.numpy()
                y = outputs.sigmoid()
                zeros = torch.zeros(y.size())
                ones = torch.ones(y.size())
                y = y.cpu()

                # make mask
                y = torch.where(y > 0.9, ones, zeros)
                
                dice = compute_per_channel_dice(y, labels_test.cpu())
                predictions = predictions.cpu().data.numpy()
                predicted_classes = np.argmax(predictions, axis=1)

                accuracy_val = compute_accuracy(predictions=predicted_classes, labels=labels_test.cpu().data.numpy())

                accuracies.append(accuracy_val)
                val_losses.append(val_loss_data)

                print("----------accuracy val----------", accuracy_val)
                print("Test Dice:",dice)
               
                # self.writer.add_scalar('Validation/Accuracy', accuracy_val, ite) #mean_acc
                # self.writer.add_scalar('Validation/Loss', val_loss.data, ite) #mean_val_loss

                

        # self.network.train()
        mean_acc = np.mean(accuracies)
        mean_val_loss = np.mean(val_losses)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc
            print("--------best validation accuracy--------", self.best_accuracy_val)
            

    
            outfile = os.path.join('/home/vfuser/sungjoo/Meta_Unet/exp', str(ite)+'_baseline_spine.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

        self.writer.close()


    def unseen_fourth_mod(self):
        self.network.eval()

        ds = iter(self.batImageGenTests)
        print(" <<<<<< length of the test loader >>>>> ", len(ds))

        PATH = '/home/vfuser/sungjoo/Meta_Unet/exp/baseline_spine_xvert.tar'
        tmp=torch.load(PATH)
        pretrained_dict=tmp['state']
        print(">>>> let us see the state pretrained_dict", tmp['ite'])
        model_dict=self.network.state_dict()
        pretrained_dict={k:v for k, v in pretrained_dict.items() if k in model_dict and v.size()==model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        num_images = len(ds)
        accuracies = []
        val_losses = []
        it=0
        acc_four_image = []

        for im in range(num_images):
            trains, labels = next(ds)
            trains = trains.squeeze(0)
            labels = labels.squeeze(0)

            bs = 1
            images_train_three_domains_shape = np.shape(trains)
            total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            print("total number of batches >>>>>>>>", total_batches_per_image)
            batch_index = 0

            acc_this_image = []

            for batch in range(1, total_batches_per_image + 1):

                images_test, labels_test = trains[batch_index: batch_index + bs, :, :, :], labels[batch_index: batch_index + bs, :, :]

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                images_test = Variable(images_test).cuda()
                labels_test = Variable(labels_test).float().cuda()

                outputs, predictions = self.network(images_test, meta_step_size=0.001, meta_loss=None,
                                                        stop_gradient=False)

                val_loss = self.loss_fn(outputs, labels_test.long())
                val_loss_data = val_loss.cpu().data.numpy()

                predictions = predictions.cpu().data.numpy()
                predicted_classes = np.argmax(predictions, axis=1)

                l_test = labels_test.cpu().data.numpy()

                if np.sum(l_test) != 0:

                    accuracy_val = compute_accuracy(predictions=predicted_classes, labels=labels_test.cpu().data.numpy())
                    accuracies.append(accuracy_val)

                    imk = np.reshape(images_test.cpu(), (64,64,64))
                    imk = imk[32,:,:]
                    imk = np.reshape(imk, (1,64,64))

                    lb = np.reshape(labels_test.cpu(), (64,64,64))
                    lb = lb[32,:,:]
                    lb = np.reshape(lb, (1,64,64))

                    pre = np.reshape(predicted_classes, (64,64,64))
                    pre = pre[32,:,:]
                    pre = np.reshape(pre, (1,64,64))

                    img_batch = np.stack((imk,lb,pre))
                    self.writer.add_image('Test/three' + str(it), img_batch, dataformats='NCHW') # bs,1,64,64,64
                    self.writer.add_scalar('Test/Dice', accuracy_val, it) #mean_acc

                    #if it%100==0:
                    #    print("----------accuracy unseen fourth modality data----------", accuracy_val)
                    it = it + 1
                    acc_this_image.append(accuracy_val)

                val_losses.append(val_loss_data)

                del outputs, val_loss

                batch_index += bs

            mean_acc_this_image = np.mean(acc_this_image)
            acc_four_image.append(mean_acc_this_image)
            print("----------accuracy this image----------", mean_acc_this_image)

        mean_acc = np.mean(accuracies)
        mean_val_loss = np.mean(val_losses)
        print("---------- final test accuracy for unseen fourth modality data----------", mean_acc)

        std_acc = np.std(acc_four_image)
        print("--- the four values ---", acc_four_image)
        print("---------- final test std for unseen fourth modality data----------", std_acc)

###### call the model here
baseline = ModelBaseline()
baseline.train()
#baseline.unseen_fourth_mod()
