from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Subset, DataLoader
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare
import wandb
from thesisfolder.pytorch_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
#from thesisfolder.pytorch_grad_cam.pytorch_grad_cam import grad_cam 
import pickle
import os
import torch
from torch.nn import init
import torch.nn as nn
import random
from torch.nn import Module
import numpy as np
import h5py
from torchvision.datasets.cifar import CIFAR10
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class indexed_CIFAR10(CIFAR10):
    def __getitem__(self, index:int ):
        return super().__getitem__(index) + (index,)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Available device = ", device)

       
simpnetparams ={
    "model_version": "cnn_distill",
    "subset_percent": 1.,
    "augmentation": "no",
    "teacher": "resnext29",
    "alpha": 0.9,
    "temperature": 20,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 100,
    "dropout_rate": 0.5, 
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
params = simpnetparams
params = DotDict(simpnetparams)
params.cuda = torch.cuda.is_available()

random.seed(230)
torch.manual_seed(230)
resnext = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
resnext = nn.DataParallel(resnext).cuda()

train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
devset = indexed_CIFAR10(root='/home/smartinez/thesisfolder/knowledge_distillation_pytorch/data-cifar10', train=False,
        download=True, transform=train_transformer)
trainloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)
print("initialising teacher model")
teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
resnext.to(device);
load_checkpoint(teacher_checkpoint, resnext );
print("teacher model loaded")
resnext.eval()



resnext_target_layers = [ 
                             
                             resnext.module.stage_1.stage_1_bottleneck_2,
              
                           
                            resnext.module.stage_2.stage_2_bottleneck_2,
                            
                           
                            resnext.module.stage_3.stage_3_bottleneck_2]
teacher_layer_strings= [
                            
                             "stg1 BNK3",
              
                             "stg2 BNK3",
                            
                             "stg3 BNK3"]

resnext_cam = GradCAM(model=resnext, target_layers=resnext_target_layers)


# file_path = "/home/smartinez/thesisfolder/train_dl_cam_dict_resnext_teach_only.h5"
# with h5py.File(file_path, 'w') as hf:
#     resnext29_cam_dict = {}
#     with tqdm(total=trainset.data.shape[0]) as t:
#         for i in range(len(trainset)):
#                 # move to GPU if available
        

#            # sample = Variable(sample)
#                 # compute model output
#                 #with torch.no_grad():
#             sample, label = trainset[i]    
#             #print(sample.shape)
            
#             sample = sample.unsqueeze(0)
        
#             resnext_output = resnext(sample)
            
#            # teacher_preds = torch.sigmoid(resnext_output)
#            # teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
#            # max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
                    
#             resnext_heatmap_stack = resnext_cam(sample)

#             resnext_heatmap_stack = np.array(resnext_heatmap_stack).squeeze()
         
#            # resnext_heatmap_stack = resnext_heatmap_stack.swapaxes(0,1)
          
#             #print(f"resnext heatmap stack shape after axis swap: {resnext_heatmap_stack.shape}")
          

#             resnext29_cam_dict[sample] = resnext_heatmap_stack

#             t.update() 
 
#     hf.create_dataset('resnext29cams', data=resnext29_cam_dict)

# file_path = "/home/smartinez/thesisfolder/train_dl_cam_dict_resnext_teach_only.pkl"
# resnext29_cam_dict = {}

# with tqdm(total=trainset.data.shape[0]) as t:
#     for i in range(len(trainset)):
#         sample, label = trainset[i]    
#         sample = sample.unsqueeze(0)
     
#         #resnext_output = resnext(sample)
#         resnext_heatmap_stack = resnext_cam(sample)
#         resnext_heatmap_stack = np.array(resnext_heatmap_stack).squeeze()
      
#         resnext29_cam_dict[sample] = resnext_heatmap_stack
#         t.update() 

# with open(file_path, 'wb') as pickle_file:
#     pickle.dump(resnext29_cam_dict, pickle_file)
#     print("DONE WITH THE WHOLE THING")
        


file_path = "/home/smartinez/thesisfolder/val_dl_cam_dict_resnext_teach_only.pkl"
resnext29_cam_dict = {}

# Define batch size
batch_size = 64
heatmap_thing = {}

# Define batch size
batch_size = 64
with tqdm(len(trainloader)) as t:
    for i, (data_batch, labels_batch, index_batch) in enumerate(trainloader):   

        if params.cuda:
            data_batch, labels_batch, index_batch = data_batch.cuda(), labels_batch.cuda(), index_batch.cuda()
       # print(f"index batch type {index_batch.dtype}")
        #print(f"index batch:")
        #print(index_batch)  

        # Compute model output for the batch


        # Compute heatmaps for the batch
        resnext_heatmap_stack_batch = resnext_cam(data_batch)
        resnext_heatmap_stack_batch = np.array(resnext_heatmap_stack_batch).squeeze()
        
        resnext_heatmap_stack_batch = resnext_heatmap_stack_batch.swapaxes(0,1)
       # print(f"shape{resnext_heatmap_stack_batch.shape}")

        for key, value in zip(index_batch, resnext_heatmap_stack_batch):
            #print("value shape" + value.shape)
            #print(f"key item {key.item()}")
            heatmap_thing[key.item()] = value
    
        t.update()
with open(file_path, 'wb') as pickle_file:
    pickle.dump(heatmap_thing, pickle_file)
print("DONE WITH THE WHOLE THING")