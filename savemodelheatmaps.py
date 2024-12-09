from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
#resnext = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
#resnext = nn.DataParallel(resnext).cuda()
simpnet_vanilla_noblur = Net(params).cuda() if params.cuda else Net(params)
simpnet_vanilla_noblur_checkpoint = '/home/smartinez/thesisfolder/modelruns/vanilla_simpnet_noKD_noblur_best.pth.tar'
simpnet_vanilla_noblur.to(device)
load_checkpoint(simpnet_vanilla_noblur_checkpoint, simpnet_vanilla_noblur);

train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
trainset = indexed_CIFAR10(root='/home/smartinez/thesisfolder/knowledge_distillation_pytorch/data-cifar10', train=True,
        download=True, transform=train_transformer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

simpnet_vanilla_noblur.eval()



resnext_target_layers = [ 
                             
                             resnext.module.stage_1.stage_1_bottleneck_2,
              
                           
                            resnext.module.stage_2.stage_2_bottleneck_2,
                            
                           
                            resnext.module.stage_3.stage_3_bottleneck_2]
teacher_layer_strings= [
                            
                             "stg1 BNK3",
              
                             "stg2 BNK3",
                            
                             "stg3 BNK3"]


simpnet_target_layers = [simpnet_vanilla_noblur.conv1, simpnet_vanilla_noblur.conv2, simpnet_vanilla_noblur.conv3]
simpnet_layer_strings= ["conv1", "conv2", "conv3"]
model_cam = GradCAM(model=simpnet_vanilla_noblur, target_layers=simpnet_layer_strings)


file_path = "/home/smartinez/thesisfolder/train_dl_cam_dict_simpnetKD_noblur.pkl"

def save_heatmapset(model, dataloader, model_cam, file_path, params):
# Define batch size
    heatmap_thing = {}
    

    with tqdm(len(dataloader)) as t:
        for i, (data_batch, labels_batch, index_batch) in enumerate(dataloader):   

            if params.cuda:
                data_batch, labels_batch, index_batch = data_batch.cuda(), labels_batch.cuda(), index_batch.cuda()
        # print(f"index batch type {index_batch.dtype}")
            #print(f"index batch:")
            #print(index_batch)  

            # Compute model output for the batch


            # Compute heatmaps for the batch
            model_heatmap_stack_batch = model_cam(data_batch)
            model_heatmap_stack_batch = np.array(model_heatmap_stack_batch).squeeze()
            
            model_heatmap_stack_batch = model_heatmap_stack_batch.swapaxes(0,1)
        # print(f"shape{resnext_heatmap_stack_batch.shape}")

            for key, value in zip(index_batch, model_heatmap_stack_batch):
                #print("value shape" + value.shape)
                #print(f"key item {key.item()}")
                heatmap_thing[key.item()] = value
        
            t.update()
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(heatmap_thing, pickle_file)
    print("DONE WITH THE WHOLE THING")