from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare, save_heatmapset, save_TopTeacher_heatdict, save_TopTeacher_heatdict_indexing
# Define batch size
from torchvision.datasets.cifar import CIFAR10
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
    "model_version": "resnet18_distill",
    "subset_percent": 1.,
    "augmentation": "no",
    "teacher": "Resnext29",
    "alpha": 0.9,
    "temperature": 20,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 100,
    "dropout_rate": 0.5, 
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4,
    "cam_layer": 1
}
params = simpnetparams
params = DotDict(simpnetparams)
params.cuda = torch.cuda.is_available()

random.seed(230)
torch.manual_seed(230)
#resnext = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
#resnext = nn.DataParallel(resnext).cuda()


if params.model_version == "base_cnn" or params.model_version == "cnn_distill":
        model = Net(params).cuda() if params.cuda else Net(params)
        target_layers = [model.conv1, model.conv2, model.conv3]
        if params.model_version == "base_cnn":
                print("simpnet student, no KD")
                if params.cam_layer == -1:
                        model_checkpoint = '/home/smartinez/thesisfolder/modelruns/vanilla_simpnet_noKD_noblur_best.pth.tar'
                        print("simpnet no KD, no blur")
                        train_file_path = "/home/smartinez/thesisfolder/heatmaps/train_dl_simpnet_vanilla_noblur.pkl"
                        val_file_path = "/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_noblur.pkl"
                else:
                        print(f"simpnet student , no KD, blur layer {params.cam_layer}")
                        model_checkpoint = f'/home/smartinez/thesisfolder/modelruns/vanilla_simpnet_noKD_camlayer_{params.cam_layer}_best.pth.tar'

                        train_file_path = f"/home/smartinez/thesisfolder/heatmaps/train_dl_simpnet_vanilla_blur_layer_{params.cam_layer}.pkl"
                        val_file_path = f"/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_blur_layer_{params.cam_layer}.pkl"        
                                
        elif params.model_version == "cnn_distill":
                print("simpnet KD")
                if params.cam_layer == -1:
                        model_checkpoint = '/home/smartinez/thesisfolder/modelruns/stud_simpnet_teach_resnext29_noblur_best.pth.tar'
                        print("simpnet KD, no blur")
                        train_file_path = "/home/smartinez/thesisfolder/heatmaps/train_dl_simpnet_KD_noblur.pkl"
                        val_file_path = "/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_KD_noblur.pkl"
                else:
                        print("you're about to do KD blur which in the past had a problem with the weights, make sure you have saved them properly")
                        model_checkpoint = f'/home/smartinez/thesisfolder/modelruns/kd_blur_student_simpnet_teacher_resnext29_layer{params.cam_layer}_best.pth.tar'
                        train_file_path = f"/home/smartinez/thesisfolder/heatmaps/train_dl_simpnet_KD_blur_layer_{params.cam_layer}.pkl"
                        val_file_path = f"/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_KD_blur_layer_{params.cam_layer}.pkl"
elif params.model_version == "resnet18_distill" or params.model_version == "base_resnet18":
        model = ResNet18().cuda() if params.cuda else ResNet()
        target_layers = [model.layer1[1].conv2,

                        model.layer2[1].conv2,

                        model.layer4[1].conv2]

        if params.model_version == "base_resnet18":
                print("resnet18 student, no KD")
                if params.cam_layer == -1:
                        model_checkpoint = '/home/smartinez/thesisfolder/modelruns/vanilla_resnet18_noKD_noblur_best.pth.tar'
                        print("resnet18 , no KD, no blur")
                        train_file_path = "/home/smartinez/thesisfolder/heatmaps/train_dl_resnet18_vanilla_noblur.pkl"
                        val_file_path = "/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_noblur.pkl"
                        indexing_file_path = "/home/smartinez/thesisfolder/indexing/val_dl_resnet18_vanilla_noblur_indexing.pkl"
                else:
                        
                       print(f"resnet18 student , no KD, blur layer {params.cam_layer}") 
                       model_checkpoint = f'/home/smartinez/thesisfolder/modelruns/vanilla_resnet18_noKD_camlayer_{params.cam_layer}_best.pth.tar'
                       train_file_path = f"/home/smartinez/thesisfolder/heatmaps/train_dl_resnet18_vanilla_blur_layer_{params.cam_layer}.pkl"
                       val_file_path = f"/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_blur_layer_{params.cam_layer}.pkl"
 
 
        elif params.model_version == "resnet18_distill":
                print("resnet18 KD")
                if params.cam_layer == -1:
                        model_checkpoint = '/home/smartinez/thesisfolder/modelruns/stud_resnet18_teach_resnext29_noblur_best.pth.tar'
                        print(f"resnet18, KD but no blur")
                        
                        train_file_path = "/home/smartinez/thesisfolder/heatmaps/train_dl_resnet18_KD_noblur.pkl"
                        val_file_path = "/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_noblur.pkl"
                        indexing_file_path = "/home/smartinez/thesisfolder/indexing/val_dl_resnet18_KD_noblur_indexing.pkl"
                else:
                        print("you're about to do KD blur which in the past had a problem with the weights, make sure you have saved them properly")
                        
                        model_checkpoint = f'/home/smartinez/thesisfolder/modelruns/kd_blur_student_resnet18_teacher_resnext29_layer{params.cam_layer}_best.pth.tar'
                        train_file_path = f"/home/smartinez/thesisfolder/heatmaps/train_dl_resnet18_KD_blur_layer_{params.cam_layer}.pkl"
                        val_file_path = f"/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_blur_layer_{params.cam_layer}.pkl"  
                        indexing_file_path = f"/home/smartinez/thesisfolder/indexing/val_dl_resnet18_KD_blur_layer_{params.cam_layer}indexing.pkl"

else:
        print("model is neither simpnet nor resnet18")
model.to(device)        
print(f"model loaded to device, now loading checkpoint {model_checkpoint}")
print(f"filepath for the resulting file for the training dataset {train_file_path}")
print(f"filepath for the resulting file for the validation dataset {val_file_path}")
# #simpnet_vanilla_noblur = Net(params).cuda() if params.cuda else Net(params)
# simpnet_vanilla_noblur_checkpoint = '/home/smartinez/thesisfolder/modelruns/vanilla_simpnet_noKD_noblur_best.pth.tar'
# simpnet_vanilla_noblur = Net(params).cuda()
# simpnet_vanilla_noblur.to(device)
load_checkpoint(model_checkpoint, model);



#resnet18_ = ResNet18().cuda() if params.cuda else ResNet()

#simpnet_vanilla_noblur_checkpoint = '/home/smartinez/thesisfolder/modelruns/vanilla_simpnet_noKD_noblur_best.pth.tar'
#model = Net(params).cuda()
#model.to(device)
#load_checkpoint(simpnet_vanilla_noblur_checkpoint, model);



train_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
trainset = indexed_CIFAR10(root='/home/smartinez/thesisfolder/knowledge_distillation_pytorch/data-cifar10', train=True,
        download=True, transform=train_transformer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)




dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

valset = indexed_CIFAR10(root='/home/smartinez/thesisfolder/knowledge_distillation_pytorch/data-cifar10', train=False,
        download=True, transform=dev_transformer)
val_dl = torch.utils.data.DataLoader(valset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)


model.eval()



resnext = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
resnext = nn.DataParallel(resnext).cuda()
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




print("now trying to call gradcam ")
model_cam = GradCAM(model, target_layers)
print("finished loading the gradcam")


# save_TopTeacher_heatdict(trainloader, model_cam, resnext, train_file_path,params)
# save_TopTeacher_heatdict(val_dl, model_cam, resnext, val_file_path,params)

save_TopTeacher_heatdict_indexing(val_dl,model, model_cam, resnext, val_file_path, indexing_file_path, params)
