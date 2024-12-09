from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, camblur_CIFAR10, fullblurkdtrainrun
import wandb
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import v2

#from thesisfolder.pytorch_grad_cam.pytorch_grad_cam import grad_cam 
import os
import torch
from torch.nn import init
import torch.nn as nn
import random
from torch.nn import Module
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB__SERVICE_WAIT"] = "300"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Available device = ", device)

       
params ={
    "model_version": "resnet18_distill",
    "subset_percent": 1.0,
    "augmentation": "yes",
    "teacher": "resnext29",
    "alpha": 0.9,
    "temperature": 20,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "num_epochs": 50,
    "dropout_rate": 0.5, 
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 0,
    "cam_layer": 2
}
params = DotDict(params)
params.cuda = torch.cuda.is_available()
random.seed(230)
torch.manual_seed(230)
# student_model = Net(params).cuda() if params.cuda else Net(params)
# student_model.to(device)

if params.augmentation == "yes":
    train_transformer = transforms.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),  # randomly flip image horizontally
        v2.ConvertDtype(dtype=torch.float32),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
# else:
#     train_transformer = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
dev_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = camblur_CIFAR10(root='/home/smartinez/thesisfolder/knowledge_distillation_pytorch/data-cifar10', train=True,
        download=False, transform=train_transformer, cam_layer=params.cam_layer)
train_dl = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=False)

    
val_dl = fetch_dataloader('dev', params)
#full_test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Select the first 100 indices to create a subset
#subset_indices = range(2)
#subset_dataset = Subset(val_dl.dataset, subset_indices)
#val_dl = DataLoader(subset_dataset, batch_size=4, shuffle=False)



# Create a DataLoader for the subset
#val_dl = fetch_dataloader('dev', params)
#print("initialising teacher model")
teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
#load_checkpoint(teacher_checkpoint, teacher_model);
#teacher_model = nn.DataParallel(teacher_model).cuda()
#print("teacher model loaded")
#teacher_model.to(device);
#print("teacher model")
#print(teacher_model)

config = { "student": params.model_version, "teacher": "None", "batch_size": 64,'layer_cam': params.cam_layer}


config = DotDict(config)
if params.model_version == "resnet18_distill" or params.model_version == "base_resnet18" or params.model_version == "resnet18_vanilla":
    runname_model = "resnet18kd"
    student_model = ResNet18().cuda() if params.cuda else ResNet()
    if params.model_version == "resnet18_distill":
        print("resnet18 KD student")
        config.student = "resnet18_KD"
        student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
    if params.model_version == "resnet18_vanilla" or params.model_version == "base_resnet18":
        params.model_version = "base_resnet18"
        config.student = "resnet18_vanilla"
        print("resnet18 vanilla student")
        student_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnet18/best.pth.tar'
elif params.model_version == "cnn_distill" or params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
    runname_model = "simpnetkd"            
    student_model = Net(params).cuda() if params.cuda else Net(params)
    if params.model_version == "base_cnn":
        config.student = "simpnet_vanilla"
        print("student: simpnet vanilla")
      
           
    
else:
    print("student is neither simpnet nor resnet18")    
student_model.to(device)

metrics = {
        'accuracy': accuracy
        }       

runname = runname_model +"_layercam" + str(params.cam_layer)
#print(f"current configuration: {config}")
fullblurkdtrainrun(runname = runname, projectname="blurmaskruns", config=config ,params = params, train_dl = train_dl, val_dl = val_dl)
