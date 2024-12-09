from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare, heatmap, annotate_heatmap, KD_diss_compare_to_reference
import wandb
from thesisfolder.pytorch_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
#from thesisfolder.pytorch_grad_cam.pytorch_grad_cam import grad_cam 
import os
import torch
from torch.nn import init
import torch.nn as nn
import random
from torch.nn import Module
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Available device = ", device)

metrics = {
        'accuracy': accuracy
        }  
       
simpnet_KD_params ={
    "model_version": "cnn_distill",
    "subset_percent": 1.,
    "augmentation": "yes",
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
simpnet_KD_params = DotDict(simpnet_KD_params)
simpnet_KD_params.cuda = torch.cuda.is_available()

simpnet_vanilla_params ={
    "model_version": "base_cnn",
    "subset_percent": 1.,
    "augmentation": "yes",
    "teacher": "None",
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
simpnet_vanilla_params = DotDict(simpnet_vanilla_params)
simpnet_vanilla_params.cuda = torch.cuda.is_available()

resnet18_KD_params ={
    "model_version": "resnet18_distill",
    "subset_percent": 1.,
    "augmentation": "yes",
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
resnet18_KD_params = DotDict(resnet18_KD_params)
resnet18_KD_params.cuda = torch.cuda.is_available()

resnet18_vanilla_params ={
    "model_version": "base_resnet18",
    "subset_percent": 1.,
    "augmentation": "yes",
    "teacher": "None",
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
resnet18_vanilla_params = DotDict(resnet18_vanilla_params)
resnet18_vanilla_params.cuda = torch.cuda.is_available()

resnext_params ={
    "model_version": "resnext",
    "subset_percent": 1.,
    "augmentation": "no",
    "teacher": "None",
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
resnext_params = DotDict(resnext_params)
resnext_params.cuda = torch.cuda.is_available()
random.seed(230)
torch.manual_seed(230)
resnext = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
resnext = nn.DataParallel(resnext).cuda()

val_dl = fetch_dataloader('dev', resnet18_vanilla_params)

print("initialising teacher model")
teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
resnext.to(device);
load_checkpoint(teacher_checkpoint, resnext );
print("teacher model loaded")

resnet18_KD = ResNet18()
resnet18_KD_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
resnet18_KD.to(device)
load_checkpoint(resnet18_KD_checkpoint, resnet18_KD);

resnet18_vanilla = ResNet18()
resnet18_vanilla_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnet18/best.pth.tar'
resnet18_vanilla.to(device)
load_checkpoint(resnet18_vanilla_checkpoint, resnet18_vanilla);

simpnet_KD = Net(simpnet_KD_params).cuda() if simpnet_KD_params.cuda else Net(simpnet_KD_params)
simpnet_KD_checkpoint = '/home/smartinez/thesisfolder/modelruns/kd_nothresh_student_simplenet_teacher_resnext29_experiment_NoHeatmap_KLDgamma_100_best.pth.tar' 
simpnet_KD.to(device)
load_checkpoint(simpnet_KD_checkpoint, simpnet_KD);

simpnet_vanilla = Net(simpnet_vanilla_params).cuda() if simpnet_vanilla_params.cuda else Net(simpnet_vanilla_params)
simpnet_vanilla_checkpoint = '/home/smartinez/thesisfolder/modelruns/simpnet_noKD_best.pth.tar' 
simpnet_vanilla.to(device)
load_checkpoint(simpnet_vanilla_checkpoint, simpnet_vanilla);

print("loaded student weights")
batch_count = len(val_dl)



resnet18_vanilla_target_layers =[resnet18_vanilla.conv1, #initial conv
                                #layer1
                                #first basis block
                                resnet18_vanilla.layer1[0].conv1,
                                resnet18_vanilla.layer1[0].conv2,
                                #second basis block
                                resnet18_vanilla.layer1[1].conv1,
                                resnet18_vanilla.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                resnet18_vanilla.layer2[0].conv1,
                                resnet18_vanilla.layer2[0].conv2,
                                #second basis block
                                resnet18_vanilla.layer2[1].conv1,
                                resnet18_vanilla.layer2[1].conv2,
                                #layer3
                                #first basis block
                                resnet18_vanilla.layer3[0].conv1,
                                resnet18_vanilla.layer3[0].conv2,
                                #second basis block
                                resnet18_vanilla.layer3[1].conv1,
                                resnet18_vanilla.layer3[1].conv2,
                                #layer4
                                #first basis block
                                resnet18_vanilla.layer4[0].conv1,
                                resnet18_vanilla.layer4[0].conv2,
                                #second basis block
                                resnet18_vanilla.layer4[1].conv1,
                                resnet18_vanilla.layer4[1].conv2
                                ]

resnet18_KD_target_layers =[resnet18_KD.conv1, #initial conv
                                #layer1
                                #first basis block
                                resnet18_KD.layer1[0].conv1,
                                resnet18_KD.layer1[0].conv2,
                                #second basis block
                                resnet18_KD.layer1[1].conv1,
                                resnet18_KD.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                resnet18_KD.layer2[0].conv1,
                                resnet18_KD.layer2[0].conv2,
                                #second basis block
                                resnet18_KD.layer2[1].conv1,
                                resnet18_KD.layer2[1].conv2,
                                #layer3
                                #first basis block
                                resnet18_KD.layer3[0].conv1,
                                resnet18_KD.layer3[0].conv2,
                                #second basis block
                                resnet18_KD.layer3[1].conv1,
                                resnet18_KD.layer3[1].conv2,
                                #layer4
                                #first basis block
                                resnet18_KD.layer4[0].conv1,
                                resnet18_KD.layer4[0].conv2,
                                #second basis block
                                resnet18_KD.layer4[1].conv1,
                                resnet18_KD.layer4[1].conv2
                                ]
resnet18_layer_strings=["conv1", 
                                #layer1
                                #first basis block
                                "layer1.block1.conv1",
                                "layer1.block1.conv2",
    
                                "layer1.block2.conv1",
                                "layer1.block2.conv2",
                                
                                "layer2.block1.conv1",
                                "layer2.block1.conv2",
    
                                "layer2.block2.conv1",
                                "layer2.block2.conv2",
                                #layer3
                                "layer3.block1.conv1",
                                "layer3.block1.conv2",
    
                                "layer3.block2.conv1",
                                "layer3.block2.conv2",
                                #layer4
                                "layer4.block1.conv1",
                                "layer4.block1.conv2",
    
                                "layer4.block2.conv1",
                                "layer4.block2.conv2"
                                ]
simpnet_KD_target_layers = [simpnet_KD.conv1, simpnet_KD.conv2, simpnet_KD.conv3]
simpnet_vanilla_target_layers = [simpnet_vanilla.conv1, simpnet_vanilla.conv2, simpnet_vanilla.conv3]
simpnet_layer_strings= ["conv1", "conv2", "conv3"]
resnext_target_layers = [ 
                             resnext.module.stage_1.stage_1_bottleneck_0,
                             resnext.module.stage_1.stage_1_bottleneck_1,
                             resnext.module.stage_1.stage_1_bottleneck_2,
              
                            resnext.module.stage_2.stage_2_bottleneck_0,
                            resnext.module.stage_2.stage_2_bottleneck_1,
                            resnext.module.stage_2.stage_2_bottleneck_2,
                            
                            resnext.module.stage_3.stage_3_bottleneck_0,
                            resnext.module.stage_3.stage_3_bottleneck_1]
teacher_layer_strings= [
                              "stg1 BNK1",
                             "stg1 BNK2",
                             "stg1 BNK3",
              
                            "stg2 BNK1",
                             "stg2 BNK2",
                             "stg2 BNK3",
                            
                              "stg3 BNK1",
                             "stg3 BNK",
                             "stg3 BNK3"]


simpnet_KD_config = { "student": "simpnet_KD", "teacher": "resnext29", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}
simpnet_vanilla_config = { "student": "simpnet_vanilla", "teacher": "None", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}
resnet18_KD_config = { "student": "resnet18_KD", "teacher": "resnext29", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}
resnet18_vanilla_config = { "student": "resnet18_vanilla", "teacher": "None", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}
resnext_config = { "student": "resnext29", "teacher": "None", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}


# print("beginning with resnext29 as a baseline")
# wandb.init(project="heatmapeval", config= resnext_config, name="resnext29")
# val_metrics, resnext_diss_matrix = posteval(resnext, resnext, val_dl, metrics, resnext_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
# val_acc = val_metrics['accuracy']
# print("validation accuracy: "+str(val_acc))
# val_kl_loss = val_metrics['kl_loss']
# print("validation kl_loss: "+str(val_kl_loss))
# wandb.log({"val_acc": val_acc})            
# wandb.log({"val_kl_loss": val_kl_loss})       
# print("finished with resnext29 baseline")
# wandb.finish()





# print("beginning with simpnetKD")
# wandb.init(project="heatmapeval", config= simpnet_KD_config, name="simpnet_KD")
# val_metrics, simpnet_KD_avg_diss_matrix = posteval(simpnet_KD, resnext, val_dl, metrics, simpnet_KD_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
# val_acc = val_metrics['accuracy']
# print("validation accuracy: "+str(val_acc))
# val_kl_loss = val_metrics['kl_loss']
# print("validation kl_loss: "+str(val_kl_loss))
# wandb.log({"val_acc": val_acc})            
# wandb.log({"val_kl_loss": val_kl_loss})       
# wandb.finish()
# print("finished with simpnetKD")

# print("beginning with simpnet vanilla")
# wandb.init(project="heatmapeval", config= simpnet_vanilla_config, name="simpnet_vanilla")
# val_metrics, simpnet_vanilla_avg_diss_matrix = posteval(simpnet_vanilla, resnext, val_dl, metrics, simpnet_vanilla_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
# val_acc = val_metrics['accuracy']
# print("validation accuracy: "+str(val_acc))
# val_kl_loss = val_metrics['kl_loss']
# print("validation kl_loss: "+str(val_kl_loss))
# wandb.log({"val_acc": val_acc})            
# wandb.log({"val_kl_loss": val_kl_loss})       

# print("finished with simpnet vanilla")

# print("generating simpnet plot depicting dissimilarity change after using KD ")
# simpnet_avg_diss_change_matrix = (simpnet_vanilla_avg_diss_matrix - simpnet_KD_avg_diss_matrix) *-1
# figsizetuple = (8,8)
# fig, ax = plt.subplots(figsize=figsizetuple)
# im, cbar = heatmap(simpnet_avg_diss_change_matrix ,simpnet_layer_strings, teacher_layer_strings , ax=ax,
#                 cmap="coolwarm", cbarlabel="Diff. in avg. cam diss. (MSE)")
# texts = annotate_heatmap(im, valfmt="{x:.3f}")

# fig.tight_layout()
# plt.title(f"Impact of KD on avg. simpnet-resnext29 cam diss.", fontsize=10)
# plt.subplots_adjust(top=0.93)
# # Log the plot to wandb
# wandb.log({f"KD impact on avg. dissmat simpnet-resnext29": wandb.Image(fig)})
# plt.show()
# print("finished comparing the avg heatmap dissimilarity of simpnet with and without KD ")

# print("now comparing the dissimilarity of the vanilla and KD simpnet versions wrt to resnext29")
# KD_diss_compare_to_reference(simpnet_vanilla, simpnet_KD, resnext, val_dl, metrics, simpnet_vanilla_params)
# wandb.finish()


print("beginning with resnet18KD")
wandb.init(project="heatmapeval", config= resnet18_KD_config, name="resnet18_KD")
val_metrics, resnet18_KD_avg_diss_matrix = posteval(resnet18_KD, resnext, val_dl, metrics, resnet18_KD_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
val_acc = val_metrics['accuracy']
print("validation accuracy: "+str(val_acc))
val_kl_loss = val_metrics['kl_loss']
print("validation kl_loss: "+str(val_kl_loss))
wandb.log({"val_acc": val_acc})            
wandb.log({"val_kl_loss": val_kl_loss})       
wandb.finish()
print("finished with resnet18KD")

print("beginning with resnet18 vanilla")
wandb.init(project="heatmapeval", config= resnet18_vanilla_config, name="resnet18_vanilla")
val_metrics, resnet18_vanilla_avg_diss_matrix = posteval(resnet18_vanilla, resnext, val_dl, metrics, resnet18_vanilla_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
val_acc = val_metrics['accuracy']
print("validation accuracy: "+str(val_acc))
val_kl_loss = val_metrics['kl_loss']
print("validation kl_loss: "+str(val_kl_loss))
wandb.log({"val_acc": val_acc})            
wandb.log({"val_kl_loss": val_kl_loss})       

print("finished with resnet18 vanilla")

print("generating resnet18 plot depicting dissimilarity change after using KD ")
resnet18_avg_diss_change_matrix = (resnet18_vanilla_avg_diss_matrix - resnet18_KD_avg_diss_matrix) *-1
figsizetuple = (9.5,9.5)
fig, ax = plt.subplots(figsize=figsizetuple)
im, cbar = heatmap(resnet18_avg_diss_change_matrix ,resnet18_layer_strings, teacher_layer_strings , ax=ax,
                cmap="coolwarm", cbarlabel="Diff. in avg. cam diss. (MSE)")
texts = annotate_heatmap(im, valfmt="{x:.3f}")

fig.tight_layout()
plt.title(f"Impact of KD on avg. resnet18-resnext29 cam diss.", fontsize=10)
plt.subplots_adjust(top=0.93)
# Log the plot to wandb
wandb.log({f"KD impact on avg. dissmat simpnet-resnext29": wandb.Image(fig)})
plt.show()
print("finished comparing the avg heatmap dissimilarity of simpnet with and without KD ")

print("now comparing the dissimilarity of the vanilla and KD resnet18 versions wrt to resnext29")
KD_diss_compare_to_reference(resnet18_vanilla, resnet18_KD, resnext, val_dl, metrics, resnet18_vanilla_params)
wandb.finish()