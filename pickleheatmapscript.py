import pickle
from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare, heatmap, annotate_heatmap, percentchangeheatmap,  KD_diss_compare_to_reference, pickledposteval
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


simpnet_KD_noblur_params ={
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
    "num_workers": 4,
    "cam_layer": -1
}
simpnet_KD_noblur_params = DotDict(simpnet_KD_noblur_params)
simpnet_KD_noblur_params.cuda = torch.cuda.is_available()

simpnet_vanilla_noblur_params ={
    "model_version": "base_cnn",
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
    "num_workers": 4,
    "cam_layer": -1
}
simpnet_vanilla_noblur_params = DotDict(simpnet_vanilla_noblur_params)
simpnet_vanilla_noblur_params.cuda = torch.cuda.is_available()



simpnet_vanilla_blurlayer_1_params ={
    "model_version": "base_cnn",
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
    "num_workers": 4,
    "cam_layer": 1
}
simpnet_vanilla_blurlayer_1_params = DotDict(simpnet_vanilla_blurlayer_1_params)
simpnet_vanilla_blurlayer_1_params.cuda = torch.cuda.is_available()

simpnet_KD_blurlayer_1_params ={
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
    "num_workers": 4,
    "cam_layer": 1
}
simpnet_KD_blurlayer_1_params = DotDict(simpnet_KD_blurlayer_1_params)
simpnet_KD_blurlayer_1_params.cuda = torch.cuda.is_available()




simpnet_vanilla_blurlayer_2_params ={
    "model_version": "base_cnn",
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
    "num_workers": 4,
    "cam_layer": 2
}
simpnet_vanilla_blurlayer_2_params = DotDict(simpnet_vanilla_blurlayer_2_params)
simpnet_vanilla_blurlayer_2_params.cuda = torch.cuda.is_available()

simpnet_KD_blurlayer_2_params ={
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
    "num_workers": 4,
    "cam_layer": 2
}
simpnet_KD_blurlayer_2_params = DotDict(simpnet_KD_blurlayer_2_params)
simpnet_KD_blurlayer_2_params.cuda = torch.cuda.is_available()




simpnet_vanilla_blurlayer_3_params ={
    "model_version": "base_cnn",
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
    "num_workers": 4,
    "cam_layer": 3
}
simpnet_vanilla_blurlayer_3_params = DotDict(simpnet_vanilla_blurlayer_3_params)
simpnet_vanilla_blurlayer_3_params.cuda = torch.cuda.is_available()

simpnet_KD_blurlayer_3_params ={
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
    "num_workers": 4,
    "cam_layer": 3
}
simpnet_KD_blurlayer_3_params = DotDict(simpnet_KD_blurlayer_3_params)
simpnet_KD_blurlayer_3_params.cuda = torch.cuda.is_available()









resnet18_KD_noblur_params ={
    "model_version": "resnet18_distill",
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
    "num_workers": 4,
    "cam_layer": -1
}
resnet18_KD_noblur_params = DotDict(resnet18_KD_noblur_params)
resnet18_KD_noblur_params.cuda = torch.cuda.is_available()

resnet18_vanilla_noblur_params ={
    "model_version": "base_resnet18",
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
    "num_workers": 4,
    "cam_layer": -1
}
resnet18_vanilla_noblur_params = DotDict(resnet18_vanilla_noblur_params)
resnet18_vanilla_noblur_params.cuda = torch.cuda.is_available()






resnet18_vanilla_blurlayer_1_params ={
    "model_version": "base_resnet18",
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
    "num_workers": 4,
    "cam_layer": 1
}
resnet18_vanilla_blurlayer_1_params = DotDict(resnet18_vanilla_blurlayer_1_params)
resnet18_vanilla_blurlayer_1_params.cuda = torch.cuda.is_available()

resnet18_KD_blurlayer_1_params ={
    "model_version": "resnet18_distill",
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
    "num_workers": 4,
    "cam_layer": 1
}
resnet18_KD_blurlayer_1_params = DotDict(resnet18_KD_blurlayer_1_params)
resnet18_KD_blurlayer_1_params.cuda = torch.cuda.is_available()




resnet18_vanilla_blurlayer_2_params ={
    "model_version": "base_resnet18",
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
    "num_workers": 4,
    "cam_layer": 2
}
resnet18_vanilla_blurlayer_2_params = DotDict(resnet18_vanilla_blurlayer_2_params)
resnet18_vanilla_blurlayer_2_params.cuda = torch.cuda.is_available()

resnet18_KD_blurlayer_2_params ={
    "model_version": "resnet18_distill",
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
    "num_workers": 4,
    "cam_layer": 2
}
resnet18_KD_blurlayer_2_params = DotDict(resnet18_KD_blurlayer_2_params)
resnet18_KD_blurlayer_2_params.cuda = torch.cuda.is_available()




resnet18_vanilla_blurlayer_3_params ={
    "model_version": "base_resnet18",
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
    "num_workers": 4,
    "cam_layer": 3
}
resnet18_vanilla_blurlayer_3_params = DotDict(resnet18_vanilla_blurlayer_3_params)
resnet18_vanilla_blurlayer_3_params.cuda = torch.cuda.is_available()

resnet18_KD_blurlayer_3_params ={
    "model_version": "resnet18_distill",
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
    "num_workers": 4,
    "cam_layer": 3
}
resnet18_KD_blurlayer_3_params = DotDict(resnet18_KD_blurlayer_3_params)
resnet18_KD_blurlayer_3_params.cuda = torch.cuda.is_available()


















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


with open('/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_noblur.pkl', 'rb') as file:
    val_cams_simpnet_vanilla_noblur = pickle.load(file)    
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_KD_noblur.pkl', 'rb') as file:
    val_cams_simpnetKD_noblur = pickle.load(file)


with open('/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_blur_layer_1.pkl', 'rb') as file:
    val_cams_simpnet_vanilla_blur_layer1 = pickle.load(file)        
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_blur_layer_2.pkl', 'rb') as file:
    val_cams_simpnet_vanilla_blur_layer2 = pickle.load(file)     
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_simpnet_vanilla_blur_layer_3.pkl', 'rb') as file:
    val_cams_simpnet_vanilla_blur_layer3 = pickle.load(file)     
    
    
    
    
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_noblur.pkl', 'rb') as file:
    val_cams_resnet18_vanilla_noblur = pickle.load(file) 
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_noblur.pkl', 'rb') as file:
    val_cams_resnet18KD_noblur = pickle.load(file)
   
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_blur_layer_1.pkl', 'rb') as file:
    val_cams_resnet18_vanilla_blur_layer1 = pickle.load(file)        
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_blur_layer_2.pkl', 'rb') as file:
    val_cams_resnet18_vanilla_blur_layer2 = pickle.load(file)     
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_vanilla_blur_layer_3.pkl', 'rb') as file:
    val_cams_resnet18_vanilla_blur_layer3 = pickle.load(file)     
    
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_blur_layer_1.pkl', 'rb') as file:
    val_cams_resnet18_KD_blur_layer1 = pickle.load(file)        
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_blur_layer_2.pkl', 'rb') as file:
    val_cams_resnet18_KD_blur_layer2 = pickle.load(file)     
with open('/home/smartinez/thesisfolder/heatmaps/val_dl_resnet18_KD_blur_layer_3.pkl', 'rb') as file:
    val_cams_resnet18_KD_blur_layer3 = pickle.load(file)     
    
    
    
    
    

    
    
    
with open('/home/smartinez/thesisfolder/val_dl_cam_dict_resnext_teach_only.pkl', 'rb') as file:
    val_cams_resnext29 = pickle.load(file)    

simpnet_vanilla_noblur_config = { "student": "simpnet_vanilla_noblur", "teacher": "None", "cam_layer" : -1 }
simpnet_KD_noblur_config = { "student": "simpnet_KD_noblur", "teacher": "resnext29", "cam_layer" : -1}

simpnet_vanilla_blurlayer1_config = { "student": "simpnet_vanilla_blur1", "teacher": "None", "cam_layer" : 1 }
simpnet_vanilla_blurlayer2_config = { "student": "simpnet_vanilla_blur2", "teacher": "None", "cam_layer" : 2 }
simpnet_vanilla_blurlayer3_config = { "student": "simpnet_vanilla_blur3", "teacher": "None", "cam_layer" : 3 }




resnet18_vanilla_noblur_config = { "student": "resnet18_vanilla_noblur", "teacher": "None",  "cam_layer" : -1}
resnet18_KD_noblur_config = { "student": "resnet18_KD_noblur", "teacher": "resnext29", "cam_layer" : -1}

resnet18_vanilla_blurlayer1_config = { "student": "resnet18_vanilla_blur1", "teacher": "None", "cam_layer" : 1 }
resnet18_vanilla_blurlayer2_config = { "student": "resnet18_vanilla_blur2", "teacher": "None", "cam_layer" : 2 }
resnet18_vanilla_blurlayer3_config = { "student": "resnet18_vanilla_blur3", "teacher": "None", "cam_layer" : 3 }

resnet18_KD_blurlayer1_config = { "student": "resnet18_KD_blur1", "teacher": "resnext29", "cam_layer" : 1 }
resnet18_KD_blurlayer2_config = { "student": "resnet18_KD_blur2", "teacher": "resnext29", "cam_layer" : 2 }
resnet18_KD_blurlayer3_config = { "student": "resnet18_KD_blur3", "teacher": "resnext29", "cam_layer" : 3 }




resnext_config = { "student": "resnext29", "teacher": "None", "cam_layer" : -1}


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


teacher_layer_strings= [
                      
                             "stg1 BNK3",
              
                             "stg2 BNK3",
                      
                             "stg3 BNK3"]


# print("beginning with simpnetKD_noblur")
# wandb.init(project="pickledheatmapeval", config= simpnet_KD_noblur_config, name="simpnet_KD_noblur")
# simpnet_KD_noblur_avg_diss_matrix = pickledposteval(val_cams_simpnetKD_noblur, val_cams_resnext29,simpnet_KD_noblur_params)    
# wandb.finish()
# print("finished with simpnetKD")


# print("beginning with simpnetvanilla_noblur")
# wandb.init(project="pickledheatmapeval", config= simpnet_vanilla_noblur_config, name="simpnet_vanilla_noblur")
# simpnet_vanilla_noblur_avg_diss_matrix = pickledposteval(val_cams_simpnet_vanilla_noblur, val_cams_resnext29,simpnet_vanilla_noblur_params)    

# print("finished with simpnetvanilla")
# simpnet_layer_strings= ["conv1", "conv2", "conv3"]

# print("generating simpnet plot depicting dissimilarity change after using KD ")
# print(f"simpnet vanilla matrix")
# print(simpnet_vanilla_noblur_avg_diss_matrix)
# print(f"simpnet kd matrix")
# print(simpnet_KD_noblur_avg_diss_matrix)
# #simpnet_avg_diss_change_matrix = (simpnet_vanilla_noblur_avg_diss_matrix - simpnet_KD_noblur_avg_diss_matrix)#*-1
# #simpnet_avg_diss_change_matrix = simpnet_avg_diss_change_matrix / ((simpnet_vanilla_noblur_avg_diss_matrix + simpnet_KD_noblur_avg_diss_matrix)       /2)
# #simpnet_avg_diss_change_matrix = simpnet_avg_diss_change_matrix *-1
# #simpnet_avg_diss_change_matrix *= 100
# simpnet_avg_diss_change_matrix=((simpnet_KD_noblur_avg_diss_matrix - simpnet_vanilla_noblur_avg_diss_matrix) / simpnet_vanilla_noblur_avg_diss_matrix) * 100.0
# figsizetuple = (3.8,3.8)
# fig, ax = plt.subplots(figsize=figsizetuple)
# im, cbar = percentchangeheatmap(simpnet_avg_diss_change_matrix ,simpnet_layer_strings, teacher_layer_strings , ax=ax,
#                 cmap="coolwarm", cbarlabel="Percentual diff. in avg. cam diss. (MSE)")
# texts = annotate_heatmap(im, valfmt="{x:.3f}%", threshold=-6)

# fig.tight_layout()
# plt.title(f"KD impact on avg. diss. simpnet-resnext29 (noblur)", fontsize=10)
# plt.subplots_adjust(top=0.93)
# # Log the plot to wandb
# wandb.log({f"KD impact on avg. diss. simpnet-resnext29 noblur": wandb.Image(fig)})
# plt.show()
# print("finished comparing the avg heatmap dissimilarity of simpnet with and without KD ")

# # print("now comparing the dissimilarity of the vanilla and KD simpnet versions wrt to resnext29")
# # KD_diss_compare_to_reference(simpnet_vanilla, simpnet_KD, resnext, val_dl, metrics, simpnet_vanilla_noblur_params)
# wandb.finish()



# print("beginning with simpnet_vanilla_blur_layer1")
# wandb.init(project="pickledheatmapeval", config= simpnet_vanilla_blurlayer1_config, name="simpnet_vanilla_blur1")
# simpnet_vanilla_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_simpnet_vanilla_blur_layer1, val_cams_resnext29,simpnet_vanilla_blurlayer_1_params)    
# wandb.finish()
# print("finished with simpnet_vanilla_blur_layer1")


# print("beginning with simpnet_vanilla_blur_layer2")
# wandb.init(project="pickledheatmapeval", config= simpnet_vanilla_blurlayer2_config, name="simpnet_vanilla_blur2")
# simpnet_vanilla_blurlayer2_avg_diss_matrix = pickledposteval(val_cams_simpnet_vanilla_blur_layer2, val_cams_resnext29,simpnet_vanilla_blurlayer_2_params)    
# wandb.finish()
# print("finished with simpnet_vanilla_blur_layer2")


# print("beginning with simpnet_vanilla_blur_layer3")
# wandb.init(project="pickledheatmapeval", config= simpnet_vanilla_blurlayer3_config, name="simpnet_vanilla_blur3")
# simpnet_vanilla_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_simpnet_vanilla_blur_layer3, val_cams_resnext29,simpnet_vanilla_blurlayer_3_params)    
# wandb.finish()
# print("finished with simpnet_vanilla_blur_layer3")



















resnet18_layer_strings=[
                               
                               
                                "layer1.blc2",
                                
                               
                                "layer2.blc2",
                            
                                "layer4.blc2",
                                ]



print("beginning with resnet18KD_noblur")
wandb.init(project="pickledheatmapeval", config= resnet18_KD_noblur_config, name="resnet18_KD_noblur")
resnet18_KD_noblur_avg_diss_matrix = pickledposteval(val_cams_resnet18KD_noblur, val_cams_resnext29,resnet18_KD_noblur_params)    
wandb.finish()
print("finished with resnet18KD")

teacher_layer_strings= [
                      
                             "stg1 BNK3",
              
                             "stg2 BNK3",
                      
                             "stg3 BNK3"]
print("beginning with resnet18vanilla_noblur")
wandb.init(project="pickledheatmapeval", config= resnet18_vanilla_noblur_config, name="resnet18_vanilla_noblur")
resnet18_vanilla_noblur_avg_diss_matrix = pickledposteval(val_cams_resnet18_vanilla_noblur, val_cams_resnext29,resnet18_vanilla_noblur_params)    

print("finished with resnet18vanilla")

print("generating resnet18 plot depicting dissimilarity change after using KD ")
# resnet18_avg_diss_change_matrix = (resnet18_vanilla_noblur_avg_diss_matrix - resnet18_KD_noblur_avg_diss_matrix)#*-1
# resnet18_avg_diss_change_matrix = resnet18_avg_diss_change_matrix / ((resnet18_vanilla_noblur_avg_diss_matrix + resnet18_KD_noblur_avg_diss_matrix)/2)
# resnet18_avg_diss_change_matrix = resnet18_avg_diss_change_matrix *-1
# resnet18_avg_diss_change_matrix *= 100
resnet18_avg_diss_change_matrix=((resnet18_KD_noblur_avg_diss_matrix - resnet18_vanilla_noblur_avg_diss_matrix) / resnet18_vanilla_noblur_avg_diss_matrix) * 100.0

figsizetuple = (4.3,4.3)
fig, ax = plt.subplots(figsize=figsizetuple)
im, cbar = percentchangeheatmap(resnet18_avg_diss_change_matrix ,resnet18_layer_strings, teacher_layer_strings , ax=ax,
                cmap="coolwarm", cbarlabel="Percentual diff. in avg. cam diss. (MSE)")
texts = annotate_heatmap(im, valfmt="{x:.3f}%", threshold=-60)

fig.tight_layout()
plt.title(f"KD impact on avg. diss. resnet18-resnext29 (noblur)", fontsize=10)
plt.subplots_adjust(top=0.93)
# Log the plot to wandb
wandb.log({f"KD impact on avg. diss. resnet18-resnext29 noblur": wandb.Image(fig)})
plt.show()
print("finished comparing the avg heatmap dissimilarity of resnet18 with and without KD ")


print(f"resnet18 vanilla matrix")
print(resnet18_vanilla_noblur_avg_diss_matrix)
print(f"resnet18 kd matrix")
print(resnet18_KD_noblur_avg_diss_matrix)
# print("now comparing the dissimilarity of the vanilla and KD simpnet versions wrt to resnext29")
# KD_diss_compare_to_reference(simpnet_vanilla, simpnet_KD, resnext, val_dl, metrics, simpnet_vanilla_noblur_params)
wandb.finish()





print("beginning with resnet18_vanilla_blur_layer1")
wandb.init(project="pickledheatmapeval", config= resnet18_vanilla_blurlayer1_config, name="resnet18_vanilla_blur1")
resnet18_vanilla_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_resnet18_vanilla_blur_layer1, val_cams_resnext29,resnet18_vanilla_blurlayer_1_params)    
wandb.finish()
print("finished with resnet18_vanilla_blur_layer1")


print("beginning with resnet18_vanilla_blur_layer2")
wandb.init(project="pickledheatmapeval", config= resnet18_vanilla_blurlayer2_config, name="resnet18_vanilla_blur2")
resnet18_vanilla_blurlayer2_avg_diss_matrix = pickledposteval(val_cams_resnet18_vanilla_blur_layer2, val_cams_resnext29,resnet18_vanilla_blurlayer_2_params)    
wandb.finish()
print("finished with resnet18_vanilla_blur_layer2")


print("beginning with resnet18_vanilla_blur_layer3")
wandb.init(project="pickledheatmapeval", config= resnet18_vanilla_blurlayer3_config, name="resnet18_vanilla_blur3")
resnet18_vanilla_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_resnet18_vanilla_blur_layer3, val_cams_resnext29,resnet18_vanilla_blurlayer_3_params)    
wandb.finish()
print("finished with resnet18_vanilla_blur_layer3")





print("beginning with resnet18_KD_blur_layer1")
wandb.init(project="pickledheatmapeval", config= resnet18_KD_blurlayer1_config, name="resnet18_KD_blur1")
resnet18_KD_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_resnet18_KD_blur_layer1, val_cams_resnext29,resnet18_KD_blurlayer_1_params)    
wandb.finish()
print("finished with resnet18_KD_blur_layer1")


print("beginning with resnet18_KD_blur_layer2")
wandb.init(project="pickledheatmapeval", config= resnet18_KD_blurlayer2_config, name="resnet18_KD_blur2")
resnet18_KD_blurlayer2_avg_diss_matrix = pickledposteval(val_cams_resnet18_KD_blur_layer2, val_cams_resnext29,resnet18_KD_blurlayer_2_params)    
wandb.finish()
print("finished with resnet18_vKD_blur_layer2")


print("beginning with resnet18_KD_blur_layer3")
wandb.init(project="pickledheatmapeval", config= resnet18_KD_blurlayer3_config, name="resnet18_KD_blur3")
resnet18_KD_blurlayer1_avg_diss_matrix = pickledposteval(val_cams_resnet18_KD_blur_layer3, val_cams_resnext29,resnet18_KD_blurlayer_3_params)    
wandb.finish()
print("finished with resnet18_KD_blur_layer3")








































# wandb.init(project="pickledheatmapeval", config= resnet18_KD_noblur_config, name="resnet18_KD_noblur")
# resnet18_KD_noblur_avg_diss_matrix = pickledposteval(val_cams, val_cams_resnext29,resnet18_KD_noblur_params)    
# wandb.finish()
# print("finished with simpnetKD")

# teacher_layer_strings= [
                      
#                              "stg1 BNK3",
              
#                              "stg2 BNK3",
                      
#                              "stg3 BNK3"]
# print("beginning with simpnetvanilla_noblur")
# wandb.init(project="pickledheatmapeval", config= simpnet_KD_noblur_config, name="simpnet_vanilla_noblur")
# simpnet_vanilla_noblur_avg_diss_matrix = pickledposteval(val_cams_simpnet_vanilla_noblur, val_cams_resnext29,simpnet_vanilla_noblur_params)    
# wandb.finish()
# print("finished with simpnetvanilla")
# simpnet_layer_strings= ["conv1", "conv2", "conv3"]

# print("generating simpnet plot depicting dissimilarity change after using KD ")
# simpnet_avg_diss_change_matrix = (simpnet_vanilla_noblur_avg_diss_matrix - simpnet_KD_noblur_avg_diss_matrix)#*-1
# simpnet_avg_diss_change_matrix = simpnet_avg_diss_change_matrix / ((simpnet_vanilla_noblur_avg_diss_matrix + simpnet_KD_noblur_avg_diss_matrix)       /2)
# simpnet_avg_diss_change_matrix = simpnet_avg_diss_change_matrix *-1
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

# # print("now comparing the dissimilarity of the vanilla and KD simpnet versions wrt to resnext29")
# # KD_diss_compare_to_reference(simpnet_vanilla, simpnet_KD, resnext, val_dl, metrics, simpnet_vanilla_noblur_params)
# wandb.finish()







# print("beginning with resnet18KD_noblur")
# wandb.init(project="heatmapeval", config= resnet18_KD_noblur_config, name="resnet18_KD_noblur")
# val_metrics, resnet18_KD_avg_diss_matrix = posteval(resnet18_KD, resnext, val_dl, metrics, resnet18_KD_noblur_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
# val_acc = val_metrics['accuracy']
# print("validation accuracy: "+str(val_acc))
# val_kl_loss = val_metrics['kl_loss']
# print("validation kl_loss: "+str(val_kl_loss))
# wandb.log({"val_acc": val_acc})            
# wandb.log({"val_kl_loss": val_kl_loss})       
# wandb.finish()
# print("finished with resnet18KD")

# print("beginning with resnet18 vanilla")
# wandb.init(project="heatmapeval", config= resnet18_vanilla_noblur_config, name="resnet18_vanilla")
# val_metrics, resnet18_vanilla_avg_diss_matrix = posteval(resnet18_vanilla, resnext, val_dl, metrics, resnet18_vanilla_noblur_params, experiment = 'TopClass', KLDgamma= 1, algo= "GradCAM")    
# val_acc = val_metrics['accuracy']
# print("validation accuracy: "+str(val_acc))
# val_kl_loss = val_metrics['kl_loss']
# print("validation kl_loss: "+str(val_kl_loss))
# wandb.log({"val_acc": val_acc})            
# wandb.log({"val_kl_loss": val_kl_loss})       

# print("finished with resnet18 vanilla")

# print("generating resnet18 plot depicting dissimilarity change after using KD ")
# resnet18_avg_diss_change_matrix = (resnet18_vanilla_avg_diss_matrix - resnet18_KD_avg_diss_matrix) *-1
# figsizetuple = (9.5,9.5)
# fig, ax = plt.subplots(figsize=figsizetuple)
# im, cbar = heatmap(resnet18_avg_diss_change_matrix ,resnet18_layer_strings, teacher_layer_strings , ax=ax,
#                 cmap="coolwarm", cbarlabel="Diff. in avg. cam diss. (MSE)")
# texts = annotate_heatmap(im, valfmt="{x:.3f}")

# fig.tight_layout()
# plt.title(f"Impact of KD on avg. resnet18-resnext29 cam diss.", fontsize=10)
# plt.subplots_adjust(top=0.93)
# # Log the plot to wandb
# wandb.log({f"KD impact on avg. dissmat simpnet-resnext29": wandb.Image(fig)})
# plt.show()
# print("finished comparing the avg heatmap dissimilarity of simpnet with and without KD ")

# print("now comparing the dissimilarity of the vanilla and KD resnet18 versions wrt to resnext29")
# KD_diss_compare_to_reference(resnet18_vanilla, resnet18_KD, resnext, val_dl, metrics, resnet18_vanilla_noblur_params)
# wandb.finish()