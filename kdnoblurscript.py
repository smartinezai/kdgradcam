from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
#from thesisfolder.knowledge_distillation_pytorch.model import data_loader
from thesisfolder.knowledge_distillation_pytorch.model.data_loader import fetch_dataloader, fetch_subset_dataloader
from thesisfolder.methods import DotDict, load_checkpoint, accuracy, posteval, visual_heatmap_compare, fullkdtrainrun
import wandb


import os
import torch
from torch.nn import init
import torch.nn as nn
import random
from torch.nn import Module
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Available device = ", device)

       
simpnetparams ={
    "model_version": "cnn_distill",
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
    "num_workers": 4,
    "cam_layer": -1
}
params = simpnetparams
params = DotDict(simpnetparams)
params.cuda = torch.cuda.is_available()
#eval_config = {"student": "simpnet", "teacher": "resnext29", "batch_size": 4, 'experiment': 'TopClass', "KLDgamma": 1}
#heatmapeval(eval_config)
random.seed(230)
torch.manual_seed(230)
teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
teacher_model = nn.DataParallel(teacher_model).cuda()
#print( "teacher model")
#print(teacher_model)
#val_dl = fetch_subset_dataloader('dev', params)
train_dl = fetch_dataloader('train', params)
val_dl = fetch_dataloader('dev', params)
#full_test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Select the first 100 indices to create a subset
#subset_indices = range(2)
#subset_dataset = Subset(val_dl.dataset, subset_indices)
#val_dl = DataLoader(subset_dataset, batch_size=4, shuffle=False)



# Create a DataLoader for the subset
#val_dl = fetch_dataloader('dev', params)
print("initialising teacher model")
teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
load_checkpoint(teacher_checkpoint, teacher_model);
#teacher_model = nn.DataParallel(teacher_model).cuda()
print("teacher model loaded")
teacher_model.to(device);
#print("teacher model")
#print(teacher_model)
if params.model_version == "resnet18_distill" or params.model_version == "base_resnet18":
    print("resnet student")
    runname_model = "resnet18_KD"
    student_model = ResNet18()
    student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
elif params.model_version == "cnn_vanilla" or params.model_version == "cnn_distill":
    runname_model = "simpnet_KD"     
    print("Simpnet student")       
    student_model = Net(params).cuda() if params.cuda else Net(params)
    student_checkpoint = '/home/smartinez/thesisfolder/modelruns/simpnet_noKD_best.pth.tar'
else:
    print("student isn't simpnet nor resnet18")    
student_model.to(device)
load_checkpoint(student_checkpoint, student_model)
metrics = {
        'accuracy': accuracy
        }       
#with wandb.init(config=config):
#        config = wandb.config
simpnet_config = { "student": runname_model+"noblur", "teacher": "resnext29", "batch_size": 64, 'experiment': 'TopClass', "KLDgamma": 1}
#wandb.init(project="blurmaskruns", config=simpnet_config)
simpnet_config = DotDict(simpnet_config)
runname = runname_model +"noblur"
fullkdtrainrun(runname = runname, projectname="blurmaskruns", config=simpnet_config, params= params, train_dl = train_dl, val_dl =val_dl )
#config = DotDict(simpnet_config)
#images, labels = next(iter(val_dl))
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#images, labels = images.to(device), labels.to(device)

#visual_heatmap_compare(student_model, teacher_model, val_dl, params)
#config = simpnet_config    
#val_metrics = posteval(student_model, teacher_model, val_dl, metrics, params, experiment = config.experiment, KLDgamma= config.KLDgamma, algo= "GradCAM")
        
#val_acc = val_metrics['accuracy']
#print("validation accuracy: "+str(val_acc))
#val_kl_loss = val_metrics['kl_loss']
#print("validation kl_loss: "+str(val_kl_loss))

#val_heatmap_dissimilarity = val_metrics['heatmap_dissimilarity']
#print("validation heatmap_dissimilarity: "+str(val_heatmap_dissimilarity))
#wandb.log({"val_acc": val_acc})            
#wandb.log({"val_kl_loss": val_kl_loss})       
#wandb.log({"val_heatmap_dissimilarity": val_heatmap_dissimilarity}) 