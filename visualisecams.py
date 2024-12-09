from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
import torchvision
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



transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10 validation dataset
val_dataset = torchvision.datasets.CIFAR10(root='/home/smartinez/thesisfolder/data-cifar10', train=False, download=False, transform=transform)



# Initialize empty lists to store images and labels
source_images = []
source_labels = []

# Access the first 5 samples
for i in range(5):
    image, label = val_dataset[i]
    source_images.append(image)
    source_labels.append(label)



# Specify the path to your HDF5 file
file_path = '/home/smartinez/thesisfolder/val_dl_cams_resnext_teach_simpnet_resnet18_stud.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    # Load each dataset
    resnextcams_data = file['resnextcams'][:5]
    resnext_topclass_labels_data = file['resnext_topclass_labels'][:5]
    resnet18_KD_cams_data = file['resnet18_KD_cams'][:5]
    resnet18_vanilla_cams_data = file['resnet18_vanilla_cams'][:5]
    simpnet_KD_cams_data = file['simpnet_KD_cams'][:5]
    simpnet_vanilla_cams_data = file['simpnet_vanilla_cams'][:5]

# Now, you can use the loaded data as needed
# For example, print the shapes of the loaded datasets
print("resnextcams_data shape:", resnextcams_data.shape)
print("resnext_topclass_labels_data shape:", resnext_topclass_labels_data.shape)
print("resnet18_KD_cams_data shape:", resnet18_KD_cams_data.shape)
print("resnet18_vanilla_cams_data shape:", resnet18_vanilla_cams_data.shape)
print("simpnet_KD_cams_data shape:", simpnet_KD_cams_data.shape)
print("simpnet_vanilla_cams_data shape:", simpnet_vanilla_cams_data.shape)

# Access the first 5 samples
for i in range(5):
    image, label = val_dataset[i]
    print(f"Image {i + 1}, Label: {label}")
    # If you want to display the image, you can use matplotlib or other image display libraries
    # For example, using matplotlib:
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1, 2, 0))  # Permute dimensions for displaying RGB image
    plt.show()