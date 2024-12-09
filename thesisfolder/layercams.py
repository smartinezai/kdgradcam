from knowledge_distillation_pytorch.model import net
from knowledge_distillation_pytorch.model import resnet
from knowledge_distillation_pytorch.model import resnext
from pytorch_grad_cam.pytorch_grad_cam import grad_cam
import 
import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    "batch_size": 128,
    "num_epochs": 100,
    "dropout_rate": 0.5, 
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
params = DotDict(simpnetparams)