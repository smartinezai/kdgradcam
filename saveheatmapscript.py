from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet, ResNet18
from torchvision import datasets, transforms
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
import os
import torch
from torch.nn import init
import torch.nn as nn
import random
from torch.nn import Module
import numpy as np
import h5py
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


train_dl = fetch_dataloader('train', params)

print("initialising teacher model")
teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
resnext.to(device);
load_checkpoint(teacher_checkpoint, resnext );
print("teacher model loaded")

# resnet18_KD = ResNet18()
# resnet18_KD_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
# resnet18_KD.to(device)
# load_checkpoint(resnet18_KD_checkpoint, resnet18_KD);

# resnet18_vanilla = ResNet18()
# resnet18_vanilla_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnet18/best.pth.tar'
# resnet18_vanilla.to(device)
# load_checkpoint(resnet18_vanilla_checkpoint, resnet18_vanilla);

# simpnet_KD = Net(params).cuda() if params.cuda else Net(params)
# simpnet_KD_checkpoint = '/home/smartinez/thesisfolder/modelruns/kd_nothresh_student_simplenet_teacher_resnext29_experiment_NoHeatmap_KLDgamma_100_best.pth.tar' 
# simpnet_KD.to(device)
# load_checkpoint(simpnet_KD_checkpoint, simpnet_KD);

# simpnet_vanilla = Net(params).cuda() if params.cuda else Net(params)
# simpnet_vanilla_checkpoint = '/home/smartinez/thesisfolder/modelruns/simpnet_noKD_best.pth.tar' 
# simpnet_vanilla.to(device)
# load_checkpoint(simpnet_vanilla_checkpoint, simpnet_vanilla);

#print("loaded student weights")
batch_count = len(train_dl)



# resnet18_vanilla_target_layers =[resnet18_vanilla.conv1, #initial conv
#                                 #layer1
#                                 #first basis block
#                                 resnet18_vanilla.layer1[0].conv1,
#                                 resnet18_vanilla.layer1[0].conv2,
#                                 #second basis block
#                                 resnet18_vanilla.layer1[1].conv1,
#                                 resnet18_vanilla.layer1[1].conv2,
#                                 #layer2
#                                 #first basis block                       
#                                 resnet18_vanilla.layer2[0].conv1,
#                                 resnet18_vanilla.layer2[0].conv2,
#                                 #second basis block
#                                 resnet18_vanilla.layer2[1].conv1,
#                                 resnet18_vanilla.layer2[1].conv2,
#                                 #layer3
#                                 #first basis block
#                                 resnet18_vanilla.layer3[0].conv1,
#                                 resnet18_vanilla.layer3[0].conv2,
#                                 #second basis block
#                                 resnet18_vanilla.layer3[1].conv1,
#                                 resnet18_vanilla.layer3[1].conv2,
#                                 #layer4
#                                 #first basis block
#                                 resnet18_vanilla.layer4[0].conv1,
#                                 resnet18_vanilla.layer4[0].conv2,
#                                 #second basis block
#                                 resnet18_vanilla.layer4[1].conv1,
#                                 resnet18_vanilla.layer4[1].conv2
#                                 ]
# resnet18_KD_target_layers =[resnet18_KD.conv1, #initial conv
#                                 #layer1
#                                 #first basis block
#                                 resnet18_KD.layer1[0].conv1,
#                                 resnet18_KD.layer1[0].conv2,
#                                 #second basis block
#                                 resnet18_KD.layer1[1].conv1,
#                                 resnet18_KD.layer1[1].conv2,
#                                 #layer2
#                                 #first basis block                       
#                                 resnet18_KD.layer2[0].conv1,
#                                 resnet18_KD.layer2[0].conv2,
#                                 #second basis block
#                                 resnet18_KD.layer2[1].conv1,
#                                 resnet18_KD.layer2[1].conv2,
#                                 #layer3
#                                 #first basis block
#                                 resnet18_KD.layer3[0].conv1,
#                                 resnet18_KD.layer3[0].conv2,
#                                 #second basis block
#                                 resnet18_KD.layer3[1].conv1,
#                                 resnet18_KD.layer3[1].conv2,
#                                 #layer4
#                                 #first basis block
#                                 resnet18_KD.layer4[0].conv1,
#                                 resnet18_KD.layer4[0].conv2,
#                                 #second basis block
#                                 resnet18_KD.layer4[1].conv1,
#                                 resnet18_KD.layer4[1].conv2
#                                 ]
# resnet18_layer_strings=["conv1", 
#                                 #layer1
#                                 #first basis block
#                                 "layer1.block1.conv1",
#                                 "layer1.block1.conv2",
    
#                                 "layer1.block2.conv1",
#                                 "layer1.block2.conv2",
                                
#                                 "layer2.block1.conv1",
#                                 "layer2.block1.conv2",
    
#                                 "layer2.block2.conv1",
#                                 "layer2.block2.conv2",
#                                 #layer3
#                                 "layer3.block1.conv1",
#                                 "layer3.block1.conv2",
    
#                                 "layer3.block2.conv1",
#                                 "layer3.block2.conv2",
#                                 #layer4
#                                 "layer4.block1.conv1",
#                                 "layer4.block1.conv2",
    
#                                 "layer4.block2.conv1",
#                                 "layer4.block2.conv2"
#                                 ]

# simpnet_KD_target_layers = [simpnet_KD.conv1, simpnet_KD.conv2, simpnet_KD.conv3]
# simpnet_vanilla_target_layers = [simpnet_vanilla.conv1, simpnet_vanilla.conv2, simpnet_vanilla.conv3]
# simpnet_layer_strings= ["conv1", "conv2", "conv3"]
resnext_target_layers = [ 
                             resnext.module.stage_1.stage_1_bottleneck_0,
                             resnext.module.stage_1.stage_1_bottleneck_1,
                             resnext.module.stage_1.stage_1_bottleneck_2,
              
                            resnext.module.stage_2.stage_2_bottleneck_0,
                            resnext.module.stage_2.stage_2_bottleneck_1,
                            resnext.module.stage_2.stage_2_bottleneck_2,
                            
                            resnext.module.stage_3.stage_3_bottleneck_0,
                            resnext.module.stage_3.stage_3_bottleneck_1,
                            resnext.module.stage_3.stage_3_bottleneck_2]
teacher_layer_strings= [
                              "stg1 BNK1",
                             "stg1 BNK2",
                             "stg1 BNK3",
              
                            "stg2 BNK1",
                             "stg2 BNK2",
                             "stg2 BNK3",
                            
                              "stg3 BNK1",
                             "stg3 BNK2",
                             "stg3 BNK3"]

resnext_cam = GradCAM(model=resnext, target_layers=resnext_target_layers)
# simpnet_KD_cam = GradCAM(model=simpnet_KD, target_layers=simpnet_KD_target_layers)
# simpnet_vanilla_cam = GradCAM(model=simpnet_vanilla, target_layers=simpnet_vanilla_target_layers)
# resnet18_KD_cam = GradCAM(model=resnet18_KD, target_layers=resnet18_KD_target_layers)
# resnet18_vanilla_cam = GradCAM(model=resnet18_vanilla, target_layers=resnet18_vanilla_target_layers)


file_path = "/home/smartinez/thesisfolder/train_dl_cams_resnext_teach_only.h5"
with h5py.File(file_path, 'w') as hf:
    concatenated_labels = None
    concatenated_resnext_cams = None
    # concatenated_resnet18_KD_cams = None
    # concatenated_resnet18_vanilla_cams = None
    # concatenated_simpnet_KD_cams = None
    # concatenated_simpnet_vanilla_cams = None
    with tqdm(total=batch_count) as t:
        for batch_index, (data_batch, labels_batch) in enumerate(train_dl):
                # move to GPU if available
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
                # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                # compute model output
                #with torch.no_grad():
            resnext_output_batch = resnext(data_batch)
            
            teacher_preds = torch.sigmoid(resnext_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
                    
            # simpnet_KD_heatmap_batch = simpnet_KD_cam(data_batch,max_indices)
            # simpnet_vanilla_heatmap_batch = simpnet_vanilla_cam(data_batch,max_indices)
            # resnet18_KD_heatmap_batch = resnet18_KD_cam(data_batch,max_indices) # print("student_heatmap_batch lenght (it's a list  # print(len(student_heatmap_batch))
            # resnet18_vanilla_heatmap_batch = resnet18_vanilla_cam(data_batch,max_indices) # print("student_heatmap_batch lenght (it's a list  # print(len(student_heatmap_batch))
            resnext_heatmap_batch = resnext_cam(data_batch)
            # simpnet_KD_heatmap_stack = np.array(simpnet_KD_heatmap_batch).squeeze()
            # simpnet_vanilla_heatmap_stack = np.array(simpnet_vanilla_heatmap_batch).squeeze()
            # resnet18_KD_heatmap_stack = np.array(resnet18_KD_heatmap_batch).squeeze()
            # resnet18_vanilla_heatmap_stack = np.array(resnet18_vanilla_heatmap_batch).squeeze()
            resnext_heatmap_stack = np.array(resnext_heatmap_batch).squeeze()
            # simpnet_KD_heatmap_stack = simpnet_KD_heatmap_stack.swapaxes(0,1)
            # simpnet_vanilla_heatmap_stack = simpnet_vanilla_heatmap_stack.swapaxes(0,1)
            # resnet18_KD_heatmap_stack = resnet18_KD_heatmap_stack.swapaxes(0,1)
            # resnet18_vanilla_heatmap_stack = resnet18_vanilla_heatmap_stack.swapaxes(0,1)
            resnext_heatmap_stack = resnext_heatmap_stack.swapaxes(0,1)
            #print(f"simpnet KD heatmap stack shape after axis swap: {simpnet_KD_heatmap_stack.shape}")
            #print(f"resnet18 KD heatmap stack shape after axis swap: {resnet18_KD_heatmap_stack.shape}")
            #print(f"resnext heatmap stack shape after axis swap: {resnext_heatmap_stack.shape}")
            #print(f"shape of teacher indices batch : {max_indices.shape}")
            
            if concatenated_labels is None:
                concatenated_labels = max_indices.cpu().detach().numpy()
            else:
                concatenated_labels = np.concatenate((concatenated_labels, max_indices.cpu().detach().numpy()), axis=0)



            if concatenated_resnext_cams is None:
                concatenated_resnext_cams = resnext_heatmap_stack#.cpu().detach().numpy()
            else:
                concatenated_resnext_cams = np.concatenate((concatenated_resnext_cams, resnext_heatmap_stack), axis=0)

            
            # if concatenated_resnet18_KD_cams is None:
            #     concatenated_resnet18_KD_cams = resnet18_KD_heatmap_stack
            # else:
            #     concatenated_resnet18_KD_cams = np.concatenate((concatenated_resnet18_KD_cams, resnet18_KD_heatmap_stack), axis=0)

            
            
            # if concatenated_resnet18_vanilla_cams is None:
            #     concatenated_resnet18_vanilla_cams = resnet18_vanilla_heatmap_stack
            # else:
            #     concatenated_resnet18_vanilla_cams = np.concatenate((concatenated_resnet18_vanilla_cams, resnet18_vanilla_heatmap_stack), axis=0)


            # if concatenated_simpnet_KD_cams is None:
            #     concatenated_simpnet_KD_cams = simpnet_KD_heatmap_stack
            # else:
            #     concatenated_simpnet_KD_cams = np.concatenate((concatenated_simpnet_KD_cams, simpnet_KD_heatmap_stack), axis=0)

            # if concatenated_simpnet_vanilla_cams is None:
            #     concatenated_simpnet_vanilla_cams = simpnet_vanilla_heatmap_stack
            # else:
            #     concatenated_simpnet_vanilla_cams = np.concatenate((concatenated_simpnet_vanilla_cams, simpnet_vanilla_heatmap_stack), axis=0)

            t.update() 
 
            #  print(student_heatmap_stack.shape)
            # print("teacher heatmap batch shape")
            #print(teacher_heatmap_stack.shape)  
    print(f"shape of concatenated labels: {concatenated_labels.shape}")
    # print(f"shape of concatenated simpnet KD cams: {concatenated_simpnet_KD_cams.shape}")
    # print(f"shape of concatenated simpnet vanilla cams: {concatenated_simpnet_vanilla_cams.shape}")
    # print(f"shape of concatenated resnet18 KD cams: {concatenated_resnet18_KD_cams.shape}")
    # print(f"shape of concatenated resnet18 vanilla cams : {concatenated_resnet18_vanilla_cams.shape}")
    print(f"shape of concatenated rexnext cams : {concatenated_resnext_cams.shape}")        
    hf.create_dataset('resnextcams', data=concatenated_resnext_cams)
    hf.create_dataset('resnext_topclass_labels', data=concatenated_labels)
    # hf.create_dataset('resnet18_KD_cams', data=concatenated_resnet18_KD_cams)
    # hf.create_dataset('resnet18_vanilla_cams', data=concatenated_resnet18_vanilla_cams)
    # hf.create_dataset('simpnet_KD_cams', data=concatenated_simpnet_KD_cams)
    # hf.create_dataset('simpnet_vanilla_cams', data=concatenated_simpnet_vanilla_cams)
    print("DONE WITH THE WHOLE THING")
        


        

    """# with torch.no_grad():
                #    student_heatmap_tensor = torch.tensor(student_heatmap_stack)
                #   teacher_heatmap_tensor = torch.tensor(teacher_heatmap_stack)
                #  for student_layer_index, student_layer in enumerate(student_heatmap_tensor):
                #     for teacher_layer_index, teacher_layer in enumerate(teacher_heatmap_tensor):
                    
                            
                
                #output_batch = student_output_batch
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                #output_batch = output_batch.data.cpu().numpy()
                #labels_batch = labels_batch.data.cpu().numpy()
    """
            

    """
            #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
            #            "train_avg_combinedLoss":combinedLoss_avg() }         )
                t.set_postfix({'val_avg_kl_loss': '{:05.3f}'.format(float(kl_loss_avg()))})
                    
        """       
    





