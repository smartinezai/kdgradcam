from typing import List
import wandb
from PIL import Image
import torch
from torch.nn import init
import torch.nn as nn
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import matplotlib as mpl
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
import os
import matplotlib.colors as mcolors
import random
import numpy as np
from torchvision.transforms import v2, ToPILImage
import json
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import cv2
import shutil
from typing import Any, Callable, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.cifar import CIFAR10
from thesisfolder.knowledge_distillation_pytorch.utils import RunningAverage
from thesisfolder.knowledge_distillation_pytorch.model.net import Net
from thesisfolder.knowledge_distillation_pytorch.model.resnet import ResNet18
from thesisfolder.knowledge_distillation_pytorch.model.resnext import CifarResNeXt
from thesisfolder.pytorch_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from thesisfolder.pytorch_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image
import pickle
#from knowledge_distillation_pytorch import utils


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
   # print(type(teacher_outputs))
    #print(type(outputs))
    KD_loss = nn.KLDivLoss(reduction = "batchmean")(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss



def apply_variable_gaussian_blur(image, heatmap):
    """
    Applies a variable Gaussian blur to an image based on a heatmap.
    
    Parameters:
    - image: A numpy array of shape (32, 32, 3) representing the RGB image.
    - heatmap: A numpy array of shape (32, 32) with values ranging from 0 to 255,
               representing the intensity of blur to apply at each pixel.
               
    Returns:
    - blurred_image: A numpy array of the same shape as `image` after applying the variable blur.
    """
    # Ensure the heatmap values are in the expected range
    heatmap = np.clip(heatmap, 0, 1)
    # Convert heatmap values to kernel sizes (odd numbers from 1 to a max size, e.g., 31).
    # The mapping from heatmap value to kernel size can be adjusted based on desired effect.
    kernel_sizes = np.interp(heatmap, [0, 1], [1, 31]).astype(int)
    # Make sure kernel sizes are odd to comply with cv2 Gaussian blur requirements.
    kernel_sizes = (kernel_sizes // 2) * 2 + 1
    
    blurred_image = np.zeros_like(image)
    # Apply a Gaussian blur based on the kernel size derived from the heatmap value.
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            kernel_size = kernel_sizes[i, j]
            #print(f"kernel size: {kernel_size}")
            #print(f"shape of kernel_sizeZ: {kernel_sizes.shape}")
            if kernel_size > 1:
                # Define the region to apply blur (could overlap, hence take min/max into account)
                x_start, x_end = max(0, i - kernel_size//2), min(32, i + kernel_size//2 + 1)
                y_start, y_end = max(0, j - kernel_size//2), min(32, j + kernel_size//2 + 1)
                # Apply blur to the local region
                blurred_image[x_start:x_end, y_start:y_end] = cv2.GaussianBlur(
                    image[x_start:x_end, y_start:y_end], (kernel_size, kernel_size), 0)
            else:  # If kernel_size is 1, no blur is applied
                blurred_image[i, j] = image[i, j]
    
    return blurred_image

def check_nan(array):
   if isinstance(array, np.ndarray):   
    if np.isnan(array).any():
      print("Error, NaN detected")
      return True
    else:
      return False

   elif isinstance(array, torch.Tensor):
    if torch.isnan(array).any():
      print("Error, NaN detected")
      return True
    else:
      return False
   else:
   # print("array: "+str(array))
  #  print("array type: "+str(array.dtype))
    print("array is most likely a list and not a numpy array or a pytorch tensor, ask for help on how to fix this")
    raise ValueError("Unsupported data type. Only numpy arrays and PyTorch tensors are supported.")
     
    
def MSEloss(inputs, gt):
    c = nn.MSELoss()
    loss = c(inputs,gt)
    return loss


def newblurmask(image, heatmap, kernel=5, sigma=3):
    
  if isinstance(image,torch.Tensor) or isinstance(image, Image.Image):
    if isinstance(image, torch.Tensor):                                                 #if image.shape[0] == 3:
        converted_image = np.transpose(image.numpy(), (1, 2, 0))
       # print(f"converting from tensor") 
    else:
      #print(f"converting from pil")
      converted_image = np.asarray(image)
          #  print(f"image shape {converted_image.shape}")
            #print(f"image {image}")
          #  print(f"heatmap {heatmap}")
  else:
    converted_image = image 
  #print(f"image type {converted_image.dtype}, image shape {converted_image.shape}")
  blurred_image = cv2.GaussianBlur(converted_image, (kernel, kernel),sigmaX=sigma) # Blur it as intense as you would like to have your maxmum intensity
  #print(f"blurred image shape {blurred_image.shape}")
  blended_image = np.zeros_like(converted_image) # We want an zero-array in the shape of the original image

  blend_ratio = np.repeat(heatmap[:,:,None], 3, axis=-1)
  #print(f"blend_ratio matrix shape {blend_ratio.shape}")
  if blend_ratio.shape != converted_image.shape:
   # print("transposing blend_ratio matrix")
    blend_ratio = np.transpose(blend_ratio,(2,0,1))
  blended_image = (blend_ratio * converted_image + (1-blend_ratio)*blurred_image)
  return blended_image       

def tensorblurmask(image, heatmap, kernel=5, sigma=3):
    #image received from blucrifar10 is 3,32,32 uint8
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
   # print(f"image shape {image.shape}")    
    blur = v2.GaussianBlur(kernel_size=(kernel, kernel), sigma=sigma).to(device) #for this to work we need shape, C, H, W
    blurred_image = blur(image).to(device)
    blended_image = torch.zeros_like(image).to(device) # We want an zero-array in the shape of the original image
    tensor_heatmap = torch.from_numpy(heatmap).to(device)
    #print("tensor heatmap shape "+str(tensor_heatmap.shape))
    tensor_heatmap = tensor_heatmap.unsqueeze(0)#.to(device)
    #print(f"tensor heatmap shape after unsqueezing {tensor_heatmap.shape}")
    tensor_heatmap = tensor_heatmap.expand(3, -1, -1)#.to(device)
    #print(f"tensor heatmap shape after expansion {tensor_heatmap.shape}")
    if tensor_heatmap.shape != blurred_image.shape:
        
        print("tensor heatmap and blurred image tensor are not of the same shape")
    #blend_ratio = np.transpose(blend_ratio,(2,0,1)).
    # print(f"image shap[e] {image.shape}")
    # print(f"image dtype {image.dtype}")
    # print(f"max value of image {image.max()}")
    
    # print(f"blurimgshape {blurred_image.shape}")
    # print(f"blurimg dtype {blurred_image.dtype}")
    # print(f"max value of blurimg {blurred_image.max()}")
    
    blended_image = (tensor_heatmap * image + (1-tensor_heatmap)*blurred_image)
    # print(f"blendimgshape {blended_image.shape}")
    # print(f"blendimg dtype {blended_image.dtype}")
    # print(f"max value of blendimg {blended_image.max()}")
    blended_image = blended_image.to(torch.uint8)
   # print(f"datatype after int cast that I am sure works {blended_image.dtype}")
    #if blended_image.shape[2] == 3:
    #    blended_image = blended_image.permute(2,0,1)
      #  print("shape after permuting +str(blended_image.shape)")
    return blended_image       

def oldblurmask(image, heatmap, kernel=5, sigma=3):
  if isinstance(image,torch.Tensor) or isinstance(image, Image.Image):
    if isinstance(image, torch.Tensor):                                                 #if image.shape[0] == 3:
        converted_image = np.transpose(image.numpy(), (1, 2, 0))
        #print(f"converting from tensor") 
    else:
           # print(f"converting from pil")
      converted_image = np.asarray(image)
          #  print(f"image shape {converted_image.shape}")
            #print(f"image {image}")
          #  print(f"heatmap {heatmap}")
   # print(f"image type {image.dtype}")
  else:
    converted_image = image 
  blurred_image = cv2.GaussianBlur(image, (kernel, kernel),sigmaX=sigma) # Blur it as intense as you would like to have your maxmum intensity

  blended_image = np.zeros_like(image) # We want an zero-array in the shape of the original image

  #Now we pblend in the blurred image with the original image based on the heatmap values
#  for i in range(image.shape[0]):
  #     for j in range(image.shape[1]):
  #        blend_ratio = heatmap[i,j] # This gives me the intensity value of the blurring between 0 and 1
    #       blended_image[i,j] = image[i,j] * blend_ratio + blurred_image[i,j] * 1- blend_ratio
  blend_ratio = np.repeat(heatmap[:,:,None], 3, axis=-1)
  blended_image = (blend_ratio * image + (1-blend_ratio)*blurred_image)
  return blended_image        

def blurmask(image, heatmap, kernel=5, sigma=3):
    if isinstance(image,torch.Tensor) or isinstance(image, Image.Image):
        if isinstance(image, torch.Tensor):                                                 #if image.shape[0] == 3:
            converted_image = np.transpose(image.numpy(), (1, 2, 0))
            print(f"converting from tensor") 
        else:
           # print(f"converting from pil")
            converted_image = np.asarray(image)
          #  print(f"image shape {converted_image.shape}")
            #print(f"image {image}")
          #  print(f"heatmap {heatmap}")
   # print(f"image type {image.dtype}")
    else:
        converted_image = image 
     #   print(f"image shape {converted_image.shape}")
    blurred_image = cv2.GaussianBlur(converted_image/255, (kernel, kernel),sigmaX=sigma) # Blur it as intense as you would like to have your maxmum intensity
    #print(f"blurred image shape{blurred_image.shape}")
    blended_image = np.zeros_like(converted_image) # We want an zero-array in the shape of the original image
    #print(f"blended image shape {blended_image.shape}")
    blend_ratio = np.repeat(heatmap[:,:,None], 3, axis=-1)
    blended_image = (blend_ratio * converted_image + (1-blend_ratio)*blurred_image)
    #blended_image = np.transpose(blended_image, (2,0,1))
  #  print(f"blended image shape {blended_image.shape}")
    #print(f"blended image type {blended_image.dtype}")
    #print(f"blended image shape {blended_image.shape}")
    blended_image = Image.fromarray((blended_image * 255).astype(np.uint8))
    
    return blended_image      

        

def loss_fn_kd_nothresh(outputs, labels, teacher_outputs, params, student_heatmap_batch, teacher_heatmap_batch

               ):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    if not isinstance(student_heatmap_batch, torch.Tensor):
        print("DANGER: student heatmap batch is NOT of type Tensor ")
        #print(student_heatmap_batch.dtype)
    if not isinstance(teacher_heatmap_batch, torch.Tensor):
        print("DANGER: teacher heatmap batch is NOT of type Tensor ")  
        #print(teacher_heatmap_batch.dtype)
    alpha = params.alpha
    T = params.temperature
    #print("outputs shape "+str(outputs.shape))
    #print("teacher_outputs shape "+str(teacher_outputs.shape))
    #print("labels shape "+str(labels.shape))
    KD_loss = nn.KLDivLoss(reduction = "batchmean")(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    heatmap_MSE = MSEloss(student_heatmap_batch, teacher_heatmap_batch)
  
    return KD_loss, heatmap_MSE


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar_kw = {'shrink': 0.6} 
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #cbar.set_shrink(0.7)

    # Show all ticks and label them with the respective list entries.
   # ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    #ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def percentchangeheatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Set the color map range
    kwargs.setdefault('vmin', -100)
    kwargs.setdefault('vmax', 100)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar_kw = {'shrink': 0.5}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("white", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=9,**kw)
            texts.append(text)

    return texts


def train(model, optimizer, loss_fn, dataloader, metrics, params, wandbrun):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    if params.cam_layer == -1:
        print("NO LAYER FOR BLURRING WAS SELECTED")

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch, = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch= Variable(train_batch), Variable(labels_batch)
            
            
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            #print("check before update")
            t.update()
            
    if wandbrun == True:
        wandb.log({"train_Loss": float(loss_avg())})

    # compute mean of all metrics in summary
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    

def blurtrain(model, optimizer, loss_fn, dataloader, metrics, params, wandbrun):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    if params.cam_layer == -1:
        print("NO LAYER FOR BLURRING WAS SELECTED")

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, originals_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch, = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch= Variable(train_batch), Variable(labels_batch)
            
            
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            #print("check before update")
            t.update()
            
    if wandbrun == True:
        wandb.log({"train_Loss": float(loss_avg())})

    # compute mean of all metrics in summary
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)    

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                  #     loss_fn, metrics, params, model_dir, model_arch, restore_file=None, wandbrun = False):
                  loss_fn, metrics, params, model_arch, restore_file=None, wandbrun = False):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    model_dir = '/home/smartinez/thesisfolder/modelruns'
    # reload weights from restore_file if specified
    if restore_file is not None:
        
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version in ["resnet18_distill" , "resnet18_KD" , "resnet18_vanilla" , "base_resnet18"]:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        model_arch = "resnet18_noKD"
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.2)
        model_arch = "simpnet_noKD"
    
    for epoch in range(params.num_epochs):
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params,wandbrun)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        
        
        
        val_loss = val_metrics['loss']
           

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        if wandbrun == True:

            wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
            wandb.log({"val_loss": val_loss, "epoch": epoch+1})    
            
            
            
         
        

        
        path_base = "/home/smartinez/thesisfolder/modelruns"
        if params.cam_layer == -1:
            destination_folder = f'{path_base}/vanilla_{model_arch}_noblur' 
    
        
        else:
            destination_folder = f'{path_base}/vanilla_{model_arch}_camlayer_{params.cam_layer}' 
    
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(destination_folder, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(destination_folder, "metrics_val_last_weights.json")    
        #save_dict_to_json(val_metrics, last_json_path)
        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=destination_folder)
        
        
        

def blurtrain_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                  #     loss_fn, metrics, params, model_dir, model_arch, restore_file=None, wandbrun = False):
                  loss_fn, metrics, params, model_arch, restore_file=None, wandbrun = False):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    model_dir = '/home/smartinez/thesisfolder/modelruns'
    # reload weights from restore_file if specified
    if restore_file is not None:
        
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version in ["resnet18_distill" , "resnet18_KD" , "resnet18_vanilla" , "base_resnet18"]:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        model_arch = "resnet18_noKD"
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
        model_arch = "simpnet_noKD"
    
    for epoch in range(params.num_epochs):
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        blurtrain(model, optimizer, loss_fn, train_dataloader, metrics, params,wandbrun)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        
        
        
        val_loss = val_metrics['loss']
           

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        if wandbrun == True:

            wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
            wandb.log({"val_loss": val_loss, "epoch": epoch+1})    
            
            
            
         
        

        
        path_base = "/home/smartinez/thesisfolder/modelruns"
        if params.cam_layer == -1:
            destination_folder = f'{path_base}/vanilla_{model_arch}_noblur' 
    
        
        else:
            destination_folder = f'{path_base}/vanilla_{model_arch}_camlayer_{params.cam_layer}' 
    
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(destination_folder, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(destination_folder, "metrics_val_last_weights.json")    
        #save_dict_to_json(val_metrics, last_json_path)
        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=destination_folder)
        



def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    m = torch.nn.Sigmoid()
   # running_loss = 0


    # compute metrics over the dataset
    with torch.no_grad():

      for data_batch, labels_batch in tqdm(dataloader):

          # move to GPU if available
          if params.cuda:
              data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
          # fetch the next evaluation batch
          data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

          # compute model output
          output_batch = model(data_batch)
          loss = loss_fn(output_batch, labels_batch)


          #running_loss += loss.item()


          # extract data from torch Variable, move to cpu, convert to numpy arrays
          output_batch = output_batch.data.cpu().numpy()
          labels_batch = labels_batch.data.cpu().numpy()

          # compute all metrics on this batch
          summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                          for metric in metrics}
          #summary_batch['loss'] = loss.data[0]
          summary_batch['loss']=loss

          summ.append(summary_batch)

         # del data_batch, labels_batch, output_batch
         # gc.collect()
          #torch.cuda.empty_cache()


      # compute mean of all metrics in summary

    #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]}
    #metrics_mean = {
    #metric: torch.mean(torch.from_numpy(np.array([x[metric] for x in summ])))
    #for metric in summ[0]
    #    }
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)



    num_samples = float(len(dataloader.dataset))
    #avg_test_loss = running_loss/num_samples

    #print('test_loss (Cross entropy loss): {:.4f}, test_avg_precision(not the same metric as accuracy):{:.3f}'.format(
     #                 avg_test_loss, test_map))
    #print('test_loss (Cross entropy loss): {:.4f}'.format(            avg_test_loss))

    #print("num_samples: "+str(num_samples))
    #print("summary length: "+str(len(summ)))
    #print("loss without averaging: "+str(running_loss))
    #print("precision without averaging: "+str(running_ap))
    #print("metrics_mean: "+str(metrics_mean))



    return metrics_mean


def train_kd_noblur(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):


    # set model to training mode
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    heatmap_dissimilarity_avg = RunningAverage()

   # running_loss = 0.0
   # running_ap = 0.0
    m = torch.nn.Sigmoid()

    tr_loss, tr_map = [], []
    with tqdm(total=len(dataloader)) as t:
      for i, (train_batch, labels_batch) in enumerate(dataloader):
        #  print("size of train batch "+str(train_batch.shape))
          # move to GPU if available
          if params.cuda:
              train_batch, labels_batch = train_batch.cuda(), \
                                          labels_batch.cuda()
          # convert to torch Variables
          # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
          #print("sie of train batch "+str(train_batch.shape))
          # compute model output, fetch teacher output, and compute KD loss
          #output_batch = model(train_batch)

          # get one batch output from teacher_outputs list
          #student_output_batch = model(train_batch)


          train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
          with torch.no_grad():
            teacher_output_batch = teacher_model(train_batch)
            #print("teacher output batch type after using no grad without going into the cuda part "+str(type(teacher_output_batch)))
            if params.cuda:
              teacher_output_batch = teacher_output_batch.cuda()
              # print("teacher output batch type after using no grad AFTER going into the cuda part "+str(type(teacher_output_batch)))


          student_output_batch = model(train_batch)


          loss = loss_fn_kd(outputs=student_output_batch, labels=labels_batch, teacher_outputs=teacher_output_batch, params=params)

          # clear previous gradients, compute gradients of all variables wrt loss
          #print(type(loss))
          optimizer.zero_grad()
          loss.backward()

          # performs updates using calculated gradients
          optimizer.step()

          # Evaluate summaries only once in a while
          if i % params.save_summary_steps == 0:
              # extract data from torch Variable, move to cpu, convert to numpy arrays
              student_output_batch = student_output_batch.data.cpu().numpy()
              labels_batch = labels_batch.data.cpu().numpy()

              # compute all metrics on this batch
              summary_batch = {metric:metrics[metric](student_output_batch, labels_batch)
                                for metric in metrics}
              summary_batch['kl_loss'] = loss.data#it was loss.data[0]
              summ.append(summary_batch)

          # update the average loss
          loss_avg.update(loss.data)#it was loss.data[0]


          t.set_postfix(train_avg_loss='{:05.3f}'.format(loss_avg()))
          t.update()

    #      del train_batch, labels_batch, student_output_batch
     #     gc.collect()
      #    torch.cuda.empty_cache()



    # compute mean of all metrics in summary
   # tb.add_scalar('KLD average loss', loss, epoch)
    #tb.add_scalar('Heatmap loss', heatmap_dissimilarity, epoch)
    #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]}
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    #num_samples = float(len(train_loader.dataset))
    #tr_loss_ = running_loss/num_samples
    #tr_map_ = running_ap/num_samples
    return loss_avg()




def blurtrain_kd(model, teacher_model, optimizer, loss_fn_kd_nothresh, dataloader, metrics,
                      params, wandbrun):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """
    # set model to training mode
    model.train()
    teacher_model.eval()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    

    student_model = model

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i, (train_batch, originals_batch, labels_batch) in enumerate(dataloader):
        #  print("size of train batch "+str(train_batch.shape))
          # move to GPU if available
        if params.cuda:
            train_batch, originals_batch, labels_batch = train_batch.cuda(), originals_batch.cuda(), labels_batch.cuda()
          # convert to torch Variables
            train_batch, originals_batch, labels_batch = Variable(train_batch), Variable(originals_batch), Variable(labels_batch)
          #print("sie of train batch "+str(train_batch.shape))
          # compute model output, fetch teacher output, and compute KD loss
          #output_batch = model(train_batch)

          # get one batch output from teacher_outputs list
          #student_output_batch = model(train_batch)
          
        teacher_output_batch = teacher_model(originals_batch)
             
            
        if params.cuda:
            teacher_output_batch = teacher_output_batch.cuda()

        else:
            train_batch, originals_batch, labels_batch = Variable(train_batch), Variable(originals_batch), Variable(labels_batch)
            with torch.no_grad():
              teacher_output_batch = teacher_model(originals_batch)
              #print("teacher output batch type after using no grad without going into the cuda part "+str(type(teacher_output_batch)))
              if params.cuda:
                teacher_output_batch = teacher_output_batch.cuda()
               # print("teacher output batch type after using no grad AFTER going into the cuda part "+str(type(teacher_output_batch)))


        student_output_batch = model(train_batch)


       
        loss = loss_fn_kd(outputs=student_output_batch, labels=labels_batch, teacher_outputs=teacher_output_batch, params=params)
      
          
        optimizer.zero_grad()
        loss.backward()
      
        optimizer.step()
       
          # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
              # extract data from torch Variable, move to cpu, convert to numpy arrays
            student_output_batch = student_output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

              # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](student_output_batch, labels_batch)
                                for metric in metrics}
            summary_batch['loss'] = loss.data
            summ.append(summary_batch)
              #wandb.log({"train_batch_kl_loss":kl_loss.data,"train_batch_heatmap_dissimilarity":heatmap_dissimilarity,
               #          "train_batch_combinedLoss":combinedLossTensor.data,} )

          # update the average loss
        loss_avg.update(loss.data)#it was loss.data[0]
         
        #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
        t.set_postfix({'train_average_Loss': '{:05.3f}'.format(float(loss_avg()))})
        
        t.update()
  
      #    torch.cuda.empty_cache()


    if wandbrun == True:
        wandb.log({"train_avg_Loss": float(loss_avg())})
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
  

def train_kd_nothresh(model, teacher_model, optimizer, loss_fn_kd_nothresh, dataloader, metrics,
                      params, experiment, student_activated_features, teacher_activated_features, KLDgamma, wandbrun):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """
    #print("inside train_kd_nothresh, experiment: "+str(experiment))
    # set model to training mode
    model.train()
    teacher_model.eval()
    #print("inside train kd no thresh")
    # summary for current training loop and a running average object for loss
    summ = []
    combinedLoss_avg = RunningAverage()
    heatmap_dissimilarity_avg = RunningAverage()
    kl_loss_avg = RunningAverage()
    

    student_model = model

    student_weight_softmax_params =list(student_model.linear.parameters()) # This gives a list of weights for the fully connected layers
    student_weight_softmax = student_weight_softmax_params[0].data
    student_weight_softmax.requires_grad = True
    teacher_weight_softmax_params = list(teacher_model.module.classifier.parameters()) # This gives a list of weights for the fully connected layers
    teacher_weight_softmax = teacher_weight_softmax_params[0].data
    teacher_weight_softmax.requires_grad = True
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i, (train_batch, labels_batch) in enumerate(dataloader):
        #  print("size of train batch "+str(train_batch.shape))
          # move to GPU if available
          if params.cuda:
              train_batch, labels_batch = train_batch.cuda(), \
                                          labels_batch.cuda()
          # convert to torch Variables
          # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
          #print("sie of train batch "+str(train_batch.shape))
          # compute model output, fetch teacher output, and compute KD loss
          #output_batch = model(train_batch)

          # get one batch output from teacher_outputs list
          #student_output_batch = model(train_batch)
          if experiment != 'NoHeatmap':

            train_batch, labels_batch = Variable(train_batch, requires_grad= True), Variable(labels_batch)
            teacher_output_batch = teacher_model(train_batch)
             
            
            if params.cuda:
              teacher_output_batch = teacher_output_batch.cuda()

          else:
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            with torch.no_grad():
              teacher_output_batch = teacher_model(train_batch)
              #print("teacher output batch type after using no grad without going into the cuda part "+str(type(teacher_output_batch)))
              if params.cuda:
                teacher_output_batch = teacher_output_batch.cuda()
               # print("teacher output batch type after using no grad AFTER going into the cuda part "+str(type(teacher_output_batch)))


          student_output_batch = model(train_batch)


          if experiment =='NoHeatmap':
                loss = loss_fn_kd_noHeatMap(outputs=student_output_batch, labels=labels_batch, teacher_outputs=teacher_output_batch, params=params)

          if experiment == 'TrueClass':
            #print("using true classes")
            if (print_if_all_zero(student_activated_features.features)):
              print("student_activated_features.features is all 0s")
             # print("student_activated_features.features: "+str(student_activated_features.features))

            print("student activated features")
            print(student_activated_features.features.grad_fn)
            student_heatmap_batch = getCAMBatch(student_activated_features.features, student_weight_softmax, labels_batch)

            teacher_heatmap_batch = getCAMBatch(teacher_activated_features.features, teacher_weight_softmax, labels_batch)
           # print("one line before kl loss in train kd no thresh")
            kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)

          if experiment == 'TopClass':
           # print("using top class of teacher")
            teacher_preds = torch.sigmoid(teacher_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
            print("student activated features")
            print(student_activated_features.features.grad_fn)
            student_heatmap_batch = getCAMBatch(student_activated_features.features, student_weight_softmax, max_indices)
            if (check_nan(student_heatmap_batch)):
              print("found a NaN on the student heatmap batch")
            teacher_heatmap_batch = getCAMBatch(teacher_activated_features.features, teacher_weight_softmax, max_indices)
            #print("one line before kl loss in train kd no thresh")
            kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)


          if experiment == 'AllClasses':
            #print("using all classes")
            print("student activated features")
            print(student_activated_features.features.grad_fn)
            student_heatmap_batch = classCAMSbatch(student_activated_features.features, student_weight_softmax)
            teacher_heatmap_batch = classCAMSbatch(teacher_activated_features.features, teacher_weight_softmax)
        #    print("one line before kl loss in train kd no thresh")
            kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)
          #with torch.no_grad():
          #    output_teacher_batch = teacher_model(train_batch)
          #if params.cuda:
           #   output_teacher_batch = output_teacher_batch.cuda()
         # print("student heatmap batch type "+str(student_heatmap_batch.shape))

          #if experiment != 'NoHeatmap':

           # loss = kl_loss
        #  running_loss += loss.item()
          #running_ap += get_ap_score(torch.Tensor.cpu(labels_batch).detach().numpy(), torch.Tensor.cpu(m(output_batch)).detach().numpy())

          heatmapbeta = 1-KLDgamma
          # clear previous gradients, compute gradients of all variables wrt loss
          
          
          combinedLossTensor = KLDgamma * kl_loss + heatmapbeta * (100 * heatmap_dissimilarity)
          
          optimizer.zero_grad()
          combinedLossTensor.backward()
      #    print("type of kl loss" +str(type(kl_loss)))
       #   print("type of heatmap loss "+str(type(heatmap_dissimilarity)))  
        #  print("type of combinedLoss "+str(type(combinedLossTensor)))
         # print("combinedLossTensor "+str(combinedLossTensor))
         # print("kl loss tensor gradient "+str(kl_loss.grad))
          #print("combined loss tensor gradient "+str(combinedLossTensor.grad))
          # performs updates using calculated gradients
          
          # Get gradients before optimizer step
   
    
          
          optimizer.step()
          #gradients_after_step = [param.grad.clone() for param in student_model.parameters()]
          
      #    print("GRADIENTS AFTER STEP: are they all zero?")
       #   for parameter in student_model.parameters():
        #     if torch.all(parameter == 0): 
         #        a =1
          #   else:
           #      print("parameter found (after step) where not everything is 0")
                 #print( "gradient of that parameter")
                 #non_zero_indices = torch.nonzero(parameter)
                 #for index in non_zero_indices:
                 #   value = parameter[index[0]]
                  #  print(f"Non-zero value at index {index}: {value}")
                # print(parameter.grad)   
# Check if gradients are the same
      #    gradients_equal = all(torch.equal(grad_before, grad_after) for grad_before, grad_after in zip(gradients_before_step, gradients_after_step))
       #   if gradients_equal:
        #    print("Gradients are the same before and after optimizer step.")
        #  else:
         #   print("Gradients are different before and after optimizer step.")

          #print("GRADIENTS BEFORE STEP")
          #for parameter in student_model.parameters():
          #    print(parameter.grad)
      
          #print("GRADIENTS AFTER STEP")
          #for parameter in student_model.parameters():
           #   print(parameter.grad)
          # Evaluate summaries only once in a while
          if i % params.save_summary_steps == 0:
              # extract data from torch Variable, move to cpu, convert to numpy arrays
              student_output_batch = student_output_batch.data.cpu().numpy()
              labels_batch = labels_batch.data.cpu().numpy()

              # compute all metrics on this batch
              summary_batch = {metric:metrics[metric](student_output_batch, labels_batch)
                                for metric in metrics}
              summary_batch['combinedLoss'] = combinedLossTensor.data#it was loss.data[0]
              summary_batch['heatmap_dissimilarity'] = heatmap_dissimilarity
              summary_batch['kl_loss'] = kl_loss.data
              summ.append(summary_batch)
              #wandb.log({"train_batch_kl_loss":kl_loss.data,"train_batch_heatmap_dissimilarity":heatmap_dissimilarity,
               #          "train_batch_combinedLoss":combinedLossTensor.data,} )

          # update the average loss
          combinedLoss_avg.update(combinedLossTensor.data)#it was loss.data[0]
          if experiment != 'NoHeatmap':
          #  print("heatmap_dissimilarity "+str(heatmap_dissimilarity))

            heatmap_dissimilarity_avg.update(heatmap_dissimilarity)
            kl_loss_avg.update(kl_loss.data)
            #print("heatmap_dissimilarity average "+str(heatmap_dissimilarity_avg))
        #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
            t.set_postfix({'train_average_KL_Loss': '{:05.3f}'.format(float(kl_loss_avg())),
                           'train_avgHeatmap_dissimilarity': '{:05.3f}'.format(float(heatmap_dissimilarity_avg())),
                        'train_avg_combinedLoss': '{:05.3f}'.format(combinedLoss_avg())})
          else :
            print("you're using a non vanilla train_kd method, you should call the vanilla method instead since you're not using heatmaps")
            t.set_postfix(loss='{:05.3f}'.format(combinedLoss_avg()))
          t.update()
  
        
    #      del train_batch, labels_batch, student_output_batch
     #     gc.collect()
      #    torch.cuda.empty_cache()



    # compute mean of all metrics in summary
   # tb.add_scalar('KLD average loss', loss, epoch)
    #tb.add_scalar('Heatmap loss', heatmap_dissimilarity, epoch)
    #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]}
    if wandbrun == True:
        wandb.log({"train_avg_KL_Loss": float(kl_loss_avg())})
        wandb.log({"train_avgHeatmap_dissimilarity": float(heatmap_dissimilarity_avg())})
        wandb.log({"train_avg_combinedLoss": combinedLoss_avg()})
        wandb.log({"first layer weight grad": student_model[0].weights.grad})
        wandb.log({"first layer bias grad": student_model[0].bias.grad}) 
        wandb.log({"last layer weight grad": student_model[-1].weights.grad})
        wandb.log({"last layer bias grad": student_model[-1].bias.grad})
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
  
  
  
  #  writer = SummaryWriter("torchlogs/")
   # writer.add_graph(student_model, train_batch)
    #writer.close()
    #num_samples = float(len(train_loader.dataset))
    #tr_loss_ = running_loss/num_samples
    #tr_map_ = running_ap/num_samples
    #if experiment == 'NoHeatMap':
    #  return loss_avg()

  #  print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
    #                 tr_loss_, tr_map_))
    #return loss_avg(), heatmap_dissimilarity_avg()


                  # Append the values to global arrays
  #   tr_loss.append(tr_loss_), tr_map.append(tr_map_)

def paperloop_nothresh(model, teacher_model, optimizer, loss_fn_kd_nothresh, dataloader, metrics,
                      params, experiment, KLDgamma, wandbrun, algo):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """
    student_model = model
    
    scaler = torch.cuda.amp.GradScaler() 
    #print("inside train_kd_nothresh, experiment: "+str(experiment))
    # set model to training mode
    student_model.train()
    teacher_model.eval()
    #print("inside train kd no thresh")
    # summary for current training loop and a running average object for loss
    summ = []
    combinedLoss_avg = RunningAverage()
    heatmap_dissimilarity_avg = RunningAverage()
    kl_loss_avg = RunningAverage()
    
    
    student_final_layer = student_model.layer4[-1]
    teacher_final_layer = teacher_model.module.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model

    if algo == "GradCAM":
     student_cam = GradCAM(model=student_model, target_layer=student_final_layer)
     teacher_cam = GradCAM(model = teacher_model, target_layer = teacher_final_layer)
    elif algo == "CAM":
        student_cam = CAM(student_model,student_final_layer,input_shape=[3,32,32])
        teacher_cam = CAM(teacher_model, teacher_final_layer,input_shape = [3,32,32])         
    else:
        "somethign went wrong with the algorithm choice"    
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
      for i, (train_batch, labels_batch) in enumerate(dataloader):
        #  print("size of train batch "+str(train_batch.shape))
          # move to GPU if available
          if params.cuda:
              train_batch, labels_batch = train_batch.cuda(), \
                                          labels_batch.cuda()
          # convert to torch Variables
          # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
          #print("sie of train batch "+str(train_batch.shape))
          # compute model output, fetch teacher output, and compute KD loss
          #output_batch = model(train_batch)

          # get one batch output from teacher_outputs list
          #student_output_batch = model(train_batch)
          if experiment != 'NoHeatmap':

            train_batch, labels_batch = Variable(train_batch, requires_grad= True), Variable(labels_batch)
            
            teacher_output_batch = teacher_model(train_batch)
            #print("train_batch shape "+str(train_batch.shape)) 
            
            if params.cuda:
              teacher_output_batch = teacher_output_batch.cuda()

          else:
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            with torch.no_grad():
              teacher_output_batch = teacher_model(train_batch)
              #print("teacher output batch type after using no grad without going into the cuda part "+str(type(teacher_output_batch)))
              if params.cuda:
                teacher_output_batch = teacher_output_batch.cuda()
               # print("teacher output batch type after using no grad AFTER going into the cuda part "+str(type(teacher_output_batch)))


          student_output_batch = model(train_batch)


          if experiment =='NoHeatmap':
                loss = loss_fn_kd_noHeatMap(outputs=student_output_batch, labels=labels_batch, teacher_outputs=teacher_output_batch, params=params)

          if experiment == 'TrueClass':
           
            student_heatmap_batch = student_cam(train_batch,(4,4))
            print("student heatmap "+str(student_heatmap_batch))
            teacher_heatmap_batch = teacher_cam(train_batch,(4,4))
            print("teacher_heatmap "+str(teacher_heatmap_batch))
           # print("one line before kl loss in train kd no thresh")
            kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)


          if experiment == 'TopClass':
           # print("using top class of teacher")
            teacher_preds = torch.sigmoid(teacher_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
            
           ## targets = [ClassifierOutputTarget(
           #     category) for category in max_indices]
            if algo == 'GradCAM':
                
                teacher_heatmap_batch = teacher_cam(train_batch,(4,4))
                #  print("teacher_heatmap "+str(teacher_heatmap_batch))
                # print("max indices "+str(max_indices))
                student_heatmap_batch = student_cam(train_batch,(4,4),max_indices)
                if (check_nan(student_heatmap_batch)):
                    print("found a NaN on the student heatmap batch")
                if (check_nan(teacher_heatmap_batch)):  
                    print("found a NaN on the student heatmap batch")
                teacher_heatmap_stack = teacher_heatmap_batch
                student_heatmap_stack = student_heatmap_batch    
            elif algo == 'CAM':
                #print("max indices shape "+str(max_indices.shape))
                max_indices_list = max_indices.tolist()
                #print(max_indices.shape)
             #   element_type = type(max_indices[0])
             #   print(element_type)
                teacher_heatmap_batch = teacher_cam(scores =teacher_output_batch,class_idx= max_indices_list)
                #print("teacher heatmap batch shape "+str(teacher_heatmap_batch.shape))
                student_heatmap_batch = student_cam(scores = student_output_batch, class_idx =max_indices_list)
                teacher_heatmap_stack = torch.stack(teacher_heatmap_batch, dim=0)    
                student_heatmap_stack = torch.stack(student_heatmap_batch,dim = 0)
                teacher_heatmap_stack = F.interpolate(teacher_heatmap_stack, size=(4,4), mode='bilinear', align_corners=True)
                teacher_heatmap_stack = teacher_heatmap_stack.squeeze()
                student_heatmap_stack = student_heatmap_stack.squeeze()
                #print("teacher stack shape "+str(teacher_heatmap_stack.shape))
                #print("teacher heatmap batch shape "+str(teacher_heatmap_batch.shape))
                #print("student heatmap batch shape "+str(student_heatmap_batch.shape))
                    
        #    print("student heatmap "+str(student_heatmap_batch))
           # student_heatmap_batch = student_cam(train_batch, targets)
            
           # print("student heatmap batch shape "+str(student_heatmap_batch.shape))
                if (check_nan(student_heatmap_stack)):
                    print("found a NaN on the student heatmap batch")
                if (check_nan(teacher_heatmap_stack)):  
                    print("found a NaN on the student heatmap batch")
            
            #print("teacher heatmap batch type "+str(teacher_heatmap_batch.dtype))  
           # teacher_heatmap_batch = getCAMBatch(teacher_activated_features.features, teacher_weight_softmax, max_indices)
            #print("one line before kl loss in train kd no thresh")
            kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_stack, teacher_heatmap_stack)


        #   if experiment == 'AllClasses':
        #     #print("using all classes")
        #     print("student activated features")
        #     print(student_activated_features.features.grad_fn)
        #     student_heatmap_batch = classCAMSbatch(student_activated_features.features, student_weight_softmax)
        #     teacher_heatmap_batch = classCAMSbatch(teacher_activated_features.features, teacher_weight_softmax)
        # #    print("one line before kl loss in train kd no thresh")
        #     kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)
        #   #with torch.no_grad():
          #    output_teacher_batch = teacher_model(train_batch)
          #if params.cuda:
           #   output_teacher_batch = output_teacher_batch.cuda()
         # print("student heatmap batch type "+str(student_heatmap_batch.shape))

          #if experiment != 'NoHeatmap':

           # loss = kl_loss
        #  running_loss += loss.item()
          #running_ap += get_ap_score(torch.Tensor.cpu(labels_batch).detach().numpy(), torch.Tensor.cpu(m(output_batch)).detach().numpy())

          heatmapbeta = 1-KLDgamma
          # clear previous gradients, compute gradients of all variables wrt loss
          
          
          combinedLossTensor = KLDgamma *kl_loss +( 100* ( heatmapbeta * heatmap_dissimilarity) )
          
          optimizer.zero_grad()
          scaler.scale(combinedLossTensor).backward()
      #    print("type of kl loss" +str(type(kl_loss)))
       #   print("type of heatmap loss "+str(type(heatmap_dissimilarity)))  
        #  print("type of combinedLoss "+str(type(combinedLossTensor)))
         # print("combinedLossTensor "+str(combinedLossTensor))
         # print("kl loss tensor gradient "+str(kl_loss.grad))
          #print("combined loss tensor gradient "+str(combinedLossTensor.grad))
          # performs updates using calculated gradients
          
          # Get gradients before optimizer step
   
    
          
          scaler.step(optimizer)
          scaler.update()
          #gradients_after_step = [param.grad.clone() for param in student_model.parameters()]
          
      #    print("GRADIENTS AFTER STEP: are they all zero?")
       #   for parameter in student_model.parameters():
        #     if torch.all(parameter == 0): 
         #        a =1
          #   else:
           #      print("parameter found (after step) where not everything is 0")
                 #print( "gradient of that parameter")
                 #non_zero_indices = torch.nonzero(parameter)
                 #for index in non_zero_indices:
                 #   value = parameter[index[0]]
                  #  print(f"Non-zero value at index {index}: {value}")
                # print(parameter.grad)   
# Check if gradients are the same
      #    gradients_equal = all(torch.equal(grad_before, grad_after) for grad_before, grad_after in zip(gradients_before_step, gradients_after_step))
       #   if gradients_equal:
        #    print("Gradients are the same before and after optimizer step.")
        #  else:
         #   print("Gradients are different before and after optimizer step.")

          #print("GRADIENTS BEFORE STEP")
          #for parameter in student_model.parameters():
          #    print(parameter.grad)
      
          #print("GRADIENTS AFTER STEP")
          #for parameter in student_model.parameters():
           #   print(parameter.grad)
          # Evaluate summaries only once in a while
          if i % params.save_summary_steps == 0:
              # extract data from torch Variable, move to cpu, convert to numpy arrays
              student_output_batch = student_output_batch.data.cpu().numpy()
              labels_batch = labels_batch.data.cpu().numpy()

              # compute all metrics on this batch
              summary_batch = {metric:metrics[metric](student_output_batch, labels_batch)
                                for metric in metrics}
              summary_batch['combinedLoss'] = combinedLossTensor.data#it was loss.data[0]
              summary_batch['heatmap_dissimilarity'] = heatmap_dissimilarity
              summary_batch['kl_loss'] = kl_loss.data
              summ.append(summary_batch)
              #wandb.log({"train_batch_kl_loss":kl_loss.data,"train_batch_heatmap_dissimilarity":heatmap_dissimilarity,
               #          "train_batch_combinedLoss":combinedLossTensor.data,} )

          # update the average loss
          combinedLoss_avg.update(combinedLossTensor.data)#it was loss.data[0]
          if experiment != 'NoHeatmap':
          #  print("heatmap_dissimilarity "+str(heatmap_dissimilarity))

            heatmap_dissimilarity_avg.update(heatmap_dissimilarity)
            kl_loss_avg.update(kl_loss.data)
            #print("heatmap_dissimilarity average "+str(heatmap_dissimilarity_avg))
        #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
            t.set_postfix({'train_average_KL_Loss': '{:05.3f}'.format(float(kl_loss_avg())),
                           'train_avgHeatmap_dissimilarity': '{:05.3f}'.format(float(heatmap_dissimilarity_avg())),
                        'train_avg_combinedLoss': '{:05.3f}'.format(combinedLoss_avg())})
          else :
            print("you're using a non vanilla train_kd method, you should call the vanilla method instead since you're not using heatmaps")
            t.set_postfix(loss='{:05.3f}'.format(combinedLoss_avg()))
          t.update()
  
        
    #      del train_batch, labels_batch, student_output_batch
     #     gc.collect()
      #    torch.cuda.empty_cache()



    # compute mean of all metrics in summary
   # tb.add_scalar('KLD average loss', loss, epoch)
    #tb.add_scalar('Heatmap loss', heatmap_dissimilarity, epoch)
    #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]}
    if wandbrun == True:
        wandb.log({"train_avg_KL_Loss": float(kl_loss_avg())})
        wandb.log({"train_avgHeatmap_dissimilarity": float(heatmap_dissimilarity_avg())})
        wandb.log({"train_avg_combinedLoss": combinedLoss_avg()})
        metrics_mean = {
             metric: np.mean(
                [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
             )
                for metric in summ[0]
         }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
  
  
  
  #  writer = SummaryWriter("torchlogs/")
   # writer.add_graph(student_model, train_batch)
    #writer.close()
    #num_samples = float(len(train_loader.dataset))
    #tr_loss_ = running_loss/num_samples
    #tr_map_ = running_ap/num_samples
    #if experiment == 'NoHeatMap':
    #  return loss_avg()

  #  print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
    #                 tr_loss_, tr_map_))
    #return loss_avg(), heatmap_dissimilarity_avg()


                  # Append the values to global arrays
  #   tr_loss.append(tr_loss_), tr_map.append(tr_map_)

def evaluate_kd_nothresh(model, teacher_model, dataloader, metrics, params, experiment, student_activated_features, teacher_activated_features, KLDgamma):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    kl_loss_avg = RunningAverage()
    summ = []
    if experiment != 'NoHeatmap':
        student_weight_softmax_params =list(model.linear.parameters()) # This gives a list of weights for the fully connected layers
        student_weight_softmax = student_weight_softmax_params[0].data
        student_weight_softmax.requires_grad = True
        teacher_weight_softmax_params = list(teacher_model.module.classifier.parameters()) # This gives a list of weights for the fully connected layers
        teacher_weight_softmax = teacher_weight_softmax_params[0].data
        teacher_weight_softmax.requires_grad = True
    
    m = torch.nn.Sigmoid()


    # compute metrics over the dataset
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for i, (data_batch, labels_batch) in enumerate(dataloader):

                # move to GPU if available
                #print("shape of label batch"+str(labels_batch.shape))
                # print("shape of data_batch"+str(data_batch.shape))
                if params.cuda:
                    data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
                # fetch the next evaluation batch
                data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                # compute model output

                student_output_batch = model(data_batch)
                teacher_output_batch = teacher_model(data_batch)
                # loss = loss_fn_kd(outputs= output_batch, labels = labels_batch, teacher_outputs =output_teacher_batch, params=params)
                #loss = 0.0  #force validation loss to zero to reduce computation time
                if experiment == 'TrueClass':
                    student_heatmap_batch = getCAMBatch(student_activated_features.features, student_weight_softmax, labels_batch)

                    teacher_heatmap_batch = getCAMBatch(teacher_activated_features.features, teacher_weight_softmax, labels_batch)


                if experiment == 'TopClass':
                    teacher_preds = torch.sigmoid(teacher_output_batch)
                    teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
                    max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
                    student_heatmap_batch = getCAMBatch(student_activated_features.features, student_weight_softmax, max_indices)
                    teacher_heatmap_batch = getCAMBatch(teacher_activated_features.features, teacher_weight_softmax, max_indices)
                # print("made teacher and student heatmap batches")

                if experiment == 'AllClasses':
                    student_heatmap_batch = classCAMSbatch(student_activated_features.features, student_weight_softmax)
                    teacher_heatmap_batch = classCAMSbatch(teacher_activated_features.features, teacher_weight_softmax)

            
                if experiment != 'NoHeatmap':     
                    kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_batch, teacher_heatmap_batch)
                    #print("loss made")
                    heatmapbeta = 1-KLDgamma
                    combinedLossTensor = KLDgamma * kl_loss + heatmapbeta * heatmap_dissimilarity
                # running_loss += loss.item() # sum up batch loss
                #running_ap += get_ap_score(torch.Tensor.cpu(labels_batch).detach().numpy(), torch.Tensor.cpu(m(output_batch)).detach().numpy())
                else:
                    kl_loss = loss_fn_kd_noHeatMap(student_output_batch, labels_batch, teacher_output_batch, params)
                
                output_batch = student_output_batch
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
                # summary_batch['loss'] = loss.data[0]
                if experiment !='NoHeatmap':
                    summary_batch['combinedLoss'] = combinedLossTensor.data
                    summary_batch['heatmap_dissimilarity'] = heatmap_dissimilarity
                summary_batch['kl_loss'] = kl_loss.data
                summ.append(summary_batch)
                
                kl_loss_avg.update(kl_loss.data)
                t.set_postfix(val_avg_kl_loss= '{:05.3f}'.format(float(kl_loss_avg())))
                t.update()
    
            #  wandb.log({"val_batch_kl_loss": kl_loss.data, "val_batch_heatmap_dissimilarity": heatmap_dissimilarity,
            #           "val_batch_combinedLoss":combinedLossTensor.data})  



            #del data_batch, labels_batch, output_batch
            #gc.collect()
            #torch.cuda.empty_cache()


        # compute mean of all metrics in summary
        #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]} #changef from np.mean to torch.mean
            metrics_mean = {
            metric: np.mean(
                [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
            )
            for metric in summ[0]
            }
        
        
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def evaluate_kd_noblur(model, teacher_model, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    loss_avg = RunningAverage()
    summ = []
    m = torch.nn.Sigmoid()


    # compute metrics over the dataset
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for i, (data_batch, labels_batch) in enumerate(dataloader):

                # move to GPU if available
                #print("shape of label batch"+str(labels_batch.shape))
                # print("shape of data_batch"+str(data_batch.shape))
                if params.cuda:
                    data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
                # fetch the next evaluation batch
                data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                # compute model output

                student_output_batch = model(data_batch)
                teacher_output_batch = teacher_model(data_batch)
                # loss = loss_fn_kd(outputs= output_batch, labels = labels_batch, teacher_outputs =output_teacher_batch, params=params)
                #loss = 0.0  #force validation loss to zero to reduce computation time
    
                kl_loss = loss_fn_kd(student_output_batch, labels_batch, teacher_output_batch, params)
                
                output_batch = student_output_batch
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
                # summary_batch['loss'] = loss.data[0]
                summary_batch['loss'] = kl_loss.data
                summ.append(summary_batch)
                
                loss_avg.update(kl_loss.data)
                t.set_postfix(val_avg_loss= '{:05.3f}'.format(float(loss_avg())))
                t.update()
    
            #  wandb.log({"val_batch_kl_loss": kl_loss.data, "val_batch_heatmap_dissimilarity": heatmap_dissimilarity,
            #           "val_batch_combinedLoss":combinedLossTensor.data})  



            #del data_batch, labels_batch, output_batch
            #gc.collect()
            #torch.cuda.empty_cache()


        # compute mean of all metrics in summary
        #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]} #changef from np.mean to torch.mean
            metrics_mean = {
            metric: np.mean(
                [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
            )
            for metric in summ[0]
            }
        
        
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def blurevaluate_kd(model, teacher_model, val_dataloader, metrics, params):
    # set model to evaluation mode
    model.eval()
    # summary for current eval loop
    loss_avg = RunningAverage()
    summ = []
    m = torch.nn.Sigmoid()
    # compute metrics over the dataset
    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as t:
            for i, (data_batch, labels_batch) in enumerate(val_dataloader):

                # move to GPU if available
                #print("shape of label batch"+str(labels_batch.shape))
                # print("shape of data_batch"+str(data_batch.shape))
                if params.cuda:
                    data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
                # fetch the next evaluation batch
                data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                # compute model output

                student_output_batch = model(data_batch)
                teacher_output_batch = teacher_model(data_batch)
                # loss = loss_fn_kd(outputs= output_batch, labels = labels_batch, teacher_outputs =output_teacher_batch, params=params)
                #loss = 0.0  #force validation loss to zero to reduce computation time
              
                loss = loss_fn_kd(student_output_batch, labels_batch, teacher_output_batch, params)
                
                output_batch = student_output_batch
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
                # summary_batch['loss'] = loss.data[0]
                summary_batch['loss'] = loss.data
                summ.append(summary_batch)
                
                loss_avg.update(loss.data)
                t.set_postfix(val_avg_loss= '{:05.3f}'.format(float(loss_avg())))
                t.update()
    
        # compute mean of all metrics in summary
        #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]} #changef from np.mean to torch.mean
            metrics_mean = {
            metric: np.mean(
                [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
            )
            for metric in summ[0]
            }
        
        
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean
    

def blurtrain_and_evaluate_kd(model, teacher_model, train_dataloader , val_dataloader, optimizer,
                       loss_fn_kd_nothresh, metrics, params, student_arch="simplenet", teacher_arch="resnext29", restore_file=None, wandbrun= False):
    
    student_model = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
           
    for epoch in range(params.num_epochs):
    #  print("memory summary before starting new epoch:")
    # print(torch.cuda.memory_summary())    
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        blurtrain_kd(model, teacher_model, optimizer, loss_fn_kd_nothresh, train_dataloader,
                metrics, params, wandbrun)
            #  wandb.log({"epoch":epoch})
        val_metrics = blurevaluate_kd(model, teacher_model, val_dataloader, metrics, params)
        val_acc = val_metrics['accuracy']
        print("validation accuracy: "+str(val_acc))
        val_loss = val_metrics['loss']
        print("validation loss: "+str(val_loss))
        if wandbrun == True:

            wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
            wandb.log({"val_loss": val_loss, "epoch": epoch+1})          
    
        # Save weights
        
        is_best = val_acc>=best_val_acc
        
        
        
        settings_dict ={
            "useKD": True,
            "student_model": student_arch,
            "teacher_model": teacher_arch,
            "usethreshold": False,          
        }
        
        path_base = "/home/smartinez/thesisfolder/modelruns"
        json_folder = '/home/smartinez/thesisfolder/modelruns/board_logs/stud_'+student_arch+'_teach_resnext29_kdblur_layer'+str(params.cam_layer)
       # destination_folder = f'{path_base}/student_{student_arch}/teacher_{teacher_arch}/experiment_{experiment}/KLDgamma_{gammastring}/' 
        destination_folder = f'{path_base}/kd_blur_student_{student_arch}_teacher_{teacher_arch}_layer{params.cam_layer}' 
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            print("destination folder "+str(destination_folder))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            
            best_json_path = os.path.join(json_folder, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(json_folder, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)



    
    
def train_and_evaluate_kd_nothresh(model, teacher_model, train_dataloader, val_dataloader, optimizer,    #  loss_fn_kd_nothresh, metrics, params, model_dir, restore_file, runconfig)
                       loss_fn_kd_nothresh, metrics, params, student_arch="simplenet", teacher_arch="resnext29", restore_file=None,experiment='TrueClass', KLDgamma=1, wandbrun= False):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    #with wandb.init(config=runconfig):
     #   config = wandb.config
    # reload weights from restore_file if specified
   
    student_model = model
#grid = torchvision.utils.make_grid(images)

#tb.add_image('images', grid)
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)




    best_val_acc = 0.0
    
    

    if experiment !='NoHeatmap':
        student_final_layer = student_model.layer4[-1]
        teacher_final_layer = teacher_model.module.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model
        student_activated_features = SaveFeatures(student_final_layer)
        #student_activated_features.requires_grad = True
        
        teacher_activated_features = SaveFeatures(teacher_final_layer) # attach the call back hook to the final layer of the model
       # teacher_activated_features.requires_grad = True
    #print("student final layer: "+str(student_final_layer))
    # Tensorboard logger setup
    
    board_logger = Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    if KLDgamma == 1: 
        print("KLD gamma = 1:  heatmap dissimilarity will be tracked but will have no impact on the loss calculation")
        
    if KLDgamma == 0:
         print("KLD gamma = 0: KL_loss will be tracked but will have no impact on the loss calculation, only heatmap dissimilarity will matter")
           
    for epoch in range(params.num_epochs):
    #  print("memory summary before starting new epoch:")
    # print(torch.cuda.memory_summary())    
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        if experiment != 'NoHeatmap':
            train_kd_nothresh(model, teacher_model, optimizer, loss_fn_kd_nothresh, train_dataloader,
                metrics, params, experiment, student_activated_features, teacher_activated_features, KLDgamma, wandbrun)
            #  wandb.log({"epoch":epoch})
            val_metrics = evaluate_kd_nothresh(model, teacher_model, val_dataloader, metrics, params, experiment, 
                                            student_activated_features, teacher_activated_features, KLDgamma)
            val_acc = val_metrics['accuracy']
            print("validation accuracy: "+str(val_acc))
            val_combinedLoss = val_metrics['combinedLoss']
            print("validation combinedLoss: "+str(val_combinedLoss))
            val_kl_loss = val_metrics['kl_loss']
            print("validation kl_loss: "+str(val_kl_loss))
            val_heatmap_dissimilarity = val_metrics['heatmap_dissimilarity']
            print("validation heatmap_dissimilarity: "+str(val_heatmap_dissimilarity))
            if wandbrun == True:

                wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
                wandb.log({"val_combined_loss": val_combinedLoss, "epoch": epoch+1})       
                wandb.log({"val_kl_loss": val_kl_loss, "epoch": epoch+1})       
                wandb.log({"val_heatmap_dissimilarity": val_heatmap_dissimilarity, "epoch": epoch+1})      
        
        else:
            train_kd_noblur(model,teacher_model,optimizer,loss_fn_kd_noHeatMap,train_dataloader,metrics,params)
        # Evaluate for one epoch on validation set
        #print("memory summary after a round of training but before evaluating")
        #print(torch.cuda.memory_summary())
            val_metrics = evaluate_kd_nothresh(model, teacher_model, val_dataloader, metrics, params, experiment, 
                                            None, None, KLDgamma)
            val_acc = val_metrics['accuracy']
            print("validation accuracy: "+str(val_acc))
    
          
            val_kl_loss = val_metrics['kl_loss']
            print("validation kl_loss: "+str(val_kl_loss))
        
            if wandbrun == True:

                wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
            
                wandb.log({"val_kl_loss": val_kl_loss, "epoch": epoch+1})       
          
        
        #print("memory summary after evaluating")
        #print(torch.cuda.memory_summary())
        
        
        # Save weights
        
        is_best = val_acc>=best_val_acc
        
        
        
        settings_dict ={
            "useKD": True,
            "student_model": student_arch,
            "teacher_model": teacher_arch,
            "usethreshold": False,
            "experiment": experiment,
            "KLDgamma": KLDgamma           
        }
        
        gammastring = int(KLDgamma * 100)
        path_base = "/home/smartinez/modelruns"
        json_folder = '/home/smartinez/modelruns/board_logs/stud_simplenet_teach_resnext29_NoHeatmap'
       # destination_folder = f'{path_base}/student_{student_arch}/teacher_{teacher_arch}/experiment_{experiment}/KLDgamma_{gammastring}/' 
        destination_folder = f'{path_base}/kd_nothresh_student_{student_arch}_teacher_{teacher_arch}_experiment_{experiment}_KLDgamma_{gammastring}' 
    
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            print("destination folder "+str(destination_folder))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            
            best_json_path = os.path.join(json_folder, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(json_folder, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)


        #UNLESS YOU HAVE AT LEAST 300GB VRAM OR MORE NEVER ACTIVATE THIS
        # #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
    # info = {
                #            'val accuracy': val_acc,
            # 'KL avg loss': loss,
            #'Heatmap loss': heatmap_dissimilarity
        #     'val accuracy': val_acc,
        #    'val KL  loss': val_kl_loss,
        #   'val Heatmap loss': val_heatmap_dissimilarity,
        #  'val combined Loss': val_combinedLoss

        #}

        # }

        #for tag, value in info.items():
        #    board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        #for tag, value in model.named_parameters():
        # tag = tag.replace('.', '/')
        #  board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)


    if  experiment != 'NoHeatmap':
        teacher_activated_features.remove()
        student_activated_features.remove()




    
    
def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,    #  loss_fn_kd_nothresh, metrics, params, model_dir, restore_file, runconfig)
                       loss_fn_kd_nothresh, metrics, params, student_arch="simplenet", teacher_arch="resnext29", restore_file=None, wandbrun= False):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    #with wandb.init(config=runconfig):
     #   config = wandb.config
    # reload weights from restore_file if specified
   
    student_model = model
#grid = torchvision.utils.make_grid(images)

#tb.add_image('images', grid)
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)




    best_val_acc = 0.0
    

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    for epoch in range(params.num_epochs):
    #  print("memory summary before starting new epoch:")
    # print(torch.cuda.memory_summary())    
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd_noblur(model, teacher_model, optimizer, loss_fn_kd, train_dataloader, metrics, params)
        
        val_metrics = evaluate_kd_noblur(model, teacher_model, val_dataloader, metrics, params)
        val_acc = val_metrics['accuracy']
        print("validation accuracy: "+str(val_acc))
         
        val_kl_loss = val_metrics['loss']
        print("validation oss: "+str(val_kl_loss))
          
        if wandbrun == True:

            wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
                  
            wandb.log({"val_loss": val_kl_loss, "epoch": epoch+1})       
            
       
         
      
        
        #print("memory summary after evaluating")
        #print(torch.cuda.memory_summary())
        
        
        # Save weights
        
        is_best = val_acc>=best_val_acc
        
        
        
        settings_dict ={
            "useKD": True,
            "student_model": student_arch,
            "teacher_model": teacher_arch,          
        }
        
        if student_arch == "simplenet":
            student_arch ="simpnet"
        path_base = "/home/smartinez/thesisfolder/modelruns"
        json_folder = f'/home/smartinez/thesisfolder/modelruns/board_logs/stud_{student_arch}_teach_resnext29_noblur'
       # destination_folder = f'{path_base}/student_{student_arch}/teacher_{teacher_arch}/experiment_{experiment}/KLDgamma_{gammastring}/' 
        destination_folder = f'{path_base}/stud_{student_arch}_teach_{teacher_arch}_noblur' 
    
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            print("destination folder "+str(destination_folder))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            
            best_json_path = os.path.join(json_folder, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(json_folder, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)


def papereval(student_model, teacher_model, dataloader, metrics, params, experiment, KLDgamma, algo):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    student_model.eval()

    # summary for current eval loop
    summ = []
    
    m = torch.nn.Sigmoid()
    combinedLoss_avg = RunningAverage()
    heatmap_dissimilarity_avg = RunningAverage()
    kl_loss_avg = RunningAverage()
    
    if params.model_version == "resnet18_distill":
        student_name = "resnet18"
        student_final_layer = student_model.layer4[-1]
    elif params.model_version == "cnn_distill":
        student_name = "simpnet"
        student_final_layer = student_model.conv2    
    teacher_final_layer = teacher_model.module.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model
    teacher_name = "resnext29"
    if algo == "GradCAM":
        student_cam = GradCAM(model=student_model, target_layer=student_final_layer)
        teacher_cam = GradCAM(model = teacher_model, target_layer = teacher_final_layer)
    elif algo == "CAM":
        student_cam = CAM(student_model, student_final_layer)
        teacher_cam = CAM(teacher_model, teacher_final_layer)
    else:
        print("something went pretty wrong with the algorithm for heatmaps at the evaluation")            
    # compute metrics over the dataset
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):

            # move to GPU if available
            #print("shape of label batch"+str(labels_batch.shape))
        # print("shape of data_batch"+str(data_batch.shape))
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            with torch.no_grad():
                student_output_batch = student_model(data_batch)
                teacher_output_batch = teacher_model(data_batch)
                # loss = loss_fn_kd(outputs= output_batch, labels = labels_batch, teacher_outputs =output_teacher_batch, params=params)
            #loss = 0.0  #force validation loss to zero to reduce computation time
            if experiment == 'TrueClass':
             print(" True Jeff")


            if experiment == 'TopClass':
                teacher_preds = torch.sigmoid(teacher_output_batch)
                teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
                max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
                
            # # print("about to make evaluation teacher heatmap")
                if algo == "GradCAM":
                    teacher_heatmap_batch = teacher_cam(data_batch,(4,4), mode = "eval")
                    student_heatmap_batch = student_cam(data_batch,(4,4),max_indices, "eval")
                    teacher_heatmap_stack = teacher_heatmap_batch
                    student_heatmap_stack = student_heatmap_batch    
                elif algo == "CAM":
                    max_indices_list = max_indices.tolist()
                    teacher_heatmap_batch = teacher_cam(scores = teacher_output_batch,class_idx= max_indices_list)
                    student_heatmap_batch = student_cam(scores = student_output_batch, class_idx =max_indices_list)
                    teacher_heatmap_stack = torch.stack(teacher_heatmap_batch, dim=0)    
                    student_heatmap_stack = torch.stack(student_heatmap_batch,dim = 0)
                    teacher_heatmap_stack = F.interpolate(teacher_heatmap_stack, size=(4,4), mode='bilinear', align_corners=True)
                    teacher_heatmap_stack = teacher_heatmap_stack.squeeze()
                    student_heatmap_stack = student_heatmap_stack.squeeze()
                #print("teacher stack shape "+str(teacher_heatmap_stack.shape))
                #print("teacher heatmap batch shape "+str(teacher_heatmap_batch.shape))
                #print("student heatmap batch shape "+str(student_heatmap_batch.shape))
                            
                else:
                    print("something went wrong with the heatma algorithm choice when generating the batch of heatmaps in eval ")    

                # print("made teacher and student heatmap batches")

            if experiment == 'AllClasses':
                print("all jeff")


            with torch.no_grad():
            #print("attempting loss")  
                kl_loss, heatmap_dissimilarity = loss_fn_kd_nothresh(student_output_batch, labels_batch, teacher_output_batch, params, student_heatmap_stack, teacher_heatmap_stack)
                #print("loss made")
                heatmapbeta = 1-KLDgamma
                combinedLossTensor = KLDgamma * kl_loss +(100* (heatmapbeta * heatmap_dissimilarity))
                

            # running_loss += loss.item() # sum up batch loss
            #running_ap += get_ap_score(torch.Tensor.cpu(labels_batch).detach().numpy(), torch.Tensor.cpu(m(output_batch)).detach().numpy())

            output_batch = student_output_batch
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
            # summary_batch['loss'] = loss.data[0]
            summary_batch['combinedLoss'] = combinedLossTensor.data
            if experiment !='NoHeatmap':
             summary_batch['heatmap_dissimilarity'] = heatmap_dissimilarity
            summary_batch['kl_loss'] = kl_loss.data
            summ.append(summary_batch)
        #  wandb.log({"val_batch_kl_loss": kl_loss.data, "val_batch_heatmap_dissimilarity": heatmap_dissimilarity,
            #           "val_batch_combinedLoss":combinedLossTensor.data})  



        #del data_batch, labels_batch, output_batch
        #gc.collect()
        #torch.cuda.empty_cache()
          # update the average loss
            combinedLoss_avg.update(combinedLossTensor.data)#it was loss.data[0
            heatmap_dissimilarity_avg.update(heatmap_dissimilarity.data)
            kl_loss_avg.update(kl_loss.data)
            #print("heatmap_dissimilarity average "+str(heatmap_dissimilarity_avg))
        #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
            t.set_postfix({'val_avg_kl_loss': '{:05.3f}'.format(float(kl_loss_avg())),
                           'val_avg_heatmap_loss': '{:05.3f}'.format(float(heatmap_dissimilarity_avg())),
                        'val_avg_combined_loss': '{:05.3f}'.format(combinedLoss_avg())})

            t.update()
  
    # compute mean of all metrics in summary
    #metrics_mean = {metric:torch.mean(torch.as_tensor([x[metric] for x in summ])) for metric in summ[0]} #changef from np.mean to torch.mean
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean

def posteval(student_model, teacher_model, dataloader, metrics, params, experiment, KLDgamma, algo):
    # set model to evaluation mode
    student_model.eval()
    # summary for current eval loop
    summ = []
    kl_loss_avg = RunningAverage()
    batch_count = len(dataloader)
    print("number of batches in this dataset (with the current batch size)")
    print(batch_count)
    if params.teacher == "resnext" or params.teacher == "resnext29" or params.model_version =="resnext" or params.model_version == "resnext29" or params.teacher == "None":
        teacher_name = "resnext29"
        teacher_final_layer = teacher_model.module.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model
        teacher_target_layers = [ 
                    #         teacher_model.module.stage_1.stage_1_bottleneck_0,
                   #          teacher_model.module.stage_1.stage_1_bottleneck_1,
                             teacher_model.module.stage_1.stage_1_bottleneck_2,
              
                     #       teacher_model.module.stage_2.stage_2_bottleneck_0,
                      #      teacher_model.module.stage_2.stage_2_bottleneck_1,
                            teacher_model.module.stage_2.stage_2_bottleneck_2,
                            
                       #     teacher_model.module.stage_3.stage_3_bottleneck_0,
                        #    teacher_model.module.stage_3.stage_3_bottleneck_1,
                            teacher_model.module.stage_3.stage_3_bottleneck_2]
        print("length of list of teacher layers")
        print(len(teacher_target_layers))
        teacher_layer_strings= [
                         #    "stg1 BNK1",
                          #   "stg1 BNK2",
                             "stg1 BNK3",
              
                           # "stg2 BNK1",
                            # "stg2 BNK2",
                             "stg2 BNK3",
                            
                             # "stg3 BNK1",
                             #"stg3 BNK2",
                             "stg3 BNK3"]
    
    if params.model_version == "resnet18_distill" or params.model_version == "resnet18_vanilla" or params.model_version == "base_resnet18":
        student_final_layer = student_model.layer4[-1]
        student_target_layers =[student_model.conv1, #initial conv
                                #layer1
                                #first basis block
                                student_model.layer1[0].conv1,
                                student_model.layer1[0].conv2,
                                #second basis block
                                student_model.layer1[1].conv1,
                                student_model.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                student_model.layer2[0].conv1,
                                student_model.layer2[0].conv2,
                                #second basis block
                                student_model.layer2[1].conv1,
                                student_model.layer2[1].conv2,
                                #layer3
                                #first basis block
                                student_model.layer3[0].conv1,
                                student_model.layer3[0].conv2,
                                #second basis block
                                student_model.layer3[1].conv1,
                                student_model.layer3[1].conv2,
                                #layer4
                                #first basis block
                                student_model.layer4[0].conv1,
                                student_model.layer4[0].conv2,
                                #second basis block
                                student_model.layer4[1].conv1,
                                student_model.layer4[1].conv2
                                ]
        student_layer_strings=["conv1", 
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
                                "layer4.block2.conv2",
                                ]
        if params.teacher =="base-KD_compare":
                teacher_layer_strings = student_layer_strings
                teacher_target_layers =[teacher_model.conv1, #initial conv
                                #layer1
                                #first basis block
                                teacher_model.layer1[0].conv1,
                                teacher_model.layer1[0].conv2,
                                #second basis block
                                teacher_model.layer1[1].conv1,
                                teacher_model.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                teacher_model.layer2[0].conv1,
                                teacher_model.layer2[0].conv2,
                                #second basis block
                                teacher_model.layer2[1].conv1,
                                teacher_model.layer2[1].conv2,
                                #layer3
                                #first basis block
                                teacher_model.layer3[0].conv1,
                                teacher_model.layer3[0].conv2,
                                #second basis block
                                teacher_model.layer3[1].conv1,
                                teacher_model.layer3[1].conv2,
                                #layer4
                                #first basis block
                                teacher_model.layer4[0].conv1,
                                teacher_model.layer4[0].conv2,
                                #second basis block
                                teacher_model.layer4[1].conv1,
                                teacher_model.layer4[1].conv2
                                ]        
        if params.model_version == "resnet18_distill": 
            print("student: resnet18 KD")
            student_name = "resnet18_KD"
            if params.teacher == "base-KD_compare":
                teacher_name = "resnet18_vanilla"
        elif params.model_version == "resnet18_vanilla" or params.model_version == "base_resnet18":
            print("student: resnet18 vanilla")    
            student_name = "resnet18_vanilla"
            if params.teacher == "base-KD_compare":
                teacher_name = "resnet18_KD"
    elif params.model_version == "cnn_distill" or params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
        student_final_layer = student_model.conv2    
        student_target_layers = [student_model.conv1, student_model.conv2, student_model.conv3]
        student_layer_strings= ["conv1", "conv2", "conv3"]
        if params.teacher == "base-KD_compare":
            teacher_layer_strings = student_layer_strings
            teacher_target_layers = [teacher_model.conv1, teacher_model.conv2, teacher_model.conv3]    
        if params.model_version == "cnn_distill":
            student_name = "simpnet_KD"
            print("student: simpnet KD")
            if params.teacher == "base-KD_compare":
                teacher_name = "simpnet_vanilla"
        elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
            student_name = "simpnet_vanilla"
            print("student: simpnet vanilla")
            print(f"teacher: {params.teacher}")
            if params.teacher == "base-KD_compare":
                teacher_name = "simpnet_KD"   
        print("student target layers")
        for element in student_target_layers:
            print(element)
    elif params.model_version =="resnext" or params.model_version == "resnext29":
        student_name = "resnext29"
        print("resnext29 student ")
        student_target_layers = teacher_target_layers
        student_layer_strings= teacher_layer_strings 
    print("length of list of student layers")
    print(len(student_target_layers))        
    avg_diss_matrix  = np.zeros((len(student_target_layers), len(teacher_target_layers)))
    baseline_avg_diss_matrix = np.zeros((len(student_target_layers), len(student_target_layers)))
    print("shape of final dissimilarity matrix")
    print(avg_diss_matrix.shape)
    if algo == "GradCAM":
        teacher_cam = GradCAM(model=teacher_model, target_layers=teacher_target_layers)
        if student_name != "resnext29":
            student_cam = GradCAM(model=student_model, target_layers=student_target_layers)
        else:
            student_cam = teacher_cam
        #print("using student also as teacher??? for some reason?")
        #teacher_cam = GradCAM(model = student_model, target_layers = student_target_layers)
    elif algo == "CAM":
        student_cam = CAM(student_model, student_final_layer)
        teacher_cam = CAM(teacher_model, teacher_final_layer)
    else:
        print("something went pretty wrong with the algorithm for heatmaps at the evaluation")            
    # compute metrics over the dataset
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):

            # move to GPU if available
            #print("shape of label batch"+str(labels_batch.shape))
        # print("shape of data_batch"+str(data_batch.shape))
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            #with torch.no_grad():
            teacher_output_batch = teacher_model(data_batch)
            if student_name != "resnext29":
                student_output_batch = student_model(data_batch)
            else: 
                student_output_batch = teacher_output_batch    
        

            if experiment == 'TopClass':
                teacher_preds = torch.sigmoid(teacher_output_batch)
                teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
                max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
                teacher_heatmap_batch = teacher_cam(data_batch)
                if student_name != "resnext29":
                    student_heatmap_batch = student_cam(data_batch,max_indices)
                else:
                    student_heatmap_batch = teacher_heatmap_batch
                
                  #  print("done with teacher heatmaps for ")
                teacher_heatmap_stack = np.array(teacher_heatmap_batch).squeeze()
                student_heatmap_stack = np.array(student_heatmap_batch).squeeze()
                
            
            kl_loss = loss_fn_kd(student_output_batch, labels_batch, teacher_output_batch, params)
            batch_diss_matrix = np.zeros((student_heatmap_stack.shape[0], teacher_heatmap_stack.shape[0]))
            baseline_batch_diss_matrix = np.zeros((student_heatmap_stack.shape[0], student_heatmap_stack.shape[0]))
            student_heatmap_tensor = torch.tensor(student_heatmap_stack)
            teacher_heatmap_tensor = torch.tensor(teacher_heatmap_stack)
            for student_layer_index, student_layer in enumerate(student_heatmap_tensor):
                for teacher_layer_index, teacher_layer in enumerate(teacher_heatmap_tensor):
                    heat_diss = MSEloss(student_heatmap_tensor[student_layer_index],teacher_heatmap_tensor[teacher_layer_index])
                    batch_diss_matrix[student_layer_index][teacher_layer_index]= heat_diss
                    
                for baseline_layer_index, baseline_student_layer in enumerate(student_heatmap_tensor):
                    baseline_heat_diss = MSEloss(student_heatmap_tensor[student_layer_index],student_heatmap_tensor[baseline_layer_index])    
                    baseline_batch_diss_matrix[student_layer_index][baseline_layer_index] = baseline_heat_diss
                    
            avg_diss_matrix = avg_diss_matrix + batch_diss_matrix  
            baseline_avg_diss_matrix = baseline_avg_diss_matrix + baseline_batch_diss_matrix       
                        
            output_batch = student_output_batch
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
            summary_batch['kl_loss'] = kl_loss.data
            summ.append(summary_batch)
        #  wandb.log({"val_batch_kl_loss": kl_loss.data, "val_batch_heatmap_dissimilarity": heatmap_dissimilarity,
            #           "val_batch_combinedLoss":combinedLossTensor.data})  



        #del data_batch, labels_batch, output_batch
        #gc.collect()
        #torch.cuda.empty_cache()
            kl_loss_avg.update(kl_loss.data)
            #print("heatmap_dissimilarity average "+str(heatmap_dissimilarity_avg))
        #    wandb.log({"train_avg_kl_loss": kl_loss_avg(), "train_avg_heatmap_dissimilarity":heatmap_dissimilarity_avg(),
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
            t.set_postfix({'val_avg_kl_loss': '{:05.3f}'.format(float(kl_loss_avg()))})
                          # 'val_avg_heatmap_loss': '{:05.3f}'.format(float(heatmap_dissimilarity_avg())),
                        #'val_avg_combined_loss': '{:05.3f}'.format(combinedLoss_avg())})

            t.update()
    avg_diss_matrix = avg_diss_matrix / batch_count
    baseline_avg_diss_matrix = baseline_avg_diss_matrix / batch_count        
    wandb.log({"avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"avg dissimilarity between {student_name} and {teacher_name}")})
    wandb.log({"avg_dissimilarity_matrix":avg_diss_matrix})
    wandb.log({"baseline avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"baseline avg dissimilarity between {student_name} and {student_name}")})
    wandb.log({"baseline avg_dissimilarity_matrix":avg_diss_matrix})
    
    #avg_diss_matrix
    
    if student_name == "simpnet_vanilla" or student_name == "simpnet_KD":
        figsizetuple = (8,8)
        baselinesize = (4.5, 4.5)
    elif student_name == "resnet18_vanilla" or student_name == "resnet18_KD":  
        figsizetuple = (9.4,9.4)  
        baselinesize = (10.4, 10.4)
    else:#student_name == "resnext29":
        figsizetuple = (8,8)
        baselinesize = (8.1, 10.1)        
    fig, ax = plt.subplots(figsize=figsizetuple)
    
    im, cbar = heatmap(avg_diss_matrix,student_layer_strings, teacher_layer_strings , ax=ax,
                   cmap="cividis", cbarlabel="avg. cam diss. (MSE)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    fig.tight_layout()
    plt.title(f"Avg  {student_name} - {teacher_name} cam diss.", fontsize=11)
    plt.subplots_adjust(top=0.93)
    # Log the plot to wandb
    wandb.log({f"Avg dissimilaritymatrix {student_name} - {teacher_name}": wandb.Image(fig)})
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=baselinesize)
    im, cbar = heatmap(baseline_avg_diss_matrix,student_layer_strings, student_layer_strings , ax=ax,
                   cmap="cividis", cbarlabel="avg. cam diss. (MSE)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.title(f"Base avg. {student_name}-{student_name} cam diss.", fontsize=11)
    plt.subplots_adjust(top=0.93)
    # Log the plot to wandb
    wandb.log({f"Base avg. dissmat {student_name}-{student_name}": wandb.Image(fig)})
    plt.show()
    
    # compute mean of all metrics in summary
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean, avg_diss_matrix

def get_layer_strings(modelname):
    if modelname == "resnext29":
        model_layer_strings= [
                             "stg1 BNK1",
                             "stg1 BNK2",
                             "stg1 BNK3",
              
                            "stg2 BNK1",
                             "stg2 BNK2",
                             "stg2 BNK3",
                            
                              "stg3 BNK1",
                             "stg3 BNK2",
                             "stg3 BNK3"]
    elif modelname in ["simpnet" ,  "simpnet_KD" , "simpnet_vanilla" , "base_cnn" , "cnn_distill"]:
        model_layer_strings= ["conv1", "conv2", "conv3"]
    elif modelname in ["resnet18", "resnet18_distill", "resnet18_KD", "resnet18_vanilla", "base_resnet18"]:
         model_layer_strings=["conv1", 
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
                                "layer4.block2.conv2",
                                ]         
    return model_layer_strings

def get_target_layers(model, modelname):
    if modelname == "resnext29":
        model_target_layers = [ 
                             model.module.stage_1.stage_1_bottleneck_0,
                             model.module.stage_1.stage_1_bottleneck_1,
                             model.module.stage_1.stage_1_bottleneck_2,
              
                            model.module.stage_2.stage_2_bottleneck_0,
                            model.module.stage_2.stage_2_bottleneck_1,
                            model.module.stage_2.stage_2_bottleneck_2,
                            
                            model.module.stage_3.stage_3_bottleneck_0,
                            model.module.stage_3.stage_3_bottleneck_1,
                            model.module.stage_3.stage_3_bottleneck_2]
        model_layer_strings= [
                             "stg1 BNK1",
                             "stg1 BNK2",
                             "stg1 BNK3",
              
                            "stg2 BNK1",
                             "stg2 BNK2",
                             "stg2 BNK3",
                            
                              "stg3 BNK1",
                             "stg3 BNK2",
                             "stg3 BNK3"]
    elif modelname in ["simpnet" ,  "simpnet_KD" , "simpnet_vanilla" , "base_cnn" , "cnn_distill"]:
        model_target_layers = [model.conv1, model.conv2, model.conv3]
        model_layer_strings= ["conv1", "conv2", "conv3"]
    elif modelname in ["resnet18", "resnet18_distill", "resnet18_KD", "resnet18_vanilla", "base_resnet18"]:
        model_target_layers =[model.conv1, #initial conv
                                #layer1
                                #first basis block
                                model.layer1[0].conv1,
                                model.layer1[0].conv2,
                                #second basis block
                                model.layer1[1].conv1,
                                model.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                model.layer2[0].conv1,
                                model.layer2[0].conv2,
                                #second basis block
                                model.layer2[1].conv1,
                                model.layer2[1].conv2,
                                #layer3
                                #first basis block
                                model.layer3[0].conv1,
                                model.layer3[0].conv2,
                                #second basis block
                                model.layer3[1].conv1,
                                model.layer3[1].conv2,
                                #layer4
                                #first basis block
                                model.layer4[0].conv1,
                                model.layer4[0].conv2,
                                #second basis block
                                model.layer4[1].conv1,
                                model.layer4[1].conv2
                                ]
        model_layer_strings=["conv1", 
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
                                "layer4.block2.conv2",
                                ]    
    return model_target_layers, model_layer_strings           

def KD_diss_compare_to_reference(vanilla_model, KD_model, teacher_model, dataloader, metrics, params):
    # set model to evaluation mode
    modelA = vanilla_model
    modelB = KD_model
    modelA.eval()
    modelB.eval()
    teacher_model.eval()
    vanilla_kl_loss_avg = RunningAverage()
    KD_kl_loss_avg = RunningAverage()
    # summary for current eval loop
    summ = []
    batch_count = len(dataloader)
    print("number of batches in this dataset (with the current batch size)")
    print(batch_count)
    resnext29_name_strings = [ "resnext" , "resnext29" ,"resnext" , "resnext29" , "None", "base-KD_compare"]
    if params.teacher in resnext29_name_strings:
        teacher_name = "resnext29"
        teacher_target_layers, teacher_layer_strings = get_target_layers(teacher_model, teacher_name)
    else:
        print(f"Teacher name in params.teacher isn't any of these strings: {resnext_29_name_strings}")    
    if params.model_version in ["resnet18_distill" , "resnet18_KD" , "resnet18_vanilla" , "base_resnet18"]:
        modelA_name = "resnet18_vanilla"
        modelB_name = "resnet18_KD"
    elif params.model_version in ["cnn_distill" , "cnn_vanilla" , "base_cnn", "simpnet_vanilla", "simpnet_KD"]:
        modelA_name = "simpnet_vanilla"
        modelB_name = "simpnet_KD"
    modelA_target_layers, modelA_layer_strings = get_target_layers(modelA, modelA_name)
    modelB_target_layers, modelB_layer_strings = get_target_layers(modelB, modelB_name)
    print(f"ModelA: {modelA_name}, modelB: {modelB_name}")
    #print("(student) model target layers:")    
    #for element in student_target_layers:
    #    print(element)
    avg_diss_matrix  = np.zeros((len(modelA_target_layers), len(modelB_target_layers)))
   
    vanilla_cam = GradCAM(model=modelA, target_layers=modelA_target_layers)
    KD_cam = GradCAM(model=modelB, target_layers=modelB_target_layers)
    teacher_cam = GradCAM(model=teacher_model, target_layers = teacher_target_layers)
    #elif algo == "CAM":
     #   student_cam = CAM(student_model, student_final_layer)
      #  teacher_cam = CAM(teacher_model, teacher_final_layer)
    # compute metrics over the dataset
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            #print("shape of label batch"+str(labels_batch.shape))
        # print("shape of data_batch"+str(data_batch.shape))
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            #with torch.no_grad():
            teacher_output_batch = teacher_model(data_batch)
            vanilla_output_batch = vanilla_model(data_batch)
            KD_output_batch = KD_model(data_batch)

            
            
            teacher_preds = torch.sigmoid(teacher_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
            teacher_heatmap_batch = teacher_cam(data_batch)
            vanilla_heatmap_batch = vanilla_cam(data_batch, max_indices)
            KD_heatmap_batch = KD_cam(data_batch, max_indices)
                
                  #  print("done with teacher heatmaps for ")
            teacher_heatmap_stack = np.array(teacher_heatmap_batch).squeeze()
            vanilla_heatmap_stack = np.array(vanilla_heatmap_batch).squeeze()
            KD_heatmap_stack = np.array(KD_heatmap_batch).squeeze()
                
            
           
            batch_diss_matrix = np.zeros((vanilla_heatmap_stack.shape[0], KD_heatmap_stack.shape[0]))
            vanilla_heatmap_tensor = torch.tensor(vanilla_heatmap_stack)
            KD_heatmap_tensor = torch.tensor(KD_heatmap_stack)
            #teacher_heatmap_tensor = torch.tensor(teacher_heatmap_stack)
            for vanilla_layer_index, vanilla_layer in enumerate(vanilla_heatmap_tensor):
                for KD_layer_index, KD_layer in enumerate(KD_heatmap_tensor):
                    heat_diss = MSEloss(vanilla_heatmap_tensor[vanilla_layer_index],KD_heatmap_tensor[KD_layer_index])
                    batch_diss_matrix[vanilla_layer_index][KD_layer_index]= heat_diss
                    
            avg_diss_matrix = avg_diss_matrix + batch_diss_matrix  
               
            
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            vanilla_output_batch = vanilla_output_batch.data.cpu().numpy()
            KD_output_batch = KD_output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            vanilla_kl_loss = loss_fn_kd(vanilla_output_batch, labels_batch, teacher_output_batch, params)
            KD_kl_loss = loss_fn_kd(KD_output_batch, labels_batch, teacher_output_batch, params)
            # compute all metrics on this batch
            summary_batch = {"vanilla_acc": accuracy(vanilla_output_batch, labels_batch),
                             "KD_acc": accuracy(KD_output_batch, labels_batch)}

            summary_batch['vanilla_kl_loss'] = vanilla_kl_loss.data
            summary_batch['KD_kl_loss'] = KD_kl_loss.data
            summ.append(summary_batch)
            wandb.log({"val_batch_KD_kl_loss": KD_kl_loss.data, "val_batch_vanilla_kl_loss": vanilla_kl_loss.data})
            #           "val_batch_combinedLoss":combinedLossTensor.data})  



        #del data_batch, labels_batch, output_batch
        #gc.collect()
        #torch.cuda.empty_cache()
            vanilla_kl_loss_avg.update(vanilla_kl_loss.data)
            KD_kl_loss_avg.update(KD_kl_loss.data)
            #print("heatmap_dissimilarity average "+str(heatmap_dissimilarity_avg))
            wandb.log({"val_avg_vanilla_kl_loss": vanilla_kl_loss_avg(), "val_avg_KD_kl_loss":DK_kl_loss_avg()})
         #            "train_avg_combinedLoss":combinedLoss_avg() }         )
            t.set_postfix({'val_avg_vanilla_kl_loss': '{:05.3f}'.format(float(vanilla_kl_loss_avg())),
               'val_avg_KD_loss': '{:05.3f}'.format(float(KD_kl_loss_avg()))})
            #t.set_postfix({'val_avg_vanilla_kl_loss': '{:05.3f}'.format(float(vanilla_kl_loss_avg())),
             #              'val_avg_KD_loss': '{:05.3f}'.format(float(KD_kl_loss_avg()))})
                        #'val_avg_combined_loss': '{:05.3f}'.format(combinedLoss_avg())})

            t.update()
    avg_diss_matrix = avg_diss_matrix / batch_count
   # baseline_avg_diss_matrix = baseline_avg_diss_matrix / batch_count        
    wandb.log({"avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"avg. diss. (vanilla-KD) of {params.model_version} wrt. {teacher_name}")})
    wandb.log({"avg_dissimilarity_matrix":avg_diss_matrix})
   # wandb.log({"baseline avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"baseline avg dissimilarity between {student_name} and {student_name}")})
   # wandb.log({"baseline avg_dissimilarity_matrix":avg_diss_matrix})
    
    #avg_diss_matrix
    
    if modelA_name == "simpnet_vanilla" or modelA_name == "simpnet_KD":
        baselinesize = (4.5, 4.5)
        #figsizetuple = (8,8)
        figsizetuple = baselinesize
        titlestring = "simpnet"
        
    elif modelA_name == "resnet18_vanilla" or modelA_name == "resnet18_KD":  
        #figsizetuple = (9.4,9.4)  
        baselinesize = (10.4, 10.4)
        figsizetuple = baselinesize
        titlestring = "resnet18"
    else:#student_name == "resnext29":
        figsizetuple = (8,8)
        baselinesize = (8.1, 10.1)        
    fig, ax = plt.subplots(figsize=figsizetuple)
    
    im, cbar = heatmap(avg_diss_matrix,modelA_layer_strings, modelB_layer_strings , ax=ax,
                   cmap="cividis", cbarlabel="avg. cam diss. (MSE)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    fig.tight_layout()
    plt.title(f"Avg. {titlestring} vanilla_KD cam diss. (wrt. resnext29)", fontsize=11)
    plt.subplots_adjust(top=0.93)
    # Log the plot to wandb
    wandb.log({f"Avg dissimilaritymatrix {titlestring} KD-vanilla wrt. resnext29": wandb.Image(fig)})
    plt.show()
    
    
    
    # compute mean of all metrics in summary
    metrics_mean = {
    metric: np.mean(
        [x[metric].cpu().detach().numpy() if isinstance(x[metric], torch.Tensor) else x[metric] for x in summ]
    )
    for metric in summ[0]
    }
    
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean, avg_diss_matrix
    
    
def multilayer_batch_overlay(images, heatmap_matrix, use_rgb= True):
    print(f"shape of hetmap matrix {heatmap_matrix.shape}")
    overlay_matrix= np.zeros((heatmap_matrix.shape[0], heatmap_matrix.shape[1], 3, heatmap_matrix.shape[2], heatmap_matrix.shape[3]))
    print(f"shape of overlay matrix: {overlay_matrix.shape}")
    if isinstance(images, torch.Tensor):
        print("the images given are tensors")  
        images =  images.cpu().detach().numpy()
        print(f"successfully converted to numpy arrays")
    for sample_index in range(heatmap_matrix.shape[0]):
        for layer_index in range(heatmap_matrix.shape[1]):
      #print(f"now inside nested loop of multilayebatchoverlay method with sample index {sample_index}, layer index {layer_index}")  
      #print(f"there are {heatmap_matrix.shape[0]} samples and {heatmap_matrix.shape[1]} target layers")
            current_heatmap = heatmap_matrix[sample_index][layer_index]
            if isinstance(current_heatmap, torch.Tensor):
                #print("the current heatmap is a tensor")
                current_heatmap = current_heatmap.cpu().detach().numpy()
            if isinstance(images[sample_index], torch.Tensor):
                #print("the current image is a tensor")
                images[sample_index] =  images[sample_index].cpu().detach().numpy()
            print("shape of current heatmap: "+str(current_heatmap.shape))
            print("current heatmap datatype: "+str(current_heatmap.dtype))
            print("shape of current image: "+str(images[sample_index].shape))
            print(f"current image datatype: {images[sample_index].dtype}")
            print(f"current image: {images[sample_index]}")
            if images[sample_index].dtype == np.uint8:
                print("converting image from int [0,255] to float32[0,1]")
                images[sample_index]= np.float32(images[sample_index]) / 255 
            current_overlay = show_cam_on_image(images[sample_index], current_heatmap, use_rgb)
            print(f"current overlay shape: {current_overlay.shape}")
            overlay_matrix[sample_index][layer_index] = current_overlay
        
    return overlay_matrix

def visual_heatmap_compare(student_model, teacher_model, dataloader, params):
    student_model.eval()
    # summary for current eval loop
    summ = []
    if params.model_version == "resnet18_distill":
        print("student: resnet18")
        student_name = "resnet18"
        student_final_layer = student_model.layer4[-1]
        student_target_layers =[student_model.conv1, #initial conv
                                #layer1
                                #first basis block
                                student_model.layer1[0].conv1,
                                student_model.layer1[0].conv2,
                                #second basis block
                                student_model.layer1[1].conv1,
                                student_model.layer1[1].conv2,
                                #layer2
                                #first basis block                       
                                student_model.layer2[0].conv1,
                                student_model.layer2[0].conv2,
                                #second basis block
                                student_model.layer2[1].conv1,
                                student_model.layer2[1].conv2,
                                #layer3
                                #first basis block
                                student_model.layer3[0].conv1,
                                student_model.layer3[0].conv2,
                                #second basis block
                                student_model.layer3[1].conv1,
                                student_model.layer3[1].conv2,
                                #layer4
                                #first basis block
                                student_model.layer4[0].conv1,
                                student_model.layer4[0].conv2,
                                #second basis block
                                student_model.layer4[1].conv1,
                                student_model.layer4[1].conv2
                                ]
        
        student_layer_strings=["conv1", 
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
                                "layer4.block2.conv2",
                                ]
        
    elif params.model_version == "cnn_distill":
        student_name = "simpnet"
        student_final_layer = student_model.conv2    
        student_target_layers = [student_model.conv1, student_model.conv2, student_model.conv3]
        student_layer_strings= ["conv1", "conv2", "conv3"]
    else: 
        print("student isn't either resnet18 nor simplenet ")  
    print("length of list of student layers")
    print(len(student_target_layers))        
    teacher_name = "resnext29"
    #teacher_final_layer = teacher_model.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model
    teacher_target_layers = [ 
#                             teacher_model.stage_1.stage_1_bottleneck_0,
 #                            teacher_model.stage_1.stage_1_bottleneck_1,
                             teacher_model.module.stage_1.stage_1_bottleneck_2,
              
  #                          teacher_model.stage_2.stage_2_bottleneck_0,
   #                         teacher_model.stage_2.stage_2_bottleneck_1,
                            teacher_model.module.stage_2.stage_2_bottleneck_2,
                            
    #                        teacher_model.stage_3.stage_3_bottleneck_0,
     #                       teacher_model.stage_3.stage_3_bottleneck_1,
                            teacher_model.module.stage_3.stage_3_bottleneck_2]
    print("length of list of teacher layers")
    print(len(teacher_target_layers))
    
    teacher_layer_strings= [
      #                       "stage_1 bottleneck_1",
       #                      "stage_1 bottleneck_2",
                             "stage_1 bottleneck_3",
              
        #                    "stage_2 bottleneck_1",
         #                    "stage_2 bottleneck_2",
                             "stage_2 bottleneck_3",
                            
          #                    "stage_3 bottleneck_1",
           #                  "stage_3 bottleneck_2",
                             "stage_3 bottleneck_3"]
    

    student_cam = GradCAM(model=student_model, target_layers=student_target_layers)
    teacher_cam = GradCAM(model=teacher_model, target_layers=teacher_target_layers)
    
    images, labels = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = images.to(device), labels.to(device)
    data_batch, labels_batch = Variable(images), Variable(labels)
    print("shape of images tensor (batch of input samples))")
    print(images.shape)
    student_output_batch = student_model(data_batch)
    teacher_output_batch = teacher_model(data_batch)
    
    teacher_preds = torch.sigmoid(teacher_output_batch)
    teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
    max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
    
    student_heatmap_batch = student_cam(data_batch, max_indices)
    teacher_heatmap_batch = teacher_cam(data_batch)
    
    teacher_heatmap_stack = np.array(teacher_heatmap_batch).squeeze()
    student_heatmap_stack = np.array(student_heatmap_batch).squeeze()
    
    print("student_heatmap_stack.shape")
    print(student_heatmap_stack.shape)
    print("teacher_heatmap_stack.shape")
    print(teacher_heatmap_stack.shape)
    
    studentcam = student_heatmap_stack.swapaxes(0,1)
    teachercam = teacher_heatmap_stack.swapaxes(0,1)
    
    print("studentcam shape")
    print(studentcam.shape)
    print("teachercam shape")
    print(teachercam.shape)
    data_type = studentcam.dtype

    print(f"Data type of studentcam: {data_type}")
    print(f"studentcam (shape {studentcam.shape}, dtype {studentcam.dtype}), images (shape {images.shape}, dtype {images.dtype})")
    student_visual = multilayer_batch_overlay(images, studentcam, use_rgb=True)
    teacher_visual = multilayer_batch_overlay(images, teachercam, use_rgb=True)
    print("student visualisatoin matrix shape "+str(student_visual.shape))
    print("teacher visualisatoin matrix shape "+str(teacher_visual.shape))
    student_rows, student_cols = student_visual.shape
    teacher_rows, teacher_cols = teacher_visual.shape

   

    

def paper_traineval(student_model, teacher_model, train_dataloader, val_dataloader, optimizer,    #  loss_fn_kd_nothresh, metrics, params, model_dir, restore_file, runconfig)
                       loss_fn_kd_nothresh, metrics, params, student_arch, teacher_arch, restore_file=None,experiment='TrueClass', KLDgamma=1, wandbrun= False, algo = 'Grad-CAM'):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    #with wandb.init(config=runconfig):
     #   config = wandb.config
    # reload weights from restore_file if specified
   
    #images, labels = next(iter(train_dataloader))
    
    #images, labels = images.to(device), labels.to(device)
#grid = torchvision.utils.make_grid(images)

    print("batch size "+str(params["batch_size"]))
    print("number of epochs "+str(params["num_epochs"]))
    print("Algorithm for heatmaps "+str(algo))
    best_val_acc = 0.0
    
   # student_final_layer = student_model.layer4[-1]
    #teacher_final_layer = teacher_model.module.stage_3.stage_3_bottleneck_2 # Grab the final layer of the model

    model_dir = '/home/smartinez/modelruns/'
    #print("student final layer: "+str(student_final_layer))
    # Tensorboard logger setup
    board_logger = Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=25, gamma=0.2)
    if KLDgamma == 1: 
        print("KLD gamma = 1:  heatmap dissimilarity will be tracked but will have no impact on the loss calculation")
        
    if KLDgamma == 0:
         print("KLD gamma = 0: KL_loss will be tracked but will have no impact on the loss calculation, only heatmap dissimilarity will matter")
           
    for epoch in range(params.num_epochs):
    #  print("memory summary before starting new epoch:")
    # print(torch.cuda.memory_summary())    
        print("-------Epoch {}----------".format(epoch+1))
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        if experiment != 'NoHeatmap':
            paperloop_nothresh(student_model, teacher_model, optimizer, loss_fn_kd_nothresh, train_dataloader,
                metrics, params, experiment, KLDgamma, wandbrun, algo)
            #  wandb.log({"epoch":epoch})

        else:
            train_kd_noblur(student_model,teacher_model,optimizer,loss_fn_kd_noHeatMap,train_dataloader,metrics,params)
        # Evaluate for one epoch on validation set
        #print("memory summary after a round of training but before evaluating")
        #print(torch.cuda.memory_summary())
        student_activated_features = 1
        teacher_activated_features = 1
        val_metrics = papereval(student_model, teacher_model, val_dataloader, metrics, params, experiment, 
                                  KLDgamma, algo)
        #print("memory summary after evaluating")
        #print(torch.cuda.memory_summary())
        
        val_acc = val_metrics['accuracy']
        print("validation accuracy: "+str(val_acc))
        val_combinedLoss = val_metrics['combinedLoss']
        print("validation combinedLoss: "+str(val_combinedLoss))
        val_kl_loss = val_metrics['kl_loss']
        print("validation kl_loss: "+str(val_kl_loss))
        val_heatmap_dissimilarity = val_metrics['heatmap_dissimilarity']
        print("validation heatmap_dissimilarity: "+str(val_heatmap_dissimilarity))
        if wandbrun == True:

           wandb.log({"val_acc": val_acc, "epoch": epoch+1})       
           wandb.log({"val_combined_loss": val_combinedLoss, "epoch": epoch+1})       
           wandb.log({"val_kl_loss": val_kl_loss, "epoch": epoch+1})       
           wandb.log({"val_heatmap_dissimilarity": val_heatmap_dissimilarity, "epoch": epoch+1})      
        # Save weights
        
        is_best = val_acc>=best_val_acc
        
        
        
        settings_dict ={
            "useKD": True,
            "student_model": student_arch,
            "teacher_model": teacher_arch,
            "usethreshold": False,
            "experiment": experiment,
            "KLDgamma": KLDgamma           
        }
        
        gammastring = int(KLDgamma * 100)
        path_base = "/home/smartinez/modelruns"
       # destination_folder = f'{path_base}/student_{student_arch}/teacher_{teacher_arch}/experiment_{experiment}/KLDgamma_{gammastring}/' 
        destination_folder = f'{path_base}/kd_nothresh_student_{student_arch}_teacher_{teacher_arch}_experiment_{experiment}_camlayer_{params.cam_layer}' 
    
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy     ")
            print("previous best validation accuracy: "+str(best_val_acc))
            best_val_acc = val_acc
            print("new best validation accuracy: "+str(best_val_acc))
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': student_model.state_dict(),
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=destination_folder)
            print("saved new best weights to "+ destination_folder)
            if wandbrun == True:
                
                wandb.log({"best_val_acc": best_val_acc, "epoch": epoch+1})
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)


#class to take a dictionary and modify it so that you can access items in it like like  dict.item instead of dict['item']
class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)





def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_weights(filename, model):
    state = model.state_dict()
    torch.save(state, filename)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    
   # filepath = os.path.join(checkpoint, '_last.pth.tar')
    filepath = f'{checkpoint}_last.pth.tar'
    print("filepath: "+str(filepath))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! {}".format(checkpoint))
        #os.mkdir(checkpoint)
       # print("directory created: "+str(checkpoint))
        
    else:
        print("Checkpoint Directory exists! ")

    print("attempting to save to "+str(filepath))
    torch.save(state, filepath)
    if is_best:
       # shutil.copyfile(filepath, os.path.join(checkpoint, '_best.pth.tar'))
       try:
        shutil.copyfile(filepath,  f'{checkpoint}_best.pth.tar')
        print("saved to best.pth.tar")
       except shutil.SameFileError:
        print("same file error")   
        pass
      # shutil.copyfile(filepath,  f'{checkpoint}_last.pth.tar')



def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise Exception("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def check_int(variable):
    if not isinstance(variable, int):
        #print("Error, index is not integer")
        # You can raise an exception here to stop the execution if needed.
        # raise ValueError("Error, index is not integer")
        return True

def print_if_all_zero(array_or_tensor):
    if isinstance(array_or_tensor, np.ndarray):
        array_or_tensor = torch.tensor(array_or_tensor)

    if torch.all(array_or_tensor == 0):
        print(array_or_tensor)
        return True

def heatmapeval(config=None):
    with wandb.init(config=config):
        config = wandb.config
        #print("line before dotdict")
        params = DotDict(simpnetparams)
        #print("line before params.cuda")
        params.cuda = torch.cuda.is_available()
        random.seed(230)
        torch.manual_seed(230)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if params.cuda: torch.cuda.manual_seed(230)
        if params.model_version == "resnet18_distill":
            student_model = ResNet18()
            student_checkpoint = '/home/smartinez/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
        if params.model_version == "cnn_distill":    
            student_model = Net(params).cuda() if params.cuda else Net(params)
            student_checkpoint = '/home/smartinez/modelruns/kd_nothresh_student_simplenet_teacher_resnext29_experiment_NoHeatmap_KLDgamma_100_best.pth.tar' 
        student_model.to(device)
        load_checkpoint(student_checkpoint, student_model);
        print("loaded student weights")
       
        #val_dl = fetch_subset_dataloader('dev', params)
       
        val_dl = fetch_dataloader('dev', params)
        print("initialising teacher model")
        teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
        teacher_checkpoint = '/home/smartinez/experiments/base_resnext29/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()
        print("teacher model loaded")
        teacher_model.to(device);
        print("teacher model loaded to gpu")
        load_checkpoint(teacher_checkpoint, teacher_model);
        print("loaded teacher weights")
        #optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
         #             momentum=0.9, weight_decay=5e-4)
        #print("created optimiser")
        #model_dir = '/bin/smartinez/resnet18minimaltest'
        #restore_file = None
        metrics = {
        'accuracy': accuracy
        }
        val_metrics = papereval(student_model, teacher_model, val_dl, metrics, params, experiment = config.experiment, KLDgamma= config.KLDgamma, algo= "GradCAM")
        
        val_acc = val_metrics['accuracy']
        print("validation accuracy: "+str(val_acc))
        val_kl_loss = val_metrics['kl_loss']
        print("validation kl_loss: "+str(val_kl_loss))
        val_heatmap_dissimilarity = val_metrics['heatmap_dissimilarity']
        print("validation heatmap_dissimilarity: "+str(val_heatmap_dissimilarity))
        wandb.log({"val_acc": val_acc})            
        wandb.log({"val_kl_loss": val_kl_loss})       
        wandb.log({"val_heatmap_dissimilarity": val_heatmap_dissimilarity})   
 


        
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

def fullblurkdtrainrun(runname, projectname, config=None ,params= None, train_dl = None, val_dl = None):
    with wandb.init(config=config, project = projectname, name = runname):
       # print("line before config")
        config = wandb.config
        #print("line before dotdict")
        params = DotDict(params)
        #print("line before params.cuda")
        params.cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the random seed for reproducible experiments
        random.seed(230)
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)
        if params.model_version == "resnet18_distill":
            print("resnet student")
            student_model = ResNet18().cuda() if params.cuda else ResNet18()
            #student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
            student_arch = "resnet18"
        elif params.model_version == "cnn_distill":    
            print("Simpnet student")       
            student_model = Net(params).cuda() if params.cuda else Net(params)  
            student_arch = "simpnet"
        else:
            print("student isn't simpnet nor resnet18")    
        optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
                      momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate)
        #student_model = ResNet18().cuda() if params.cuda else ResNet18()
        print("student model initialised from scratch")
        #train_dl = fetch_dataloader('train', params)
       # val_dl = fetch_dataloader('dev', params)
        teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
        teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()
        teacher_model.to(device);
        load_checkpoint(teacher_checkpoint, teacher_model);
        #optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
        #              momentum=0.9, weight_decay=5e-4)
       # model_dir = '/bin/smartinez/resnet18noheatmap'
        restore_file = None
        #experiment = 'AllClasses'
        #KLDgamma = 0.3 #0.3 is the best but 1.0 is for the control experiment

        metrics = {
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
}       
       # print("line before calling train and evaluate")
       
        blurtrain_and_evaluate_kd(student_model, teacher_model, train_dl, val_dl, optimizer,
                       loss_fn_kd_nothresh, metrics, params, student_arch, teacher_arch="resnext29", restore_file= None, wandbrun = True )



def fullkdtrainrun(runname, projectname, config=None ,params= None, train_dl = None, val_dl = None):
  with wandb.init(config=config, project = projectname, name = runname):
       # print("line before config")
        config = wandb.config
        #print("line before dotdict")
        params = DotDict(params)
        #print("line before params.cuda")
        params.cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the random seed for reproducible experiments
        random.seed(230)
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)
        if params.model_version == "resnet18_distill":
            print("resnet student")
            student_model = ResNet18().cuda() if params.cuda else ResNet18()
            student_arch = "resnet18"
            #student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
        elif params.model_version == "cnn_distill":    
            print("Simpnet student")       
            student_model = Net(params).cuda() if params.cuda else Net(params)
            student_arch = "simplenet"
        else:
            print("student isn't simpnet nor resnet18")    
        optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
                      momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate)
        #student_model = ResNet18().cuda() if params.cuda else ResNet18()
        print("student model initialised from scratch")
        #train_dl = fetch_dataloader('train', params)
       # val_dl = fetch_dataloader('dev', params)
        teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=10)
        teacher_checkpoint = '/home/smartinez/thesisfolder/experiments/base_resnext29/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()
        teacher_model.to(device);
        load_checkpoint(teacher_checkpoint, teacher_model);
        #optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
        #              momentum=0.9, weight_decay=5e-4)
       # model_dir = '/bin/smartinez/resnet18noheatmap'
        restore_file = None
        #experiment = 'AllClasses'
        #KLDgamma = 0.3 #0.3 is the best but 1.0 is for the control experiment

        metrics = {
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
}       
       # print("line before calling train and evaluate")
       
        train_and_evaluate_kd(student_model, teacher_model, train_dl, val_dl, optimizer,
                       loss_fn_kd, metrics, params, student_arch, teacher_arch="resnext29", restore_file= None, wandbrun = True )


def fullvanillarun(runname, projectname, config=None ,params = None, train_dl = None, val_dl = None):
        
    with wandb.init(config=config, project = projectname, name = runname):
       # print("line before config")
        config = wandb.config
        #print("line before dotdict")
        #params = DotDict(kd_data)
        #print("line before params.cuda")
        params.cuda = torch.cuda.is_available()

# Set the random seed for reproducible experiments
        random.seed(230)
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)
        if params.model_version == "base_resnet18":
            print("resnet student")
            student_model = ResNet18().cuda() if params.cuda else ResNet18()
            #student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
        elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":     
            print("Simpnet student")       
            student_model = Net(params).cuda() if params.cuda else Net(params)
        else:
            print("student isn't simpnet nor resnet18")    
        
        print("student model initialised from scratch")
        optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
                      momentum=0.9, weight_decay=5e-4)
        model_dir = ''
        restore_file = None
        #experiment = 'AllClasses'
        #KLDgamma = 0.3 #0.3 is the best but 1.0 is for the control experiment
        
        metrics = {
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
        }       
        train_and_evaluate(student_model, train_dl, val_dl, optimizer, loss_fn, metrics, params,
                           model_dir, restore_file, wandbrun = True) 
       # print("line before calling train and evaluate")


def fullvanillablurrun(runname, projectname, config=None ,params = None, train_dl = None, val_dl = None):
        
    with wandb.init(config=config, project = projectname, name = runname):
       # print("line before config")
        config = wandb.config
        #print("line before dotdict")
        #params = DotDict(kd_data)
        #print("line before params.cuda")
        params.cuda = torch.cuda.is_available()

# Set the random seed for reproducible experiments
        random.seed(230)
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)
        if params.model_version == "base_resnet18":
            print("resnet student")
            student_model = ResNet18().cuda() if params.cuda else ResNet18()
            #student_checkpoint = '/home/smartinez/thesisfolder/experiments/resnet18_distill/resnext_teacher/best.pth.tar'
        elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":     
            print("Simpnet student")       
            student_model = Net(params).cuda() if params.cuda else Net(params)
        else:
            print("student isn't simpnet nor resnet18")    
        
        print("student model initialised from scratch")
        optimizer = optim.SGD(student_model.parameters(), lr=params.learning_rate,
                      momentum=0.9, weight_decay=5e-4)
        model_dir = ''
        restore_file = None
        #experiment = 'AllClasses'
        #KLDgamma = 0.3 #0.3 is the best but 1.0 is for the control experiment
        
        metrics = {
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
        }       
        blurtrain_and_evaluate(student_model, train_dl, val_dl, optimizer, loss_fn, metrics, params,
                           model_dir, restore_file, wandbrun = True) 
       # print("line before calling train and evaluate")

class camblur_CIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        cam_layer: int = -1,
        heatdictpath: str = '/home/smartinez/thesisfolder/train_dl_cam_dict_resnext_teach_only.pkl'
    ):
        super().__init__(root, train, transform, target_transform, download)
        self.cam_layer = cam_layer
        print("cam layer "+str(cam_layer))
        self.heatdictpath = heatdictpath
        with open(self.heatdictpath, 'rb') as file:
            self.heatdict = pickle.load(file)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
       # print(f"img type{type(img)}")
         #this will most likely NOT WORK
        img = torch.from_numpy(img).cuda() #shape is 32,32,3 uint8 tensor
        #print(img.shape)
        #print("img itself")
        #print(img.dtype)
        img = img.permute(2,0,1)
        original = img.detach().clone().cuda()
        
        #print("shape after perfmutation" +str(img.shape))
        img = tensorblurmask(img, self.heatdict[index][(self.cam_layer)-1])   #here we should have a float image in range between 0 and 1
       # if not isinstance(img, Image.Image):
          #  print(f"img dtype {img.dtype}")
          #  print("not of type image")                
           #     print("of type numpy array")
                #if img.dtype == np.float32 :
                  #  print (f"datatype of array {img.dtype}")
                  #  print(img)
        #    ## if img.max() <= 1:
        #     #   print( "image seems to be between 0 and 1") 
        #        print(f"dtype is {img.dtype}")
        #       #  img = Image.fromarray((img * 255).astype(np.uint8))
        #     else:
        #         #print("0 to 255 apparently")
        #         print(f"dtype is {img.dtype}")
        #         print("image is between 0 and 255")
        #         #img = img.int()
                
                #img= ToPILImage(img)
                #img = Image.fromarray((img).astype(np.uint8))
        #print(f"shape of tensor image {img.shape} before transforming")        
        if self.transform is not None:
           # print(f"image shape {img.shape}")
            #print(img)
            #print("inside of self transform")
            original = self.transform(original)
            img = self.transform(img)
            
            #print(f"shape of tensor image after transforming {img.shape}")
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #print(f"index {index}")
        #print(f"shape of heatdict[index] {self.heatdict[index].shape}")
       # img = blurmask(img, self.heatdict[index][(self.cam_layer)-1])    
        #print("about to return image")
        #print(f"image type {img.dtype}")
        return img, original, target

        


# class GpuDataset(Dataset):
#     """
#     Dataset with all samples already on the GPU for much faster learning due
#     to reduced unneeded device IO.
#     Make sure device has enough memory to contain both the dataset and the model.
#     """
#     data_tensor: torch.Tensor
#     targets: List[int]
#     targets_tensor: torch.Tensor
#     _length: int

#     def __init__(self, dataset: Dataset) -> None:
#         self.data_tensor = torch.stack([t[0] for t in dataset], dim=0).cuda()
#         self.targets = dataset.targets
#         self.targets_tensor = torch.tensor(self.targets).cuda()
#         self._length = len(self.targets)

#     def __getitem__(self, index):
#         return (self.data_tensor[index], self.targets_tensor[index])

#     def __len__(self):
#         return self._length



def save_heatmapset(dataloader, model_cam, file_path, params):
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
    print(f"saved to {file_path}")
    
    


def save_TopTeacher_heatdict(dataloader, student_cam, teacher_model, file_path, params):
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

             
            teacher_output_batch = teacher_model(data_batch)
            #print("train_batch shape "+str(train_batch.shape)) 
            
            if params.cuda:
              teacher_output_batch = teacher_output_batch.cuda()
            teacher_preds = torch.sigmoid(teacher_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            max_probs, max_indices = torch.max(teacher_pred_probs, dim=1)
            student_heatmap_batch = student_cam(data_batch,max_indices.tolist())
            # Compute heatmaps for the batch
            #model_heatmap_stack_batch = student_cam(data_batch,)
            model_heatmap_stack_batch = np.array(student_heatmap_batch).squeeze()
            
            model_heatmap_stack_batch = model_heatmap_stack_batch.swapaxes(0,1)
        # print(f"shape{resnext_heatmap_stack_batch.shape}")

            for key, value in zip(index_batch, model_heatmap_stack_batch):
                #print("value shape" + value.shape)
                #print(f"key item {key.item()}")
                heatmap_thing[key.item()] = value
        
            t.update()
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(heatmap_thing, pickle_file)
    print(f"saved to {file_path}")    
    
    


def save_TopTeacher_heatdict_indexing(dataloader,student_model, student_cam, teacher_model, heatmap_file_path, indexing_file_path, params):
# Define batch size
    heatmap_thing = {}
    index_dictionary= {}
    

    with tqdm(len(dataloader)) as t:
        for i, (data_batch, labels_batch, index_batch) in enumerate(dataloader):   

            if params.cuda:
                data_batch, labels_batch, index_batch = data_batch.cuda(), labels_batch.cuda(), index_batch.cuda()
        # print(f"index batch type {index_batch.dtype}")
            #print(f"index batch:")
            #print(index_batch)  

            # Compute model output for the batch

             
            teacher_output_batch = teacher_model(data_batch)
            #print("train_batch shape "+str(train_batch.shape)) 
            student_output_batch = student_model(data_batch)
            if params.cuda:
              teacher_output_batch = teacher_output_batch.cuda()
              student_output_batch = student_output_batch.cuda()
              
            teacher_preds = torch.sigmoid(teacher_output_batch)
            teacher_pred_probs = F.softmax(teacher_preds, dim=1).data.squeeze()
            teacher_max_probs, teacher_max_indices = torch.max(teacher_pred_probs, dim=1)
            
            student_preds = torch.sigmoid(student_output_batch)
            student_pred_probs = F.softmax(student_preds, dim=1).data.squeeze()
            student_max_probs, student_max_indices = torch.max(student_pred_probs, dim=1)
            
            max_indices_array = np.array([teacher_max_indices.tolist(),student_max_indices.tolist()])
          
            student_heatmap_batch = student_cam(data_batch,teacher_max_indices.tolist())
            # Compute heatmaps for the batch
            #model_heatmap_stack_batch = student_cam(data_batch,)
            model_heatmap_stack_batch = np.array(student_heatmap_batch).squeeze()
            
            model_heatmap_stack_batch = model_heatmap_stack_batch.swapaxes(0,1)
        # print(f"shape{resnext_heatmap_stack_batch.shape}")

            for key, value in zip(index_batch, model_heatmap_stack_batch):
                #print("value shape" + value.shape)
                #print(f"key item {key.item()}")
                heatmap_thing[key.item()] = value
            
            for key, value in zip(index_batch, max_indices_array):
              # print(f"max indices array {max_indices_array}")
                
                index_dictionary[key.item()] = value    
                
                
        
            t.update()
    with open(heatmap_file_path, 'wb') as pickle_file:
        pickle.dump(heatmap_thing, pickle_file)
    print(f"saved to {heatmap_file_path}")
            
    with open(indexing_file_path, 'wb') as pickle_file:
        pickle.dump(index_dictionary, pickle_file)
    print(f"saved to {indexing_file_path}")        
    

def pickledposteval(student_dict, teacher_dict, params):
    student_heatarray = np.zeros((3,len(student_dict),32,32))
   # print(f"shape of student heatarrayt {student_heatarray.shape}")
   # print(f"dict entry index 0 shape{student_dict[0].shape}")
   # print(f"type of that entry {type(student_dict[0])}")
   # print(f"dtype of that entry {student_dict[0].dtype}")
    teacher_heatarray = np.zeros((3,len(teacher_dict),32,32))
    #print(f"lenght of student dict {len(student_dict)}")
    for indexkey, heatmaptriplet in student_dict.items():
     #   print(f"type of triplet {type(heatmaptriplet)}")
      #  print(f"shape of triplet {heatmaptriplet.shape}")
        student_heatarray[0][indexkey] = heatmaptriplet[0]
        student_heatarray[1][indexkey] = heatmaptriplet[1]
        student_heatarray[2][indexkey] = heatmaptriplet[2]
    
    for indexkey, heatmaptriplet in teacher_dict.items():
        teacher_heatarray[0][indexkey] = heatmaptriplet[0]
        teacher_heatarray[1][indexkey] = heatmaptriplet[1]
        teacher_heatarray[2][indexkey] = heatmaptriplet[2]
        
        
    
    if params.teacher == "resnext" or params.teacher == "resnext29" or params.model_version =="resnext" or params.model_version == "resnext29" or params.teacher == "None" or params.teacher == "none":
        teacher_name = "resnext29"
        teacher_layer_strings= [
                      
                             "stg1 BNK3",
              
                             "stg2 BNK3",
                      
                             "stg3 BNK3"]
    
    if params.model_version == "resnet18_distill" or params.model_version == "resnet18_vanilla" or params.model_version == "base_resnet18":
       
        student_layer_strings=[
                               
                               
                                "layer1.blc2.conv2",
                                
                               
                                "layer2.blc2.conv2",
                            
                                "layer4.blc2.conv2",
                                ]
        if params.teacher =="base-KD_compare":
                teacher_layer_strings = student_layer_strings
        if params.model_version == "resnet18_distill": 
            if params.cam_layer == -1:
                print("student: resnet18 KD noblur")
                student_name = "resnet18_KD_noblur"
            else:
                print(f"student: resnet18 KD blur{params.cam_layer}")
                student_name = f"resnet18_KD_blur{params.cam_layer}"
            #if params.teacher == "base-KD_compare":
             #   teacher_name = "resnet18_vanilla"
        elif params.model_version == "resnet18_vanilla" or params.model_version == "base_resnet18":
            if params.cam_layer == -1:
                print("student: resnet18 vanilla noblur")
                student_name = "resnet18_vanilla_noblur"
            else:
                print(f"student: resnet18 vanilla blur{params.cam_layer}")
                student_name = f"resnet18_vanilla_blur{params.cam_layer}"
            #if params.teacher == "base-KD_compare":
             #   teacher_name = "resnet18_KD"
    elif params.model_version == "cnn_distill" or params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
        
        student_layer_strings= ["conv1", "conv2", "conv3"]
     #   if params.teacher == "base-KD_compare":
      #      teacher_layer_strings = student_layer_strings
          
        if params.model_version == "cnn_distill":
            if params.cam_layer == -1:
                print("student: simpnet KD noblur")
                student_name = "simpnet_KD_noblur"
            else:
                print(f"student: simpnet KD blur{params.cam_layer}")
                student_name = f"simpnet_KD_blur{params.cam_layer}"
            #if params.teacher == "base-KD_compare":
             #   teacher_name = "simpnet_vanilla"
        elif params.model_version == "cnn_vanilla" or params.model_version == "base_cnn":
            if params.cam_layer == -1:
                print("student: simpnet vanilla noblur")
                student_name = "simpnet_vanilla_noblur"
            else:
                print(f"student: simpnet vanilla blur{params.cam_layer}")
                student_name = f"simpnet_vanilla_camlayer{params.cam_layer}"
           # if params.teacher == "base-KD_compare":
            #    teacher_name = "simpnet_KD"   
      
    elif params.model_version =="resnext" or params.model_version == "resnext29":
        student_name = "resnext29"
        print("resnext29 student ")
        student_layer_strings= teacher_layer_strings 
    avg_diss_matrix  = np.zeros((len(student_layer_strings), len(teacher_layer_strings)))
    baseline_avg_diss_matrix = np.zeros((len(student_layer_strings), len(student_layer_strings)))
    print("shape of final dissimilarity matrix")
    print(avg_diss_matrix.shape)

           
        
    student_heatarray = student_heatarray.squeeze()
    teacher_heatarray = teacher_heatarray.squeeze()
                
            
         
    student_tensor = torch.tensor(student_heatarray)
    teacher_tensor = torch.tensor(teacher_heatarray)
    if params.cuda:
        student_tensor, teacher_tensor = student_tensor.cuda(), teacher_tensor.cuda()
           
    for student_layer_index, student_layer in enumerate(student_tensor):
        for teacher_layer_index, teacher_layer in enumerate(teacher_tensor):
            heat_diss = MSEloss(student_tensor[student_layer_index],teacher_tensor[teacher_layer_index])
           # print(f"heat_dis {heat_diss}")
            avg_diss_matrix[student_layer_index][teacher_layer_index]= heat_diss
                   
        for baseline_layer_index, baseline_student_layer in enumerate(student_tensor):
            baseline_heat_diss = MSEloss(student_tensor[student_layer_index],student_tensor[baseline_layer_index])    
            baseline_avg_diss_matrix[student_layer_index][baseline_layer_index] = baseline_heat_diss
    print(f"average heat dis  {avg_diss_matrix}")                    
    avg_diss_matrix = avg_diss_matrix 
    baseline_avg_diss_matrix = baseline_avg_diss_matrix       
    wandb.log({"avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"avg dissimilarity between {student_name} and {teacher_name}")})
    wandb.log({"avg_dissimilarity_matrix":avg_diss_matrix})
    wandb.log({"baseline avg_dissimilarity_matrix": wandb.Image(avg_diss_matrix, caption=f"baseline avg dissimilarity between {student_name} and {student_name}")})
    wandb.log({"baseline avg_dissimilarity_matrix":avg_diss_matrix})
    
    #avg_diss_matrix
    
    if student_name == "simpnet_vanilla" or student_name == "simpnet_KD":
        figsizetuple = (3.4,3.4)
        baselinesize = (3.4, 3.4)
    elif student_name == "resnet18_vanilla" or student_name == "resnet18_KD":  
        figsizetuple = (4,4)  
        baselinesize = (4,4)
    else:#student_name == "resnext29":
        figsizetuple = (3.4,3.4)
        baselinesize = (3.4,3.4)        
    fig, ax = plt.subplots(figsize=figsizetuple)
    
    im, cbar = heatmap(avg_diss_matrix,student_layer_strings, teacher_layer_strings , ax=ax,
                   cmap="cividis", cbarlabel="avg. cam diss. (MSE)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    fig.tight_layout()
    plt.title(f"Avg {student_name}-{teacher_name} diss.", fontsize=11)
    plt.subplots_adjust(top=0.93)
    # Log the plot to wandbob
    wandb.log({f"Avg dissimilaritymatrix {student_name} - {teacher_name}": wandb.Image(fig)})
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=baselinesize)
    im, cbar = heatmap(baseline_avg_diss_matrix,student_layer_strings, student_layer_strings , ax=ax,
                   cmap="cividis", cbarlabel="avg. cam diss. (MSE)")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.title(f"Base avg. {student_name}-{student_name} cam diss.", fontsize=11)
    plt.subplots_adjust(top=0.93)
    # Log the plot to wandb
    wandb.log({f"Base avg. dissmat {student_name}-{student_name}": wandb.Image(fig)})
    plt.show()
    
   
    

    return avg_diss_matrix    