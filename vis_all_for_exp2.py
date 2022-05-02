import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import cv2
from PIL import Image
from torchvision import transforms


# Load model for cell counting
model1 = vgg19()
device = torch.device('cuda')
model1.to(device)
model1.load_state_dict(torch.load('/datasets/model_crowd_counting/best_model_bc.pth'))


# Create the preprocessing transformation here
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# load your images
path_1 = "/datasets/exp2_t_start"
path_2 = "/datasets/exp2_t_end"
path_3 = "/YOLOX_Output/" #The output image with bounding box from YOLOX

# creating exel document
import pandas as pd
writer = pd.ExcelWriter('result_exp2.xlsx', engine='xlsxwriter')
count_list = []
name_list = []

        
# loop over files
for file in os.listdir(path_1):
    filename_1 = os.fsdecode(file)
                       
    img1 = cv2.imread(path_1+filename_1)
                       
    filename_2 = filename_1.replace('t04', 't12')
    img2 = cv2.imread(path_2+filename_2)
    
    filename3 = filename_1.replace('.tif', '_diff.png')
    img3 = cv2.imread(path_3+filename3)
                       
    # Transform
    input1 = transform(img1)

    # unsqueeze batch dimension, in case you are dealing with a single image
    input1 = input1.unsqueeze(0)
    input1 = input1.to(device)


    # Get prediction
    with torch.set_grad_enabled(False):
        output1 = model1(input1)
        count = (torch.sum(output1).item())
        count_list.append(count)
        name_list.append(filename_1)
    df = pd.DataFrame({'Name': name_list, 'Growing cells': count_list})
    df.to_excel(writer, index=False, header=True)

        
    # visualise all the images
    fig = plt.figure()

    ax1 = fig.add_subplot(1,3,1)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.text(1, 15, "Start", fontsize=5, weight="bold", color = 'k')
    plt.text(1, 35, "Est. amount of cells: {}".format(round(torch.sum(output1).item())), fontsize=5, weight="bold", color = 'k')
    ax1.imshow(img1)

    ax2 = fig.add_subplot(1,3,2)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.text(1, 15, "End", fontsize=5, weight="bold", color = 'k')
    ax2.imshow(img2)
            
    ax3 = fig.add_subplot(1,3,3)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    ax3.imshow(img3)
    
    print('Save at', filename_1.replace('.tif', ".png"))
    plt.savefig('Output/'+filename_1.replace('.tif', ".png"), transparent=False, bbox_inches='tight', pad_inches=0.0, dpi=400)
    plt.close()

writer.save()