
# coding: utf-8

# In[2]:


# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

NUM_OF_CLASS = 14
CLASS = { "Atelectasis" : 0,
"Cardiomegaly" : 1,
"Effusion" : 2,
"Infiltration" : 3,
"Mass" : 4,
"Nodule" : 5,
"Pneumonia" : 6,
"Pneumothorax" : 7,
"Consolidation" : 8,
"Edema" : 9,
"Emphysema" : 10,
"Fibrosis" : 11,
"Pleural_Thickening" : 12,
"Hernia" : 13}

use_gpu = True

num_input = 100 #112120

########## Input  ##########
#labels
labels_dataframe = pd.read_csv("/datasets/ChestXray-NIHCC/Data_Entry_2017.csv").ix[:,:2]
labels_table = pd.DataFrame.as_matrix(labels_dataframe)[:,1]
NUM_OF_INPUT = len(labels_table)
label_list = []
for i in labels_table:
    res = [0.0]*14
    arr = i.split('|')
#     print (arr)
    for j in arr:
        if j!="No Finding":
            res[CLASS[j]] = 1.0
    label_list.append(res)
if use_gpu:
    labels = Variable(torch.from_numpy(np.asarray(label_list)).cuda())
else:
    labels = Variable(torch.from_numpy(np.asarray(label_list)))
labels = labels[0:num_input]
# print (labels)
    
#print (Variable(labels))


#images = torch.load("/datasets/tmp/gray_scaled_images/00000001_000.png")
img_path = "/datasets/tmp/images/"

image_list = []
#map each Tensor to a label
for i in range(0, num_input):
    curr_img_name = str(labels_dataframe["Image Index"][i])
    readin = torch.load(""+img_path+curr_img_name).numpy()
    image_list.append(readin)
if use_gpu:
    images = Variable(torch.from_numpy(np.asarray(image_list)).cuda())
else:
    images = Variable(torch.from_numpy(np.asarray(image_list)))

# print (images)

########### Network ##########

N_FOLD_CROSSVALID = 10
NUM_OF_VALID = num_input / N_FOLD_CROSSVALID
print ("NUM_OF_VALID is %d" % (NUM_OF_VALID))

def train_model(model, criterion, optimizer, scheduler, 
                N_FOLD_CROSSVALID, counter, num_epochs=25):
    since = time.time()
    
    global labels
    global images

    n_loss = 1.0
    start_idx = int(counter*NUM_OF_VALID)
    end_idx = int(start_idx+NUM_OF_VALID)
    print (start_idx)
    print (end_idx)
    training = []
    teaching = []
    validation_input = images[start_idx:end_idx,:,:,:]
    validation_output = labels[start_idx:end_idx,:].float()
    if start_idx>0 and end_idx<num_input:
        training = torch.cat((images[0:start_idx,:,:,:],                              images[end_idx:num_input,:,:,:]),0)
        teaching = torch.cat((labels[0:start_idx,:],                              labels[end_idx:num_input,:]),0).float()
    elif start_idx==0:
        training = images[end_idx:num_input,:,:,:]
        teaching = labels[end_idx:num_input,:].float()
    elif end_idx==num_input:
        training = images[0:start_idx,:,:,:]
        teaching = labels[0:start_idx,:].float()

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_corrects = 0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(training)
#         print (outputs.data.shape)
#         print (teaching.data.shape)

        loss = criterion(outputs, teaching)
        loss.backward()
        optimizer.step()

        # statistics
        #accuracy - check against teaching and outputs
        #process output
        mean_tensor = torch.sum(outputs,1)
        mean_tensor = (torch.div(mean_tensor,14.0))
        prediction = outputs
            
        
        for row in range(start_idx,end_idx):
            for col in range (0,14):
                if(outputs.data[row][col] < mean_tensor.data[row]):
                    prediction.data[row][col] = 0
                else:
                    prediction.data[row][col] = 1
        
        
        #calculate accuracy
        flag = 0
        for row in range(start_idx, end_idx):
            flag = 0
            for col in range (0,14):
                if(int(teaching.data[row][col]) == int(prediction.data[row][col])):
                    flag = flag + 1
            correct = float(flag/14)
        
        
        print ('Correct classifications: %f' %(correct))
        correct = 0
        
        running_loss += loss.data[0]

        # print statistics
        running_loss += loss.data[0]
        print('[net %d epoch %d] loss: %.3f' %
              (counter, epoch + 1, running_loss))
        running_loss = 0.0

    v_outputs = model(validation_input)
    n_loss = criterion(v_outputs,validation_output).data[0]
    #print ("net %d loss on validation set is %f" % (counter,n_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
#     print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return model

for counter in range(0,10):
    # Load a pretrained model and reset final fully connected layer.
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    sig = nn.Sigmoid()
    # model_ft.fc = nn.Linear(num_ftrs, NUM_OF_CLASS)
    class Sig_fc(nn.Module):
        def __init__(self, fc):
            super(Sig_fc, self).__init__()
            self.fc = fc
            self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.sig(self.fc(x))
            return x

    model_ft.fc = Sig_fc(nn.Linear(num_ftrs, NUM_OF_CLASS))


    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # Train and evaluate

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,10,counter,num_epochs=25)

