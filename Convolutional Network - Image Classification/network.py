#loading data from images and csv
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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

torch.set_default_tensor_type('torch.cuda.FloatTensor')

BIAS = -0.1
######### Network Definition ##########
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv1.bias.data.fill_(BIAS)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2.bias.data.fill_(BIAS)
        self.conv3 = nn.Conv2d(16, 24, 5)
        self.conv3_bn = nn.BatchNorm2d(24)
        self.conv3.bias.data.fill_(BIAS)
        self.conv4 = nn.Conv2d(24, 40, 5)
        self.conv4_bn = nn.BatchNorm2d(40)
        self.conv4.bias.data.fill_(BIAS)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(40 * 5 * 5, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 14)
        self.fc3_bn = nn.BatchNorm1d(14)
        self.sig = nn.Sigmoid()
        
        # weight initialization
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), (2, 2))
        self.activation_conv1 = x
        x = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(x))), (2, 2))
        self.activation_conv2 = x
        x = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(x))), (4, 4))
        self.activation_conv3 = x
        x = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(x))), (2, 2))
        self.activation_conv4 = x
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
labels = Variable(torch.from_numpy(np.asarray(label_list)).cuda())

#images = torch.load("/datasets/tmp/gray_scaled_images/00000001_000.png")
img_path = "/datasets/tmp/images/"

num_input = 500 #112120
image_list = []
#map each Tensor to a label
for i in range(0, num_input):
    curr_img_name = str(labels_dataframe["Image Index"][i])
    readin = torch.load(""+img_path+curr_img_name).numpy()
    image_list.append(readin)

images = Variable(torch.from_numpy(np.asarray(image_list)).cuda())

########## Training #########

N_FOLD_CROSSVALID = 10
NUM_OF_VALID = num_input / N_FOLD_CROSSVALID
print ("NUM_OF_VALID is %d" % (NUM_OF_VALID))

res_net = Net()
res_net.cuda()
net_id = -1
f_loss = 1

for counter in range(0,N_FOLD_CROSSVALID):
    net = Net()
    net.cuda()
    MAX_EPOCH = 20 
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
        training = torch.cat((images[0:start_idx,:,:,:], \
                             images[end_idx:num_input,:,:,:]),0)
        teaching = torch.cat((labels[0:start_idx,:], \
                             labels[end_idx:num_input,:]),0).float()
    elif start_idx==0:
        training = images[end_idx:num_input,:,:,:]
        teaching = labels[end_idx:num_input,:].float()
    elif end_idx==num_input:
        training = images[0:start_idx,:,:,:]
        teaching = labels[0:start_idx,:].float()
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        
        running_loss = 0
        ### LOSS FUNCTION
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(training)

        loss = criterion(outputs, teaching)
        loss.backward()
        optimizer.step()

        # print loss and accuracy statistics
        #loss
        running_loss += loss.data[0]
        print('[net %d epoch %d] loss: %.3f' %
              (counter, epoch + 1, running_loss))
        running_loss = 0.0

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
        
        
        #calculate accuracy for training set
        flag = 0
        for row in range(start_idx, end_idx):
            flag = 0
            for col in range (0,14):
                if(int(teaching.data[row][col]) == int(prediction.data[row][col])):
                    flag = flag + 1
            correct = float(flag/14)
        #print ('Correct classifications: %f' %(correct))

    v_outputs = net(validation_input)
    n_loss = criterion(v_outputs,validation_output).data[0]
    #print ("net %d loss on validation set is %f" % (counter,n_loss))
    if n_loss < f_loss:
        f_loss = n_loss
        net_id = counter
        res_net = net


    ##accuracy on test data
    test_mean_tensor = torch.sum(v_outputs,1)
    test_mean_tensor = (torch.div(test_mean_tensor,14.0))
    test_prediction = v_outputs


    for row in range(0,len(v_outputs)):
        for col in range (0,14):
            if(v_outputs.data[row][col] < test_mean_tensor.data[row]):
                test_prediction.data[row][col] = 0
            else:
                test_prediction.data[row][col] = 1

    test_correct = 0
    for row in range(0, len(v_outputs)):
        flag = 0
        for col in range (0,14):
            if(int(labels.data[row][col]) == int(test_prediction.data[row][col])):
                flag = flag + 1
        test_correct = float(flag/14)
        print ('test classifications: %f' %(test_correct))

print('Finished Training')
print('Final loss is %f, net_id is %d' % (f_loss,net_id))


### Visualization ###

#First layer
w1_tensor = res_net.conv1.weight.data.cpu()
# print(w1_tensor[0].shape)
trans = transforms.ToPILImage()
im11 = trans(w1_tensor[0])
plt.figure()
plt.imshow(im11)
im12 = trans(w1_tensor[1])
plt.figure()
plt.imshow(im12)
im13 = trans(w1_tensor[2])
plt.figure()
plt.imshow(im13)
im14 = trans(w1_tensor[3])
plt.figure()
plt.imshow(im14)
im15 = trans(w1_tensor[4])
plt.figure()
plt.imshow(im15)
im16 = trans(w1_tensor[5])
plt.figure()
plt.imshow(im16)

w2_tensor = res_net.conv2.weight.data.cpu()
# print(w2_tensor.shape)
size = len(w2_tensor[0][0])
img = torch.Tensor(3,size,size).zero_().cpu()
img[0] = w2_tensor[0][0]
img[1] = w2_tensor[0][0]
img[2] = w2_tensor[0][0]
im21 = trans(img)
plt.figure()
plt.imshow(im21)
img = torch.Tensor(3,size,size).zero_().cpu()
img[0] = w2_tensor[1][0]
img[1] = w2_tensor[1][0]
img[2] = w2_tensor[1][0]
im22 = trans(img)
plt.figure()
plt.imshow(im22)

orig = images[0].data.cpu()
orig_img = trans(orig)
plt.figure()
plt.imshow(orig_img)

conv1_activ = res_net.activation_conv1.data.cpu()
size = len(conv1_activ[0][0])
img = torch.Tensor(3,size,size).zero_().cpu()
for i in range(0,6):
    img[0] = conv1_activ[0][i]
    img[1] = conv1_activ[0][i]
    img[2] = conv1_activ[0][i]
    conv1_activ_img = trans(img)
    plt.figure()
    plt.imshow(conv1_activ_img)

conv2_activ = res_net.activation_conv2.data.cpu()
size = len(conv2_activ[0][0])
img = torch.Tensor(3,size,size).zero_().cpu()
for i in range(0,6):
    img[0] = conv2_activ[0][i]
    img[1] = conv2_activ[0][i]
    img[2] = conv2_activ[0][i]
    conv2_activ_img = trans(img)
    plt.figure()
    plt.imshow(conv2_activ_img)
    
conv3_activ = res_net.activation_conv3.data.cpu()
size = len(conv3_activ[0][0])
img = torch.Tensor(3,size,size).zero_().cpu()
for i in range(0,6):
    img[0] = conv3_activ[0][i]
    img[1] = conv3_activ[0][i]
    img[2] = conv3_activ[0][i]
    conv3_activ_img = trans(img)
    plt.figure()
    plt.imshow(conv3_activ_img)
