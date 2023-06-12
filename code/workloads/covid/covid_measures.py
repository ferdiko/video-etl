from __future__ import print_function
from __future__ import division

import cv2
# import sys
import time
import cvlib as cv
# import numpy as np
# import os
# import subprocess
import random
# # import argparse
# # from cvlib.object_detection import draw_bbox
#
# # from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# import torch
# from torch.autograd import Variable
# from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# from torch.optim import Adam, SGD

import sys
sys.path.append("../../src/offline")
from execution_utils import TaskGraph
import torch
from torch import optim, nn
from torchvision import models, transforms


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy

# from workload import Workload

from PIL import Image
from calibrate_camera import dlt
'''
Description:
 - COVID mask and social distancing
Knobs:
 - Frame rate mask, framerate distance, models?
 -
'''

class MaskClassifier(nn.Module):
    # TODO: Also support squeeze net

    def __init__(self, train_backbone=False):
        super(MaskClassifier, self).__init__()
        input_size = 224
        self.trans = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # self.trans = transforms.ToTensor()
        self.model = models.vgg11(pretrained=True) #vgg16()

        # don't train the backbone layers (e.g. vgg)
        if not train_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # replace last layer by our classifier layer
        num_ftrs = self.model.classifier[6].in_features
        # TODO: could use 1 but how does vgg do a softmax or softmax or so
        self.model.classifier[6] = nn.Linear(num_ftrs, 2)

        print("bef")
        self.model.load_state_dict(torch.load("covidmodel.ckp"), strict=False)
        print("after")

    def forward(self, x):
        print("here")
        x = Image.fromarray(np.uint8(x))
        x = self.trans(x)
        x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
        x = self.model(x)
        return torch.argmax(x).item()
        # return self.model(x)


class CovidWorkload():

    def __init__(self, cuda=False):
        # knobs exposed to KnobTuner
        self.knob_names = ["framerate_mask", "framerate_distance"]
        self.knob_types = ["int"]*2
        self.knobs = [1, 1]
        self.knob_domains = [[i*5 for i in range(1,31)], [i*5 for i in range(1,31)]]
        assert len(self.knob_names) == len(self.knob_types) == len(self.knobs) == len(self.knob_domains)

        # hacky ... normalize graph runtimes through this runtime
        self.norm_const = 850+160

        # covid mask models TODO
        # self.face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # self.mask_classify = MaskClassifier()

        # distancing
        # self.yolo = cv2.dnn.readNet('yolov5s.onnx')
        # if cuda:
        #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        # else:
        #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.P, _ = dlt()


    def get_taskgraph(self, num_frames=150):
        graph = TaskGraph()

        for i in range(num_frames):
            # insert mask detection tasks
            if i % self.knobs[0] == 0:
                dep = graph.num_nodes
                graph.insert(850, 3000, []) # face detection. TODO: scale with detects, smaller faces etc?
                graph.insert(320, 1300, [dep])
                graph.insert(320, 1300, [dep])
                graph.insert(320, 1300, [dep])
                graph.insert(320, 1300, [dep])
                # graph.insert(4, 4, [dep+1,dep+2,dep+3,dep+4])

            # insert social distance tasks
            if i % self.knobs[1] == 0:
                graph.insert(721, 1941, [])
                # graph.insert(721, 1941, [graph.num_nodes])

        return graph


    def set_knob(self, k):
        self.knobs = k

    def mask_detection(self, frame):
        # print(frame.shape)
        # fr = frame[int(frame.shape[0]/3):, :]
        # print(fr.shape)
        # print(cv2.imencode(".jpg", fr)[1].shape)
        # print("+++++++")

        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        start = time.time()
        for i in range(4):
            faces, neighbours, weights = self.face_detect.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(5, 5),
                flags = cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels = True
            )
        end = time.time()
        print("face det", end-start)

        visualize(frame, faces)

        print("faces", len(faces))

        if len(faces) == 0:
            return 1

        inputs = [cv2.imencode(".jpg", frame[y:y+h, x:x+w])[1] for (x, y, w, h) in faces]

        # for inp in inputs:
        #     print("input jpeg size", inp.shape)

        inputs = [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]
        # for inp in inputs:
        #     print("input size", inp.shape)



        # print("inputs", len(inputs))
        count = 0
        # start = time.time()

        for inp in inputs:
            for i in range(4):
                count += self.mask_classify(inp)

        # end = time.time()
        # print("mask time", end-start)

        return count/len(inputs)


# def format_yolov5(source):
#
#     # put the image in square big enough
#     col, row, _ = source.shape
#     _max = max(col, row)
#     resized = np.zeros((_max, _max, 3), np.uint8)
#     resized[0:col, 0:row] = source
#
#     # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
#     result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
#
#
#
#     inputImage = format_yolov5(frame)
#     outs = detect(inputImage, net)
#
#     class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
#
#
#     blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
#     net.setInput(blob)
#     preds = net.forward()
#
#     return result


    def measure_social_dist(self, frame):
        # TODO: debug etc.
        # bbox, label, conf = cv.detect_common_objects(frame)
        # visualize(frame, bbox)

        # tracked_obj = ['person']
        # bbox = [b for b, l in zip(bbox, label) if l in tracked_obj]
        # print("num dets", len(bbox))


        # feet_img = np.array([[left+right/2, bottom, 1] for left, right, bottom, _ in bb_box])
        # feet_world = np.matmul(feet_img, self.P)
        # # TODO: normalize homogenous coordinates
        # for i in range(feet_img.shape[0]):
        #     for j in range(i+1, feet_img.shape[0])):
        #         dist = np.linalg.norm(feet_world - feet_world)

        return 0


    def process(self, file='correct/00000_Mask.jpg', visualize=False):

        # TODO: Delete; this is here for faster debugging
        # TODO: Use self.knob
        runtime_total = random.uniform(1200, 14000)
        acc = random.uniform(0.5, 0.8)

        return acc, runtime_total




        total_mask_score = 0
        total_dist_score = 0

        video = cv2.VideoCapture(file)
        ok, frame = video.read()

        frame_num = 0
        while ok:
            print("ok")
            # mask detection
            if frame_num % self.knobs[0] == 0:
                # TODO: average
                total_mask_score += self.mask_detection(frame)

            # social distance
            if frame_num % self.knobs[1] == 0:
                # TODO: count seconds (using framerate)
                total_dist_score += self.measure_social_dist(frame)

            frame_num += 1
            ok, frame = video.read()

        video.release()

        return total_mask_score + total_dist_score


def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "covidmodel.ckp")
    return model, val_acc_history



def train(data_dir=""):
    # get model
    workload = CovidWorkload()
    model_ft = workload.mask_classify

    input_size = 224
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "vgg"

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True



    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


def visualize(img, boxes):
    # Draw rectangle around the faces
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite('now.png', img)


import numpy as np
if __name__ == '__main__' :

    # train()



    workload = CovidWorkload()

    workload.get_taskgraph()
    workload.process()
    # classifier = MaskClassifier()


    #
    # # Load the cascade
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # # Read the input image
    # img = cv2.imread('test.jpg')
    # print(img.shape)
    # start = time.time()
    # for i in range(1):
    #     # Convert into grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     print(type(gray))
    #     # Detect faces
    #     # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #     faces, neighbours, weights = face_cascade.detectMultiScale3(
    #         gray,
    #         scaleFactor=1.1,
    #         minNeighbors=5,
    #         minSize=(10, 10),
    #         flags = cv2.CASCADE_SCALE_IMAGE,
    #         outputRejectLevels = True
    #     )
    #
    #     # inputs = [cv2.imencode(".jpg", img[y:y+h, x:x+w])[1] for (x, y, w, h) in faces]
    #     # inputs = [img[y:y+h, x:x+w] for (x, y, w, h) in faces]
    #
    #     for (x, y, w, h) in faces:
    #         face = img[y:y+h, x:x+w]
    #         # TODO: ensure dtype = np.uint8
    #         face_tensor = transforms.ToTensor()(face)
    #
    #
    #         print(type(face))
    #         print(face.shape)
    #         print(np.array(cv2.imencode(".jpg", img[y:y+h, x:x+w])[1]).shape)
    #
    #         x_data = torch.tensor(face)
    #         print(x_data)
    #
    #     print(inputs[0].data)
    #
    #     # transforms.ToTensor()
    #
    # end = time.time()
    #
    # print(end - start)

    # print(faces, neighbours, weights)
    #
    # # rects = faces[0]
    # # neighbours = faces[1]
    # # weights = faces[2]
    # print("faces", len(faces))
    #
    # # Draw rectangle around the faces
    # for i, (x, y, w, h) in enumerate(faces):
    #     face = img[y:y+h, x:x+w]
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     cv2.imwrite("face{}.png".format(i), face)
    # #
    # # start = time.time()
    # #
    # #
    # #
    # #
    # # end = time.time()
    #
    #
    # #
    # # bbox_slice, label_slice, conf_slice = cv.detect_common_objects(img)
    # # print(label_slice)
    # # tracked_obj = ['person']
    # # bbox_slice = [b for b, l in zip(bbox_slice, label_slice) if l in tracked_obj]
    # # print(len(bbox_slice))
    # #
    # # # Convert into grayscale
    # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # # Detect faces
    # # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # # print(len(faces))
    #
    # # Draw rectangle around the faces
    # # for (x, y, w, h) in faces:
    # #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # # Display the output
    # # cv2.imshow('img', img)
