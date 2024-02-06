#!/usr/bin/env python3


import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate
import copy
import random
from extract_table import Extract_table
from separate_objects import SeparateObjects
from extract_images import ExtractImages
import cv2
from objects import Object

import os
import sys
sys.path.append('../Training') 
import glob
from dataset import Dataset
import torch
from torchvision import transforms
from model import Model
import torch.nn.functional as F


def main():

    # 1 - Get Point Cloud

    # Scenario number (01 to 14) - Manually
    num_scenario = '01'

    # Get scenario point cloud
    scenario_path = 'Scenes/' + num_scenario + '.ply'
    scenario_point_cloud = o3d.io.read_point_cloud(scenario_path)

    # Extract Table and Objects
    table_point_cloud = Extract_table(scenario_point_cloud)

    # Separate Objects
    objects_point_clouds,objects = SeparateObjects(table_point_cloud)

    # Localizate objects in image
    obj_images,proj_scene,objects = ExtractImages(objects_point_clouds,scenario_point_cloud,num_scenario,objects)

    # Classify objects
    
    
    
    #Create Model

    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = 'Images'                   
    test_filenames = glob.glob(os.path.join(data_path, '*.*'))
    test_dataset = Dataset(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=(len(test_filenames)), shuffle=True)

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Por enquanto tudo bem')

    # Load the trained model
    checkpoint = torch.load('../Training/models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)


    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(labels_gt_np)

    # Take probabilities and find the predict label
    predict_label = [sublist.index(max(sublist)) for sublist in predicted_probabilities]
    # print(predict_label)
    # print(len(predict_label))
    label_dict = Dataset.label_dict
    print(predict_label)
    print(label_dict)
    
    
    # Lable objects
    lables = [label_dict[i] for i in predict_label] # result from classification - list of lables in order of objects
    
    for obj_idx,lable in enumerate(lables):
        objects[obj_idx].lableling(lable,proj_scene)
        print(str(objects[obj_idx].real_h) + ' cm' )

    for obj_idx,obj_img in enumerate(obj_images):
        cv2.imshow('Object ' + str(obj_idx),obj_img)

    cv2.imshow('Scene',proj_scene)
    cv2.waitKey(0)


    # Audio


if __name__ == "__main__":
    main()
