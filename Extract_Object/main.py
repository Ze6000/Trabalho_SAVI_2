#!/usr/bin/env python3

import json
import open3d as o3d
import copy
from extract_table import Extract_table
from separate_objects import SeparateObjects
from extract_images import ExtractImages
import cv2
from objects import Object
from datasett import Datasett
import matplotlib.pyplot as plt
import numpy as np


import os
import sys
sys.path.append('../Training') 
import glob
import torch
from torchvision import transforms
from model import Model
from dataset import Dataset
import torch.nn.functional as F
from gtts import gTTS 


def main():
    isRunning =  True
    while isRunning is True:

        # 1 - Get Point Cloud
        
    
        print(''' 
        #                    _                                                             # 
        #   __      __     _| |_                                                           #  
        #   \ \    / /__ _(_)  _|                                                          #  
        #    \ \/\/ / _ ` | | |                                                            #  
        #     \    / (_|  | | |_                                                           #  
        #      \/\/ \___,_|_|___|                                                          #   
        #                                                                         ______   #  
        #   _____ _   __           _ _     ________                 _            / ___  \  #   
        #  |_   _| |_ \/          | | |   /  ______|               | |_         /_/   | |  #   
        #    | | |  _|___    ___ _| | |  |  /  _______  ___ _ _ ___| (_) ____        / /   #  
        #    | | | | / __|  / _ ` | | |  | |  |__   __|  _ ` | `___| | |´ ___|      | |    #  
        #   _| |_| |_\__ \ | (_|  | | |  |  \____| |  | (_|  | |   | | | (___       |_|    #  
        #  |_____|___|___/  \___,_|_|_|   \________/   \___,_|_|   |_|_|_____|      (_)    #  
            
            ''')

        # Scenario number (01 to 14) - Manually
        print('Select scenario (01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14): ')
        num_scenario = input()

        # Get scenario point cloud
        scenario_path = 'Scenes/' + num_scenario + '.ply'
        scenario_point_cloud = o3d.io.read_point_cloud(scenario_path)



        # Extract Table and Objects
        print('Extracting Table and Objects ...')
        table_point_cloud,point_cloud = Extract_table(scenario_point_cloud)
        point_cloud.paint_uniform_color([0.3,0.3,0.3])

        # Separate Objects
        print('Separating objects...')
        objects_point_clouds,objects = SeparateObjects(table_point_cloud)
        objs_pc = copy.deepcopy(objects_point_clouds)

        # Localizate objects in image
        print('Finding objects in image...')
        obj_images,proj_scene,objects = ExtractImages(objects_point_clouds,scenario_point_cloud,num_scenario,objects)

        # Classify objects
        print('Calssifying objects...')
        
        
        #Create Model

        model = Model()

        # -----------------------------------------------------------------
        # Prepare Datasets
        # -----------------------------------------------------------------
        data_path = 'Images'                   
        test_filenames = glob.glob(os.path.join(data_path, '*.*'))
        test_dataset = Datasett(test_filenames)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=(len(test_filenames)), shuffle=False)

        # -----------------------------------------------------------------
        # Prediction
        # -----------------------------------------------------------------

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
        

        with open('../Split_dataset/dataset_filenames.json', 'r') as f:
            # Reading from json file
            dataset_filenames = json.load(f)

        train_filenames = dataset_filenames['train_filenames']
        dataset_instance = Dataset(train_filenames)
        label_dict_list = dataset_instance.label_dict



        # print(label_dict_list)
    
        
        # Lable objects
        lables = [label_dict_list[i] for i in predict_label] # result from classification - list of lables in order of objects


        for obj_idx,_ in enumerate(lables):
            objects[obj_idx].lableling(lables[obj_idx],proj_scene)

        # -----------------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------------
        
        # Show each object
        for idx,obj in enumerate(obj_images):
            cv2.imshow('Object ' + str(idx+1) + ':' + objects[idx].lable,obj)


        cv2.imshow('Scene' + str(num_scenario),proj_scene)
        cv2.waitKey(0)

        point_clouds = [point_cloud]
        point_clouds.extend(objs_pc)
        o3d.visualization.draw_geometries(point_clouds)

        # ----
        # Scenario description 
        # ----
        print('--------')
        intro = 'This scenario as:'
        print(intro)
        speech = intro

        for idx,_ in enumerate(objects):
            obj_descript = ' a ' + objects[idx].color_name + ' ' + objects[idx].lable + ' with ' + str(objects[idx].real_w_x) + ' x ' + str(objects[idx].real_w_y) + ' x ' + str(objects[idx].real_h) + ' cm'
            print(obj_descript)
            speech = speech + obj_descript
        
        print(' THE END ')

        # Audio

        # Language in which you want to convert 
        language = 'en'
        
        # have a high speed 
        myobj = gTTS(text=speech, lang=language, slow=False) 
        
        # Saving the converted audio in a mp3 file named 
        # welcome  
        myobj.save('description.mp3') 

        # Specify the full path to the MP3 file
        speech_file = 'description.mp3'
        os.system('ffplay -v 0 -nodisp -autoexit ' + speech_file)

        print('Do you wish to see another scenario? (y/n)')
        ans = input()
        if ans == 'y':
            isRunning = True
        else:
            isRunning = False


if __name__ == "__main__":
    main()
