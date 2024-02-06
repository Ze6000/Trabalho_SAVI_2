import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate
import copy
import random
import os
from extract_table import Extract_table
from separate_objects import SeparateObjects
from extract_images import ExtractImages
import cv2
from objects import Object
# Import the required module for text  
# to speech conversion 
from gtts import gTTS 

  
# This module is imported so that we can  
# play the converted audio 
import os 



def main():

    # 1 - Get Point Cloud

    # Scenario number (01 to 14) - Manually
    # 5,6,7,8,13,14 has an error
    # 9,12 has one more object

    objects = {}
    
    num_scenario = '03'

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
    
    
    # Lable objects
    lables = ['chair', 'door', 'ball','bottle','tin can'] # result from classification - list of lables in order of objects
    for obj_idx,_ in enumerate(objects):
        objects[obj_idx].lableling(lables[obj_idx],proj_scene)
        print(str(objects[obj_idx].real_h) + ' cm' )

    for obj_idx,obj_img in enumerate(obj_images):
        cv2.imshow('Object ' + str(obj_idx),obj_img)

    cv2.imshow('Scene',proj_scene)
    cv2.waitKey(0)


    
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
    # os.system('ffplay -v 0 -nodisp -autoexit ' + speech_file)




if __name__ == "__main__":
    main()
