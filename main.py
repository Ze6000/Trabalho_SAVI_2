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
    
    
    # Lable objects
    lables = ['laranja', 'azeite', 'bola','garrafa','lata'] # result from classification - list of lables in order of objects
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
