import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import copy
import os
from objects import Object
from quaternion_helper import QuaternionHelper
from matplotlib import cm



def ExtractImages(objects_point_clouds,scenario_point_cloud,num_scenario,objects):
    qh = QuaternionHelper()

    # ---
    # 1 - Initialization
    # ---

    # Import point cloud
    original_point_cloud = scenario_point_cloud

    # Import objects point clouds
    objects_point_clouds = objects_point_clouds
    # path = '01_Scenario/' # folder containing objects point cloud
    # for file in os.listdir(path):
    #     if file.endswith('_object.ply'):
    #         object = o3d.io.read_point_cloud(os.path.join(path, file))
    #         objects_point_clouds.append(object)

    # The orientation of the point cloud is wrong
    #  Original transformation matrix
    original_matrix = np.array([
        [ 0.76604444, -0.64278761,  0.        , -0.02995206],
        [-0.21984631, -0.26200263, -0.93969262, -0.06200263],
        [ 0.60402277,  0.71984631, -0.34202014,  1.33987259],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])

    # Extract rotational part
    rotational_part = original_matrix[:3, :3]

    # Invert rotational part (assuming it's a rotation matrix)
    inverse_rotational_part = np.transpose(rotational_part)

    # Extract translation part
    translation_part = -original_matrix[:3, 3]

    # Invert translation part
    inverse_translation_part = np.dot(inverse_rotational_part, translation_part)

    # Construct inverted transformation matrix
    inverted_matrix = np.identity(4)
    inverted_matrix[:3, :3] = inverse_rotational_part
    inverted_matrix[:3, 3] = inverse_translation_part

    for object in  objects_point_clouds:
            object = object.transform(np.linalg.inv(inverted_matrix))
            object.paint_uniform_color([0,0,1])



    # Import image
    scenes_path = 'rgb-scenes-v2/imgs/scene_' + str(num_scenario)
    for file in os.listdir(scenes_path):
        if file.endswith('-color.png'):
            num_img = int(file.split('-')[0])
            scene = cv2.imread(os.path.join(scenes_path, file))
            scene_gui = copy.deepcopy(scene)
            h,w,_ = scene.shape
            
            # Import camera pose
            poses = []
            file = 'Scenes/'+str(num_scenario)+ '.pose'
            # for file in os.listdir('Scenes/'):
                # if file.endswith('.pose'):
                    # poses = open(scenes_path+file,'r')
            for line in open(file,'r'):
                pose = line.split()
                poses.append(pose)

            camera_pose = poses[num_img]
            
            # Camera's orientation
            quat = np.array([float(camera_pose[0]),float(camera_pose[1]),float(camera_pose[2]),float(camera_pose[3])])

            # Camera's localiztion
            t = np.array([float(camera_pose[4]),float(camera_pose[5]),float(camera_pose[6])])

            # Create camera matrix
            matrix = np.zeros((4,4))
            matrix[0:3,0:3] = np.linalg.inv(np.identity(3))
            matrix[0:3,3] = t
            matrix[3,3]  = 1

            # ---
            # 2 - Project Points
            # ---

            # Camera matrix (intrinsic)
            focal_length = 580
            center = [w/2-5, h/2+5]
            camera_matrix = np.array([[focal_length, 0,            center[0]],
                                            [0,            focal_length, center[1]],
                                            [0,            0,            1]])
            
            for obj_id,object in enumerate(objects_point_clouds):
                points = np.zeros((len(object.points),3))
                for idx,point in enumerate(object.points):
                    # Rotate points by camera orientation
                    point = qh.compute_vector_quaternion_rotation(point,quat)
                    x,y,z = point
                    points[idx,:] = [x,y,z]

                # Translate point
                point = np.ones((len(points),4))
                point[:,0:3] = points
                point = np.transpose(point)
                point = np.dot(matrix,point)
                point = np.transpose(point[0:3,:])

                # Project points into 2d 
                points_2d, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), camera_matrix, None)
                points_2d = np.round(points_2d).astype(int)

                # Create image of projected point
                px = []
                py = []
                for point in points_2d:
                    x,y = point[0]
                    px.append(x)
                    py.append(y)

                px_min  = min(px)
                py_min  = min(py)

                img_w = max(px)-min(px)
                img_h = max(py)-min(py)

                center_point = (int(py_min + img_h/2),int(px_min + img_w/2))

                proj_img = np.zeros((img_w+5,img_h+5))

                objects[obj_id].image(object,img_w,img_h,center_point)

                for point in points_2d:
                    x,y = point[0]
                    x = x - px_min
                    y = y - py_min

                    proj_img[x,y] = 255
                proj_img = proj_img.astype(np.uint8)
                
                cv2.imshow('Projected Image' + str(obj_id), proj_img)

                # Project objects on scene image
                for point in points_2d:
                    x,y = point[0]
                    scene_gui[y,x] = [0,0,255]

            # ---
            # 3 - Crop objects 
            # ---
            # create bounding box arround object
            # object hight and width
                
            colormap = cm.tab10(range(0, len(objects)))
            objs_img = []
            for idx in objects:
                color=colormap[idx,0:3]*255
                objects[idx].draw_bb(scene_gui,color)
            # crop image in the area of the bounding box
                right = objects[idx].right+10
                left = objects[idx].left-10
                top = objects[idx].top
                bottom = objects[idx].bottom
                object_img = np.zeros((objects[idx].width+20,objects[idx].hight))
                object_img = scene[left:right,top:bottom]
                objs_img.append(object_img)
                objects[idx].getColor(object_img)


            
            cv2.imshow(str(num_img) + 'Image with projected objects:', scene_gui)    
            cv2.waitKey(0)

            return objs_img,scene_gui,objects


            # only one scene
            # exit(0)