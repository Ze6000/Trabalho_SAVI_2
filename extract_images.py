import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import copy
import os
from objects import Object

def rotate(thing,axis,angle):
    # axis: 0 - x; 1 - y; 2 - z
    angle = angle*math.pi/180 

    if axis == 1:
        # x axis
        rot = np.array([
            [1,0,0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 2:
        # y axis
        rot = np.array([
            [np.cos(angle),0, -np.sin(angle)],
            [0,1,0],
            [np.sin(angle),0, np.cos(angle)]
        ])
    else:
        # z axis
        rot = np.array([
            [np.cos(angle), -np.sin(angle),0],
            [np.sin(angle), np.cos(angle),0],
            [0,0,1]
        ])
    rotation = np.dot(thing,rot)
    return rotation

def transform_point_could(object_point_cloud,r):
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

    # Create ref system and apply trnafoormation to it
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0,0,0]))
    frame.rotate(r,np.array([0,0,0]))
    for object in  object_point_cloud:
        object = object.transform(np.linalg.inv(inverted_matrix))
        object.paint_uniform_color([0,0,1])

    return object_point_cloud, frame



def project_points(matrix,object_point_cloud,center,focal_length):


    camera_matrix = np.array([[focal_length, 0,            center[0]],
                                        [0,            focal_length, center[1]],
                                        [0,            0,            1]])
    pc_points = np.zeros((len(object_point_cloud.points),3))
        
    for idx_p,point in enumerate(object_point_cloud.points):
        x,y,z = point
        pc_points[idx_p,:] = [x,y,z]

    # Transform points
    point = np.ones((len(pc_points),4))
    point[:,0:3] = pc_points
    point = np.transpose(point)
    point = np.dot(matrix,point)
    point = np.transpose(point[0:3,:])



    # Project points
    # points_2d,_ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), focal_length,center,distortion_coeffs)
    points_2d, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), camera_matrix, 0)



    # Scale the points to image pixels
    points_2d = np.round(points_2d).astype(int)
    return points_2d

# ---
# INITIALIZATION
# ---
# TODO Bounding box of each object
# TODO Fetch camera position automatically
# TODO Orientate camera - for 4 different points of view? 

# Import objects point cloud
object_point_cloud = []
for i in range(0,5):
    object_filename = str(i) + '_object.ply'
    object = o3d.io.read_point_cloud(object_filename)
    object_point_cloud.append(object)

# Import scene point cloud
point_cloud = o3d.io.read_point_cloud('Scenes/02.ply')
point_cloud.paint_uniform_color([0.3,0.3,0.3])

# Get all scenes images
scenes_path = 'Scenes'
scenes = os.listdir(scenes_path)

# Get camera pose from that image
poses = []
pose_file = [pose for pose in scenes if pose.endswith('.pose')]
with open(scenes_path + '/'+pose_file[0], 'r') as file:
    for line in file:
        pose = line.split() 

        # Convert numbers from strings to integers or floats
        pose = [float(value) for value in pose] 
        poses.append(pose)
        
scenes = [scene for scene in scenes if scene.endswith('.png')]
scenes = sorted(scenes)

idx_scene = 0

# Import scene image
filename = scenes[idx_scene]
scene = cv2.imread(scenes_path + '/' + filename)
h, w, _ = scene.shape
print(h,w)
scene_gui = copy.deepcopy(scene)

# Get image number
numeric_part = ''.join(filter(str.isdigit, filename))
num_image = int(numeric_part)

# Get camera pose
print(num_image)
camera_pose = poses[num_image]
print(camera_pose)



# Rotation
r = R.from_quat([camera_pose[0], camera_pose[1], camera_pose[2], camera_pose[3]])
r = r.as_matrix()

t = np.array([camera_pose[4],camera_pose[5],camera_pose[6]])

# Transform point cloud to original orientation
object_point_cloud,frame = transform_point_could(object_point_cloud,r)

# # In scene 002 we applied this rotatio:
r=rotate(r,3,90)

# Create camera matrix
matrix = np.zeros((4,4))
matrix[0:3,0:3] = r
matrix[0:3,3] = t
matrix[3,3]  = 1




camera_rotated = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=t)
camera_rotated = camera_rotated.rotate(r,t)


# ---
# PROJECT POINTS
# ---
objects = {}
for idx,object_pc in enumerate(object_point_cloud):


    center = [w/2-80,h/2+80]
    focal_length = 570.0
    iterative = 0
    # while True:
    #         try:
    points_2d = project_points(matrix,object_pc,center,focal_length)
    px = []
    py = []
    for bb in points_2d:
        x,y = bb[0]
        # x = int(x-2*(abs(x-w/2)))
        scene_gui[x,y] = [0,0,255]

        px.append(x)
        py.append(y)

    px_min = min(px)
    px_max = max(px)

    py_min = min(py)
    py_max = max(py)


    for point in points_2d:
        px, py = point[0]
        point[0][0] = px-px_min
        point[0][1] = py-py_min

    hight = py_max-py_min
    width = px_max-px_min
    center_point = (int(px_min + width/2),int(py_min + hight/2))

    object = Object(point_cloud,hight,width,center_point)
    objects[idx]  = object

    projected_image = np.zeros((width+5,hight+5))


    for point in points_2d:
        px, py = point[0]
        projected_image[px,py] = 255

    projected_image=projected_image.astype(np.uint8)
    cv2.imshow('Projected image' + str(idx), projected_image)
                # break
    

            # except IndexError:
            #     # center[0] -= 10
            #     # center[1] += 10
            #     focal_length -= 5
            #     iterative += 1
            #     print(iterative)
            #     print(center)
            #     print(focal_length)


    

# ---
# CROPP IMAGE
# ---

# create bounding box arround object
    # object hight and width
for idx in objects:
    objects[idx].draw_bb(scene_gui)
# crop image in the area of the bounding box
    right = objects[idx].right
    left = objects[idx].left
    top = objects[idx].top
    bottom = objects[idx].bottom
    object_img = np.zeros((objects[idx].width,objects[idx].hight))
    object_img = scene[left:right,top:bottom]
    # print(object_img.shape)
    cv2.imshow('Object' + str(idx), object_img)
    
    

# ---
# VISUALIZATION
# ---
cv2.imshow('Scene', scene)
cv2.imshow('Scene + Projected objects', scene_gui)
cv2.waitKey(0)
entities = [point_cloud]
entities.append(frame)
entities.append(camera_rotated)

entities.extend(object_point_cloud)
# entities.append(camera_point)

o3d.visualization.draw_geometries(entities)




exit(0)


