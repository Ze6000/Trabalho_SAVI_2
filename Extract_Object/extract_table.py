import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate


view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.65383729991435868, -0.56780420046251223, 0.5000951661212375 ],
			"lookat" : [ 2.6172, 2.0474999999999999, 1.532 ],
			"up" : [ -0.45904798451753814, 0.22772577357992463, 0.85872924717736909 ],
			"zoom" : 0.52120000000000011
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

class PlaneSegmentation():
    def __init__(self, input_point_cloud):
        self.input_point_cloud = input_point_cloud
        self.a = None
        self.b = None
        self.c = None
        self.d = None

    def findPlane(self, distance_threshold=0.03, ransac_n=3, num_iterations=100):
        plane_model, inlier_idxs = self.input_point_cloud.segment_plane(distance_threshold = distance_threshold,
                                                                        ransac_n=ransac_n, 
                                                                        num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model
        self.inliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = False)
        self.outliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = True)


def Extract_table(scenario_point_cloud):

    # Define Scenario point cloud
    point_cloud = scenario_point_cloud

    # --------------------------------------
    # Downsampling
    # --------------------------------------
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.05)


    # -----
    # Execution
    # -----
    # Estimate normals
    point_cloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    # Orientate normals
    point_cloud_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))

    point_cloud_downsampled.paint_uniform_color([0.2,0.2,0.2])

    # Rotate frame and tr

    # Create trnaformation T1 only with rotation
    T1 = np.zeros((4,4),dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = point_cloud_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R

    # Add a translation
    T1[0:3, 3] = [0, 0, 0]

    # Create transformation T2 only with translation
    T2 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation
    T2[0:3, 0:3] = np.identity(3) 

    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.4]

    T = np.dot(T1, T2)

    # Create table ref system and apply trnafoormation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
    frame_table = frame_table.transform(T)
    point_cloud_downsampled = point_cloud_downsampled.transform(np.linalg.inv(T))
    point_cloud = point_cloud.transform(np.linalg.inv(T))

    # ---
    # Find Table Plane
    # ---

    # 1 - Find horizontal planes

    point_cloud_horizontal = o3d.geometry.PointCloud()

    # Calculate normal for each point
    for point, normal in zip(point_cloud_downsampled.points, 
                                    point_cloud_downsampled.normals):   
        # Normal norm
        norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        # Z axis norm (1)
        z_axis = [0,0,1]
        norm_z_axis = math.sqrt(z_axis[0]**2 + z_axis[1]**2 + z_axis[2]**2)

        #  Angle between normal and z axis
        theta =  math.acos(np.dot(normal, z_axis)/norm_normal*norm_z_axis)
        delta= theta*180/math.pi

        # Keep points where angle to z_axis is small enough
        if delta < 25: # we have a point that belongs to an horizontal surface
            point_cloud_horizontal.points.append(point)
            point_cloud_horizontal.points.append(normal)
            color = [0,0,1]
            point_cloud_horizontal.points.append(color)

    point_cloud_horizontal.paint_uniform_color([0,0,1])

    # Cluster points of horizontal planes to separate them
    labels =   point_cloud_horizontal.cluster_dbscan(eps=0.2,
                                                    min_points=50,
                                                    print_progress=False)
    
    groups_idxs = list(set(labels))
    groups_idxs.remove(-1) 
    num_groups = len(groups_idxs)
    colormap = cm.Pastel1(range(0, num_groups))
    group_point_clouds = []

    for group_idx in groups_idxs:

        group_points_idxs = list(locate(labels, lambda x: x==group_idx))
        group_point_cloud = point_cloud_horizontal.select_by_index(group_points_idxs, invert = False)
        color = colormap[group_idx, 0:3]
        group_point_cloud.paint_uniform_color(color)
        group_point_clouds.append(group_point_cloud)


    # 2 - Find floor plane (reference to find table)
    minimum_z = None
    groups_center = o3d.geometry.PointCloud()

    # For each horizontal plane calculate center point
    # Floor plane it will be the lower horizontal plane
    for group in group_point_clouds:
        points = group.points
        length = len(points) 

        if length < 100:
            continue

        x = []
        y = []
        z = []
        for point in points:
            a,b,c = point
            x.append(a)
            y.append(b)
            z.append(c)
        
        x_max = max(x)
        x_min = min(x)

        y_max = max(y)
        y_min = min(y)

        z_max = max(z)
        z_min = min(z)

        width_x = abs(x_max-x_min)
        width_y = abs(y_max-y_min)
        width_z = abs(z_max-z_min)

        x_center = x_min + width_x/2
        y_center = y_min + width_y/2
        z_center = z_min + width_z/2

        point = np.array([x_center, y_center, z_center])
        groups_center.points.append(point)

        if minimum_z is None:
            floor_group = group
            floor_center = point
            minimum_z = z_center
        elif z_center < minimum_z:
            floor_group = group
            floor_center = point
            minimum_z = z_center


    floor_group.paint_uniform_color([0,0,1])
    groups_center.paint_uniform_color([0,1,0])

    # 3 - Find the horizontal plane nearest to the floor center (because table is in the center)
    # we know that the table center is at least 20 cm above the floor, and is not higher than 1 m

    # Calculate distance between the floor center and the table center
    min_dist = None
    for idx,center in enumerate(groups_center.points):
        x,y,z = center
        fx,fy,fz = floor_center
        dist = math.sqrt((fx-x)**2 + (fy-y)**2)
        if fx-1.5 < x < fx+1.5 and fy-1.5 < y < fy+1.5 and fz+0.2 < z < fz+1:
            if fx != x and fy != y and fz != z:
                if min_dist is None:
                    table_center=center
                    table_group = group_point_clouds[idx] 
                    min_dist = dist
                elif dist < min_dist:
                    table_center=center
                    table_group = group_point_clouds[idx] 
                    min_dist = dist

    # ---
    # Extract the points of from the table and the objects above it 
    # ---
    
    # Create a bounding box from the center of the table
    # box dimensions: 1.8 x 1.8 x 0.55 m
    np_vertices = np.ndarray((8, 3), dtype=float)
    x,y,z = table_center 
    max_sx =  x + 0.6
    max_sy = y + 0.6
    min_sx =  x - 0.6
    min_sy = y - 0.6
    sz_top = z + 0.5
    sz_bottom = z - 0.05
    np_vertices[0, 0:3] = [max_sx, max_sy, sz_top]
    np_vertices[1, 0:3] = [max_sx, min_sy, sz_top]
    np_vertices[2, 0:3] = [min_sx, min_sy, sz_top]
    np_vertices[3, 0:3] = [min_sx, max_sy, sz_top]
    np_vertices[4, 0:3] = [max_sx, max_sy, sz_bottom]
    np_vertices[5, 0:3] = [max_sx, min_sy, sz_bottom]
    np_vertices[6, 0:3] = [min_sx, min_sy, sz_bottom]
    np_vertices[7, 0:3] = [min_sx, max_sy, sz_bottom]


    vertices = o3d.utility.Vector3dVector(np_vertices)

    # Create a bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    # Crop the original point cloud using the bounding box
    pcd_cropped = point_cloud.crop(bbox)

    return pcd_cropped,point_cloud

    