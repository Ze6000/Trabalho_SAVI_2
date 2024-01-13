import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate
import copy

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


def main():


    # scenes_idx = ['01', '02','03','04','05','06','07','08','09','10','11','12','13','14']
    # for scene_idx in range(0,13): 
    #     # --------------------------------------
    #     # Initialization
    #     # --------------------------------------
    #     scene_name = scenes_idx[scene_idx]
    #     point_cloud = o3d.io.read_point_cloud('Scenes/' + scene_name + '.ply')
    # o3d.io.write_point_cloud("scene_01.pcd", point_cloud)
        
        point_cloud = o3d.io.read_point_cloud('Scenes/09.ply')
        



    # print(point_cloud)

    
    

        # --------------------------------------
        # Downsampling
        # --------------------------------------
        point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.05)


        # # o3d.io.write_point_cloud('factory_no_floor.ply',point_cloud_outliers)

        

        # -----
        # Execution
        # -----
        # Estimate normals
        point_cloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        # Orientate normals
        point_cloud_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))

        point_cloud_downsampled.paint_uniform_color([0.2,0.2,0.2])

        # Create trnaformation T1 only with rotation
        T1 = np.zeros((4,4),dtype=float)

        # Add homogeneous coordinates
        T1[3, 3] = 1

        # Add null rotation
        R = point_cloud_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
        T1[0:3, 0:3] = R
        # T[0:3, 0] = [1, 0, 0]  # add n vector
        # T[0:3, 1] = [0, 1, 0]  # add s vector
        # T[0:3, 2] = [0, 0, 1]  # add a vector

        # Add a translation
        T1[0:3, 3] = [0, 0, 0]
        # print('T1=\n' + str(T1))

        # Create transformation T2 only with translation
        T2 = np.zeros((4, 4), dtype=float)

        # Add homogeneous coordinates
        T2[3, 3] = 1

        # Add null rotation
        T2[0:3, 0] = [1, 0, 0]  # add n vector
        T2[0:3, 1] = [0, 1, 0]  # add s vector
        T2[0:3, 2] = [0, 0, 1]  # add a vector

        # Add a translation
        T2[0:3, 3] = [0.8, 1, -0.4]
        # print('T2=\n' + str(T2))

        T = np.dot(T1, T2)
        # print('T=\n' + str(T))


        # Create table ref system and apply trnafoormation to it
        frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))
        frame_table = frame_table.transform(T)
        point_cloud_downsampled = point_cloud_downsampled.transform(np.linalg.inv(T))
        point_cloud = point_cloud.transform(np.linalg.inv(T))



        ## --- Find table plane
        # Compare normals to vector [0,0,1] to find points belonging to horizontal planes
        # create point cloud with "horizontal" points
        point_cloud_horizontal = o3d.geometry.PointCloud()
        for point, normal in zip(point_cloud_downsampled.points, 
                                        point_cloud_downsampled.normals):
            

            # compute angle between 2 3d vectors
            # find norm of the normal vector
            norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

            # find norm of z axis
            z_axis = [0,0,1]
            norm_z_axis = math.sqrt(z_axis[0]**2 + z_axis[1]**2 + z_axis[2]**2)

            # get angle betwwen the normal vector and z axis
            theta =  math.acos(np.dot(normal, z_axis)/norm_normal*norm_z_axis)
            # print('Vector: ' + str(normal))
            print('Theta: ' + str(theta))
            delta= theta*180/math.pi
            print('Delta: ' + str(delta))
    
            # keep points where angle to z_axis is small enough
            if delta < 20: # we have a point that belongs to an horizonta surface
                point_cloud_horizontal.points.append(point)
                point_cloud_horizontal.points.append(normal)
                color = [0,0,1]
                point_cloud_horizontal.points.append(color)

        point_cloud_horizontal.paint_uniform_color([0,0,1])


        

        


        # --------------------------------------
        # Clustering
        # --------------------------------------
        labels =   point_cloud_horizontal.cluster_dbscan(eps=0.2,
                                                        min_points=40,
                                                        print_progress=True)
        
        groups_idxs = list(set(labels)) # gives a list of the values in the labels list
        groups_idxs.remove(-1) # remove last group because is the group of unassigned points
        num_groups = len(groups_idxs)
        colormap = cm.Pastel1(range(0, num_groups))
        group_point_clouds = []
        for group_idx in groups_idxs:

            group_points_idxs = list(locate(labels, lambda x: x==group_idx))
            group_point_cloud = point_cloud_horizontal.select_by_index(group_points_idxs, invert = False)
            color = colormap[group_idx, 0:3]
            group_point_cloud.paint_uniform_color(color)
            group_point_clouds.append(group_point_cloud)

        print(group_point_clouds)


        groups_width = []
        groups_center = o3d.geometry.PointCloud()
        table_group = None
        for i in groups_idxs:
                x = []
                y = []
                z = []


                for row in range(1,len(group_point_clouds[i].points)):
                    x.append(group_point_clouds[i].points[row][0])
                    y.append(group_point_clouds[i].points[row][1])
                    z.append(group_point_clouds[i].points[row][2])

                x.sort()
                y.sort()
                z.sort()

                max_x = x[-1]
                max_y = y[-1]
                max_z = z[-1]

                min_x = x[0]
                min_y = y[0]
                min_z = z[0]

                width_x = abs(max_x-min_x)
                width_y = abs(max_y-min_y)
                width_z = abs(max_z-min_z)

                x_center = min_x + width_x/2
                y_center = min_y + width_y/2
                z_center = min_z + width_z/2

                point = np.array([x_center, y_center, z_center])

                groups_width.append(np.array([width_x, width_y]))
                groups_center.points.append(point)

    
        # find floor center point
        floor_point = None
        for idx in range(0,len(groups_center.points)):
             point = groups_center.points[idx]

             if floor_point is None:
                  floor_point = point
             elif point[2] < floor_point[2]:
                  floor_point = point
 
        table_center = o3d.geometry.PointCloud()
        for idx in range(0,len(groups_center.points)):
             point = groups_center.points[idx]
             x = point[0]
             y = point[1]
             z = point[2]
             
             if floor_point[0]-1 < x < floor_point[0]+1 and floor_point[1]-1< y <floor_point[1]+1.5 and z < floor_point[2] + 1:
                  if floor_point[0] != x and floor_point[1] != y and floor_point[2] != z: 
                     table_center.points.append(point)

        table_center.paint_uniform_color([0,0,1])
        print(table_center)
   
        # print(groups_width)
        # table_group.paint_uniform_color([0,1,0])
        table_center.paint_uniform_color([1,0,0])


        # Ex3 - Create crop the points in the table
        # Create a vector3d with the points in the boundingbox
        np_vertices = np.ndarray((8, 3), dtype=float)

        sx = table_center.points[0][0] + 0.7
        sy = table_center.points[0][1] + 0.7
        sz_top = table_center.points[0][2] + 0.5
        sz_bottom = table_center.points[0][2] - 0.1
        np_vertices[0, 0:3] = [sx, sy, sz_top]
        np_vertices[1, 0:3] = [sx, -sy, sz_top]
        np_vertices[2, 0:3] = [-sx, -sy, sz_top]
        np_vertices[3, 0:3] = [-sx, sy, sz_top]
        np_vertices[4, 0:3] = [sx, sy, sz_bottom]
        np_vertices[5, 0:3] = [sx, -sy, sz_bottom]
        np_vertices[6, 0:3] = [-sx, -sy, sz_bottom]
        np_vertices[7, 0:3] = [-sx, sy, sz_bottom]

        # print('np_vertices =\n' + str(np_vertices))

        vertices = o3d.utility.Vector3dVector(np_vertices)

        # Create a bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
        # print(bbox)


        # Crop the original point cloud using the bounding box
        pcd_cropped = point_cloud.crop(bbox)

        pcd_cropped.paint_uniform_color([1,0,0])
        o3d.io.write_point_cloud('table_objects.ply',pcd_cropped)




        # -----
        # Visualization
        # -----
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))


        entities = [point_cloud_downsampled]
        entities.append(pcd_cropped) 
        entities.append(table_center)

        o3d.visualization.draw_geometries(entities,
                                        zoom=view['trajectory'][0]['zoom'],
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'],
                                        point_show_normal = False)
        


 

if __name__ == "__main__":
    main()