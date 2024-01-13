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
        
        point_cloud = o3d.io.read_point_cloud('table_objects.ply')

        # --------------------------------------
        # Downsampling
        # --------------------------------------
        point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.02)
        point_cloud_downsampled.paint_uniform_color([0.2,0.2,0.2])

        # Remove table plane
        plane_model, inliers_idx = point_cloud_downsampled.segment_plane(distance_threshold=0.025,
                                         ransac_n=3,
                                         num_iterations=100)
        a,b,c,d = plane_model

        point_cloud_table = point_cloud_downsampled.select_by_index(inliers_idx, invert = False)
        point_cloud_table.paint_uniform_color([1,0,0])

        point_cloud_objects = point_cloud_downsampled.select_by_index(inliers_idx, invert = True)
       
        # --------------------------------------
        # Clustering
        # --------------------------------------
        objects =  point_cloud_objects.cluster_dbscan(eps=0.1,
                                                        min_points=10,
                                                        print_progress=True)
        
        groups_idxs = list(set(objects)) # gives a list of the values in the labels list
        groups_idxs.remove(-1) # remove last group because is the group of unassigned points
        num_groups = len(groups_idxs)
        colormap = cm.Pastel1(range(0, num_groups))
        group_point_clouds = []
        for group_idx in groups_idxs:

            group_points_idxs = list(locate(objects, lambda x: x==group_idx))
            group_point_cloud = point_cloud_objects.select_by_index(group_points_idxs, invert = False)
            
            filename = str(group_idx) + '_object.ply'
            o3d.io.write_point_cloud(filename,group_point_cloud)

            color = colormap[group_idx, 0:3]
            group_point_cloud.paint_uniform_color(color)
            group_point_clouds.append(group_point_cloud)

        # TODO Create a class for the objects 
        # Compute the objscts' properties
        for idx in groups_idxs:
            x_max = None
            x_min = None
            y_max = None
            y_min = None
            z_max = None
            z_min = None

            for p_idx in range(0,len(group_point_clouds[idx].points)):
                 x = group_point_clouds[idx].points[p_idx][0]
                 y = group_point_clouds[idx].points[p_idx][1]
                 z = group_point_clouds[idx].points[p_idx][2]

                 if x_max is None:
                      x_max = x
                 elif x > x_max:
                      x_max = x

                 if x_min is None:
                      x_min = x
                 elif x < x_min:
                      x_min = x        

                 if y_max is None:
                      y_max = y
                 elif y > y_max:
                      y_max = y

                 if y_min is None:
                      y_min = y
                 elif y < y_min:
                      y_min = y

                 if z_max is None:
                      z_max = z
                 elif z > z_max:
                      z_max = z

                 if z_min is None:
                      z_min = z
                 elif z < z_min:
                      z_min = z       

            
            # Hight
            hight = z_max - z_min
                
            # Width
            width_x = x_max-x_min
            width_y = y_max-y_min
            # Color?
            # More ...



        # -----
        # Visualization
        # -----
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

        entities = [point_cloud_downsampled]
        entities.append(point_cloud_table)
        entities.extend(group_point_clouds)


        o3d.visualization.draw_geometries(entities,
                                        zoom=view['trajectory'][0]['zoom'],
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'],
                                        point_show_normal = False)
        


 

if __name__ == "__main__":
    main()