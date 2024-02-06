import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate
from objects import Object

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

    def findPlane(self, distance_threshold=0.1, ransac_n=3, num_iterations=100):
        plane_model, inlier_idxs = self.input_point_cloud.segment_plane(distance_threshold = distance_threshold,
                                                                        ransac_n=ransac_n, 
                                                                        num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model
        self.inliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = False)
        self.outliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = True)


def SeparateObjects(table_point_cloud):
    
    point_cloud = table_point_cloud

    # --------------------------------------
    # Downsampling
    # --------------------------------------


    # Remove table plane
    plane_model, inliers_idx = point_cloud.segment_plane(distance_threshold=0.025,
                                        ransac_n=3,
                                        num_iterations=100)
    a,b,c,d = plane_model

    point_cloud_table = point_cloud.select_by_index(inliers_idx, invert = False)
    point_cloud_table.paint_uniform_color([1,0,0])

    point_cloud_objects = point_cloud.select_by_index(inliers_idx, invert = True)
    
    # --------------------------------------
    # Clustering
    # --------------------------------------
    objects =  point_cloud_objects.cluster_dbscan(eps=0.015,
                                                    min_points=50,
                                                    print_progress=True)
    
    groups_idxs = list(set(objects)) # gives a list of the values in the labels list
    groups_idxs.remove(-1) # remove last group because is the group of unassigned points
    num_groups = len(groups_idxs)
    colormap = cm.Pastel1(range(0, num_groups))
    group_point_clouds = []
    for group_idx in groups_idxs:

        group_points_idxs = list(locate(objects, lambda x: x==group_idx))
        group_point_cloud = point_cloud_objects.select_by_index(group_points_idxs, invert = False)
        
        # filename = scenario + '_Scenario/' + str(group_idx) + '_object.ply'
        # o3d.io.write_point_cloud(filename,group_point_cloud)

        color = colormap[group_idx, 0:3]
        group_point_cloud.paint_uniform_color(color)
        # print(len(group_point_cloud.points))
        if len(group_point_cloud.points) > 5000:
            group_point_clouds.append(group_point_cloud)

    # TODO Create a class for the objects 
    objects = {}
    for idx,obj in enumerate(group_point_clouds):
        x = []
        y = []
        z = []

        for point in obj.points:
            a,b,c = point
            x.append(a)
            y.append(b)
            z.append(c)

        minx = min(x)
        miny = min(y)
        minz = min(z)
        
        maxx = max(x)
        maxy = max(y)
        maxz = max(z)

        widthx = maxx-minx
        widthy = maxy-miny

        hight = maxz-minz
        objects[idx]=Object(round(widthx*100),round(widthy*100),round(hight*100))




    # # -----
    # # Visualization
    # # -----
    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    # entities = [point_cloud]
    # entities.append(point_cloud_table)
    # entities.extend(group_point_clouds)


    # o3d.visualization.draw_geometries(entities,
    #                                     zoom=view['trajectory'][0]['zoom'],
    #                                     front=view['trajectory'][0]['front'],
    #                                     lookat=view['trajectory'][0]['lookat'],
    #                                     up=view['trajectory'][0]['up'],
    #                                     point_show_normal = False)
    
    
    return group_point_clouds,objects