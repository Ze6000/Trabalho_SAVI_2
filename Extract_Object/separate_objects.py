import open3d as o3d
import math
import numpy as np
from matplotlib import cm
from more_itertools import locate
from objects import Object

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
    
    # Getthe point cloud of the table and objects extracted before 
    point_cloud = table_point_cloud

    
    # Remove table plane
    plane_model, inliers_idx = point_cloud.segment_plane(distance_threshold=0.025,
                                        ransac_n=3,
                                        num_iterations=100)
    a,b,c,d = plane_model

    # Table's point cloud
    point_cloud_table = point_cloud.select_by_index(inliers_idx, invert = False)
    point_cloud_table.paint_uniform_color([1,0,0])

    # Objects' point cloud
    point_cloud_objects = point_cloud.select_by_index(inliers_idx, invert = True)
    
    # Find each separated object
    objects =  point_cloud_objects.cluster_dbscan(eps=0.015,
                                                    min_points=50,
                                                    print_progress=False)
    
    groups_idxs = list(set(objects)) 
    groups_idxs.remove(-1) 
    num_groups = len(groups_idxs)
    colormap = cm.Pastel1(range(0, num_groups))
    group_point_clouds = []
    for group_idx in groups_idxs:

        group_points_idxs = list(locate(objects, lambda x: x==group_idx))
        group_point_cloud = point_cloud_objects.select_by_index(group_points_idxs, invert = False)

        color = colormap[group_idx, 0:3]
        group_point_cloud.paint_uniform_color(color)
        print(len(group_point_cloud.points))
        if len(group_point_cloud.points) > 5000: # Ignore very small objects (it's just noise or pieces of table)
            group_point_clouds.append(group_point_cloud)

    # Now that the objects are separated, get their dimensions and save the data
    
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

        # Save object width in x and y axis and hight
        objects[idx]=Object(round(widthx*100),round(widthy*100),round(hight*100))
    
    return group_point_clouds,objects