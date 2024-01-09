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

    def findPlane(self, distance_threshold=0.2, ransac_n=3, num_iterations=100):
        plane_model, inlier_idxs = self.input_point_cloud.segment_plane(distance_threshold = distance_threshold,
                                                                        ransac_n=ransac_n, 
                                                                        num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model
        self.inliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = False)
        self.outliers = self.input_point_cloud.select_by_index(inlier_idxs, invert = True)


def main():



    # --------------------------------------
    # Initialization
    # --------------------------------------
    point_cloud = o3d.io.read_point_cloud('scenes/01.ply')
   # o3d.io.write_point_cloud("scene_01.pcd", point_cloud)
    



   # print(point_cloud)

   
    # # -----
    # # Plane Segmentation
    # # -----
    # plane_model, inliers_idx = point_cloud.segment_plane(distance_threshold=0.1,
    #                                      ransac_n=3,
    #                                      num_iterations=100)
    # a,b,c,d = plane_model

    # point_cloud_inliers =point_cloud.select_by_index(inliers_idx, invert = False)
    # point_cloud_inliers.paint_uniform_color([1,0,0])

    # point_cloud_outliers =point_cloud.select_by_index(inliers_idx, invert = True)
    # # save point cloud without the points from the floor
    # o3d.io.write_point_cloud('factory_no_floor.pcd',point_cloud_outliers)

    # --------------------------------------
    # Downsampling
    # --------------------------------------
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.02)


    # # o3d.io.write_point_cloud('factory_no_floor.ply',point_cloud_outliers)

    # # --------------------------------------
    # # Clustering
    # # --------------------------------------
    # labels =   point_cloud_inliers.cluster_dbscan(eps=0.05,
    #                                                 min_points=10,
    #                                                 print_progress=True)
    
    # groups_idxs = list(set(labels)) # gives a list of the values in the labels list
    # # groups_idxs.remove(-1) # remove last group because is the group of unassigned points
    # num_groups = len(groups_idxs)
    # colormap = cm.Pastel1(range(0, num_groups))
    # group_point_clouds = []
    # for group_idx in groups_idxs:

    #     group_points_idxs = list(locate(labels, lambda x: x==group_idx))
    #     group_point_cloud = point_cloud_inliers.select_by_index(group_points_idxs, invert = False)
    #     color = colormap[group_idx, 0:3]
    #     group_point_cloud.paint_uniform_color(color)
    #     group_point_clouds.append(group_point_cloud)

  # -----
    # Execution
    # -----
    # Estimate normals
    point_cloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Orientate normals
    point_cloud_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    print(np.array(point_cloud_downsampled.normals))
    exit(0)
    ## --- Find table plane
    # Compare normals to vector [0,0,1] to find points belonging to horizontal planes
    # create point cloud with "horizontal" points
    point_cloud_horizontal = o3d.geometry.PointCloud()
    for point, normal, color in zip(point_cloud_downsampled.points, 
                                    point_cloud_downsampled.normals,
                                    point_cloud_downsampled.colors):

        # compute angle between 2 3d vectors
        # find norm of the normal vector
        norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        # find norm of z axis
        z_axis = [0,0,1]
        norm_z_axis = math.sqrt(z_axis[0]**2 + z_axis[1]**2 + z_axis[2]**2)

        # get angle betwwen the normal vector and z axis
        theta =  math.acos(np.dot(normal, z_axis)/norm_normal*norm_z_axis)
        print('Theta: ' + str(theta))

        # keep points where angle to z_axis is small enough
        if theta < 0.36: # we have a point that belongs to an horizonta surface
            point_cloud_horizontal.points.append(point)
            point_cloud_horizontal.points.append(normal)
            point_cloud_horizontal.points.append(color)

    print(point_cloud_horizontal)
    # -----
    # Visualization
    # -----
    entities = [point_cloud_horizontal]
    # entities.extend(group_point_clouds) # Make sure to put the geometry in a list
    o3d.visualization.draw_geometries(entities,
                                      zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      point_show_normal = False)

 

if __name__ == "__main__":
    main()
