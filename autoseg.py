
import os 

import numpy as np

import MeshExtraction
import MeshGeometry
import MeshRegionPartition
import MeshVisualizer
import MeshSuperfacet

import sys


# Specify mesh model to segment 
MODEL_NAME = 'm02_roof_020_filtered.obj'
MODEL_PATH = os.path.join(os.getcwd(), 'model', MODEL_NAME)

# Process and extract relevant vertices and triangles 
vertices, triangles, adjacent_tri, centroids, mesh = MeshExtraction.model_extraction(path=MODEL_PATH, model_name=MODEL_NAME)

# Compute geometric properties 
"""
    Principal curvatures: k1, k2 
    Principal directions: w1, w2
    W = k1*w1, k2*w2 (size= Nx6)
"""
k1, k2 = MeshGeometry.triangle_principal_curvatures(vertices, triangles, MODEL_NAME, backup = True)
w1, w2 = MeshGeometry.triangle_principal_directions(vertices, triangles)
W = np.concatenate((w1*k1[:, np.newaxis], w2*k2[:, np.newaxis]), axis=1)
# Region probability 
"""
    Prob for each triangle to belong to a surface type 
    l1: quadric surface
    l2: blending surface
    l3: irregularity
    Pr(l1,l2,l3) = [pr_quadric, pr_blending, pr_irregularity]
"""
sigma = 0.05
Pr_triangles_i = MeshRegionPartition.gauss_partition(k1,k2, sigma)

# Topological smootheness constraints
epsilon = 0.1
V_triangle_ij = MeshRegionPartition.topo_smooth_constraint(adjacent_tri, W, epsilon)


# linear_sum_assignment
beta = 0.5
Cost_matrix = MeshRegionPartition.build_cost_matrix(np.copy(Pr_triangles_i), np.copy(V_triangle_ij), beta)


label = MeshRegionPartition.minimize_cost_matrix(Cost_matrix, visualize= True)
print(f"\n# label 1: {np.count_nonzero(label == 0)} \n# label 2: {np.count_nonzero(label == 1)} \n# label 3: {np.count_nonzero(label == 2)} \n ")

visualize= False
if visualize == True:
    MeshVisualizer.region_label(label, centroids, mesh)

# Mesh superfacets
a = 0.1 # percentage mesh size
initial_seeds = MeshSuperfacet.get_initial_seeds(centroids, a, visualize= True)

cluster_seeds = MeshSuperfacet.get_cluster_seeds(centroids, initial_seeds, visualize= False)




# #%% MESH SUPERFACETS
# """
# __ Init __

# 1.  Generate random seeds among the three regions using "ierative farthest point optimisation"
#     (number of seeds shuld be defined accordingly to the size of the entire mesh)
# 2.  Take each seed as a source and for each trianglein the "radius" neighborhood:
#       a. update geodesic distance from it 
#       b. and corresponding label 
# """
# import torch 
# from dgl.geometry import farthest_point_sampler

# """
# Args:
#   tirangles         (as numpy)
#   vetices           (as numpy)

# Returns:
#   centroids         (as numpy)
#   centrids_tensor   (as torch)

# """

# # 1. get centroids and move to tensor dtype=torch.float32
# centroids = []
# for tri in triangles:
# 	temp = 0
# 	for ver in tri:
# 		temp += vertices[ver]
# 	centroids.append(temp/3)
 
# # 2. move to tensor (dtype = torch.float32)
# centroids = np.array(centroids)
# centroids_tensor = torch.Tensor(centroids)
# centroids_tensor = torch.reshape(centroids_tensor, (1, len(centroids_tensor), 3))
# #centroids_index = np.arange(0,len(centroids))

# """
# Args:
#   centroids         (as numpy)
#   centrids_tensor   (as torch)

# Returns:
#   seeds_idx         (as list)

# """
# # 3. define number of centroids
# # for now estimate, for later find a scalable solution like (mesh_size)/(superfacet_size)
# seed_num = int(len(centroids) * 0.1)

# # 4. iterative farthest point optimization
# seeds_idx = farthest_point_sampler(centroids_tensor, npoints=seed_num, start_idx=0)
# seeds_idx = seeds_idx[0].tolist()
# #print(point_idx)

# # ## visualize
# # rgb = np.ones((len(centroids),3))*0.15 # /256
# # mask = np.zeros(len(centroids), dtype=bool)
# # mask[seeds] = 1
# # rgb[mask] = [255./256,153./256,153./256] 
# # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
# # pcd.colors = o3d.utility.Vector3dVector(rgb)
# # o3d.visualization.draw_geometries([pcd])
# # 5. assign triangles to closest superfacet

# """
# Args:
#   centroids         (as numpy)
#   seeds             (as list)

# Returns:
#   cluster_seed         (as list)

# """
# # find closest seed
# def closest_seed(tri, seeds):
#     seeds = np.asarray(seeds)
#     dist_2 = np.sum((seeds - tri)**2, axis=1)
#     return np.argmin(dist_2)

# # keep cetroids not belonging to the seeds
# rest_centroids = np.delete(centroids, seeds_idx, axis=0)

# #  = closest_seed(rest_centroids[0], centroids[seeds_idx])
# # find closest seed for each triangle 
# cluster_seed = []
# # for rc in rest_centroids:
# # 	cluster_seed.append(closest_seed(rc, centroids[seeds_idx]))

# for centroid in centroids:
# 	cluster_seed.append(closest_seed(centroid, centroids[seeds_idx]))

