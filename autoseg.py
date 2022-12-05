import numpy as np

import MeshGeometry
import MeshRegionPartition
import MeshVisualizer
import MeshSuperfacet
from MeshExtraction import ModelExtraction

import sys
from datetime import datetime
#now = datetime.now()


# specify mdodel name contained in model directory
MODEL_NAME = '604_020'

# model extraction
mesh = ModelExtraction(MODEL_NAME, filtering=True)
vertices, triangles, adjacent_tri, centroids = mesh.extraction()

# Compute geometric properties 
k1, k2 = MeshGeometry.triangle_principal_curvatures(vertices, triangles, MODEL_NAME, backup = True)
w1, w2 = MeshGeometry.triangle_principal_directions(vertices, triangles)
W = np.concatenate((w1*k1[:, np.newaxis], w2*k2[:, np.newaxis]), axis=1)

#%% TRIANGLES REGION LABELING
print('\n MeshRegionPartition -> compute \n')
sigma = 0.06
Pr_triangles_i, region_label_tri = MeshRegionPartition.gauss_partition(k1,k2, sigma)

# Topological smootheness constraints
epsilon = 1 #0.1 
V_triangle_ij = MeshRegionPartition.topo_smooth_constraint(adjacent_tri, W, region_label_tri, epsilon)

# linear_sum_assignment
beta = 1# 0.5
Cost_matrix = MeshRegionPartition.build_cost_matrix(np.copy(Pr_triangles_i), np.copy(V_triangle_ij), beta)

# Type of region for each triangle (quadric, blending, irregularity)
triangle_regions_label = MeshRegionPartition.minimize_cost_matrix(Cost_matrix, centroids, visualize= False)
#print(f"\n# label 1: {np.count_nonzero(label == 0)} \n# label 2: {np.count_nonzero(label == 1)} \n# label 3: {np.count_nonzero(label == 2)} \n ")

#%% SUPERFACET LABELING
print('\n MeshSuperfacet -> compute \n')

# sample initial seeds with farthest point algorithm
a = 0.1 # percentage mesh size
initial_seeds = MeshSuperfacet.get_initial_seeds(   centroids, 
                                                    a, 
                                                    visualize= False)

# Assign to each triangle the index of the closest seed
cluster_triangles_to_seeds = MeshSuperfacet.get_cluster_seeds(  centroids, 
                                                                initial_seeds, 
                                                                MODEL_NAME,
                                                                visualize= False)

# build new graph G = {superfacets, adjacency} 
superfacet_triangles, superfacet_regions_label = MeshSuperfacet.superfacet_graph(   initial_seeds, 
                                                                                    cluster_triangles_to_seeds, 
                                                                                    triangle_regions_label, 
                                                                                    centroids, 
                                                                                    vis=False)

# find adjacent superfacets 
superfacet_adjacency = MeshSuperfacet.get_adjacent_superfacets( initial_seeds, 
                                                                centroids)

# geometric properties superfacets
k1_sf, k2_sf, k3_sf, k4_sf, w1_sf, w2_sf = MeshSuperfacet.get_geometry( k1, k2, 
                                                                        w1, w2, 
                                                                        superfacet_triangles)

W_sf = np.concatenate((w1_sf*k1_sf[:, np.newaxis], w2_sf*k2_sf[:, np.newaxis]), axis=1)

# shape probability (plane, sphere, cylinder, other quadric, cones)
sigma = 0.5 #0.1
Pr_region_sf, label_region_sf = MeshSuperfacet.gauss_partition( k1_sf, k3_sf, k4_sf, 
                                                                sigma, 
                                                                centroids, 
                                                                superfacet_triangles,  
                                                                print_label= False, 
                                                                vis= False)

# smoothness constraints
epsilon = 1 #0.03
V_sf_ij = MeshSuperfacet.topo_smooth_constraint(W_sf, superfacet_adjacency, epsilon, label_region_sf)
#print(V_sf_ij[0:15])

# cost matrix
beta = 1 #0.5
Cost_matrix_sf = MeshSuperfacet.build_cost_matrix(np.copy(Pr_region_sf), np.copy(V_sf_ij), beta)

# shape label 
regions_label_sf = MeshSuperfacet.minimize_cost_matrix(Cost_matrix_sf, centroids, superfacet_triangles, visualize= True)
