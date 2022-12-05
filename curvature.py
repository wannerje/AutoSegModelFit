import numpy as np
import sys
import csv
import open3d as o3d
import math
# Compute probability on vertex curvatures 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import os
import pandas as pd

from curvature_utils import*


# %% Load mesh model 

#MODEL_NAME = 'castle_nut.obj'
#MODEL_NAME = 'm02_roof_008_filtered.obj'
#MODEL_NAME = 'm02_roof_020.obj'
MODEL_NAME = 'm02_roof_020_filtered.obj'
#MODEL_NAME = 'roof_mesh_008.obj'
#MODEL_NAME = '604.obj'
MODEL_PATH = 'model/' + MODEL_NAME

if '.obj' in MODEL_PATH:
  mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
  mesh.remove_duplicated_vertices()
  mesh.remove_degenerate_triangles()
  mesh.compute_vertex_normals()
  vertices = np.asarray(mesh.vertices)
  VN = np.asarray(mesh.vertex_normals) # they are normalized
  triangles =  np.asarray(mesh.triangles)
else: print('No valid file extension')

print("numer of verices: ", len(vertices))
print("numer of triangles: ", len(triangles))

# %% Compute principal curvatures

arr_K_G = []
arr_K_H = []
arr_K1 = []
arr_K2 = []
arr_K1_F = []
arr_K2_F = []

# check if curvatures where already computed       
if os.path.exists('curvature_backup/curvature_vertex_' + MODEL_NAME[:-4] + '.csv'):
	df = pd.read_csv('curvature_backup/curvature_vertex_' + MODEL_NAME[:-4] + '.csv')
	arr_K1 = df['Principal_1'].tolist()
	arr_K2 = df['Principal_2'].tolist() 
	df = pd.read_csv('curvature_backup/curvature_face_' + MODEL_NAME[:-4] + '.csv')
	arr_K1_F = df['Principal_1'].tolist()
	arr_K2_F = df['Principal_2'].tolist()

else:
	print('look for neighboring vertices for each vertex')
	neighbor_vertices = []
	for idx in tqdm(range(len(vertices))):
		#neighbors = get_neighbors(np.copy(idx),np.copy(triangles))
		neighbor_vertices.append([get_neighbors_2(idx, triangles)])
	print('compute pricipal curvatures for each vertex')
	for i in tqdm(range(len(vertices))):
		neighbors = neighbor_vertices[i][0]
		a_mixed = A_mixed(i,vertices[i],np.copy(neighbors),np.copy(vertices),np.copy(triangles))
		if a_mixed=='#' or a_mixed==0:
			arr_K_G.append(0.)
			arr_K_H.append(0.)
			arr_K1.append(0.)
			arr_K2.append(0.)
			continue
		K_G = gaussian_curvature(np.copy(i),np.copy(vertices[i]),np.copy(a_mixed),np.copy(neighbors),np.copy(vertices),np.copy(triangles)) 
		K = mean_normal_curvature(np.copy(i),np.copy(vertices[i]),np.copy(a_mixed),np.copy(neighbors),np.copy(vertices),np.copy(triangles))
		K_H = mean_curvature(K)
		K1, K2 = principal_curvature(K_H,K_G)
		
		arr_K_G.append(K_G)
		arr_K_H.append(K_H)
		arr_K1.append(K1)
		arr_K2.append(K2)
	
	print('avarage principal curvatures for each face')
	for ver_idx in tqdm(triangles):
		arr_K1_F.append(np.average([arr_K1[ver_idx[0]], arr_K1[ver_idx[1]], arr_K1[ver_idx[2]]]))
		arr_K2_F.append(np.average([arr_K2[ver_idx[0]], arr_K2[ver_idx[1]], arr_K2[ver_idx[2]]]))

	# save to csv file 
	with open('curvature_backup/curvature_vertex_' + MODEL_NAME[:-4] + '.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['Vertex', 'Principal_1', 'Principal_2', 'Mean', 'Gaussian', 'Principal_1_F', 'Principal_2_F'])
		for i in range(len(arr_K1)):
			writer.writerow([i, arr_K1[i], arr_K2[i], arr_K_H[i], arr_K_G[i]])
	with open('curvature_backup/curvature_face_' + MODEL_NAME[:-4] + '.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['Triangle_id', 'Principal_1', 'Principal_2'])	
		for j in range(len(arr_K1_F)):
			writer.writerow([i, arr_K1_F[j], arr_K2_F[j]])
		
#%% Label regions 

def normalize_curvature(K1,K2):
	K = [K1, K2]
	K_sum_positive =  sum([abs(ele) for ele in K])
	K = [float(i)/K_sum_positive for i in K]
	return K[0], K[1]


label_region = []
label_region_F = []
Pr_triangles = []
Pr_vertices = []
sigma = 0.2

# define non-normalized gaussian distribution
def prob_gauss(x):
  return math.exp(-0.5*(x/sigma)**2)  #/ (math.sqrt(2*math.pi) * sigma)

# compute probabilities
for i in range(len(arr_K1)):
	Pr1 = prob_gauss(arr_K1[i])*prob_gauss(arr_K2[i])
	Pr2 = prob_gauss(1-arr_K1[i])*prob_gauss(arr_K2[i])
	Pr3 = prob_gauss(1-arr_K1[i])*prob_gauss(1-arr_K2[i])
	Pr = [Pr1, Pr2, Pr3]
	Pr_vertices.append(Pr)
	label_region.append(Pr.index(max(Pr)))

for i in range(len(arr_K1_F)):
	#K1, K2 = normalize_curvature(arr_K1_F[i], arr_K2_F[i])
	K1 = arr_K1_F[i]
	K2 = arr_K2_F[i]
	Pr1_F = prob_gauss(K1)*prob_gauss(K2)
	Pr2_F = prob_gauss(1-K1)*prob_gauss(K2)
	Pr3_F = prob_gauss(1-K1)*prob_gauss(1-K2)
	Pr_F = [Pr1_F, Pr2_F, Pr3_F]
	Pr_triangles.append(Pr_F)
	label_region_F.append(Pr_F.index(max(Pr_F)))

print("--------- End ---------")
print("Number of Quadric surface (vertex / faces): ", label_region.count(0), " / ", label_region_F.count(0))
print("Number of Rolling-ball blending surface (vertex / faces): ", label_region.count(1), " / ", label_region_F.count(1))
print("Number of irregularity (vertex / faces): ", label_region.count(2), " / ", label_region_F.count(2))

# label_region = np.array(label_region)
# mask_0 = np.zeros(len(vertices), dtype=bool)
# mask_1 = np.zeros(len(vertices), dtype=bool)
# mask_2 = np.zeros(len(vertices), dtype=bool)
# mask_0[np.where(label_region==0)[0]] = 1
# mask_1[np.where(label_region==1)[0]] = 1
# mask_2[np.where(label_region==2)[0]] = 1

# rgb = np.ones((len(vertices),3))
# rgb[mask_0] = [153./256,204./256,255./256]
# rgb[mask_1] = [153./256,255./256,153./256]
# rgb[mask_2] = [255./256,153./256,153./256]

# mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
# o3d.visualization.draw_geometries([mesh])


#%% Find adjacent faces 
# import trimesh

# mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
# adjacent = trimesh.graph.face_adjacency(mesh=mesh)
#print(adjacent)

#trimesh.graph.face_adjacency()




#%% Region label minimization
import igl

## Cost matrix (num_label x num_triangles)
cost_matrix = np.empty(shape=(3,len(triangles)))

## Consistency term
# Pr_triangles = np.array(Pr_triangles)
# Pr_triangles = np.transpose(Pr_triangles)
# # normalize probabilities
# Pr_triangles = Pr_triangles / Pr_triangles.sum(axis=0)[np.newaxis, :]
# consistency_term_per_triangle = np.ones_like(Pr_triangles)
# consistency_term_per_triangle = consistency_term_per_triangle - Pr_triangles

# #print(consistency_term_sum) # nan values only!

# ## Topological smoothness constraints
# beta = 0.1
# # principal directions (numpay Nx3)
# w1, w2, _, _ = igl.principal_curvature(vertices, triangles)
# print(type(w1))
# print(w1.shape)
# arr_K1_F = np.array(arr_K1_F)
# print(type(arr_K1_F))
# print(arr_K1_F.shape)
# print(len(arr_K1))
#a1 = arr_K1_F * w1
w = []


smoothness_term_per_triangle = None




#%% MESH SUPERFACETS
"""
__ Init __

1.  Generate random seeds among the three regions using "ierative farthest point optimisation"
    (number of seeds shuld be defined accordingly to the size of the entire mesh)
2.  Take each seed as a source and for each trianglein the "radius" neighborhood:
      a. update geodesic distance from it 
      b. and corresponding label 
"""
import torch 
from dgl.geometry import farthest_point_sampler

"""
Args:
  tirangles         (as numpy)
  vetices           (as numpy)

Returns:
  centroids         (as numpy)
  centrids_tensor   (as torch)

"""

# 1. get centroids and move to tensor dtype=torch.float32
centroids = []
print('centroids')
for tri in tqdm(triangles):
	temp = 0
	for ver in tri:
		temp += vertices[ver]
	centroids.append(temp/3)
 
# 2. move to tensor (dtype = torch.float32)
centroids = np.array(centroids)
centroids_tensor = torch.Tensor(centroids)
centroids_tensor = torch.reshape(centroids_tensor, (1, len(centroids_tensor), 3))
#centroids_index = np.arange(0,len(centroids))

"""
Args:
  centroids         (as numpy)
  centrids_tensor   (as torch)

Returns:
  seeds_idx         (as list)

"""
# 3. define number of centroids
# for now estimate, for later find a scalable solution like (mesh_size)/(superfacet_size)
seed_num = int(len(centroids) * 0.1)

# 4. iterative farthest point optimization
seeds_idx = farthest_point_sampler(centroids_tensor, npoints=seed_num, start_idx=0)
seeds_idx = seeds_idx[0].tolist()
#print(point_idx)



# ## visualize
# rgb = np.ones((len(centroids),3))*0.15 # /256
# mask = np.zeros(len(centroids), dtype=bool)
# mask[seeds_idx] = 1
# rgb[mask] = [255./256,153./256,153./256] 
# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
# pcd.colors = o3d.utility.Vector3dVector(rgb)
# o3d.visualization.draw_geometries([pcd])
# 5. assign triangles to closest superfacet

"""
Args:
  centroids         (as numpy)
  seeds             (as list)

Returns:
  cluster_seed         (as list)

"""
# find closest seed
def closest_seed(tri, seeds):
    seeds = np.asarray(seeds)
    dist_2 = np.sum((seeds - tri)**2, axis=1)
    return np.argmin(dist_2)

# keep cetroids not belonging to the seeds
rest_centroids = np.delete(centroids, seeds_idx, axis=0)

#  = closest_seed(rest_centroids[0], centroids[seeds_idx])
# find closest seed for each triangle 
cluster_seed = []
# for rc in rest_centroids:
# 	cluster_seed.append(closest_seed(rc, centroids[seeds_idx]))

for centroid in centroids:
	cluster_seed.append(closest_seed(centroid, centroids[seeds_idx]))



# # Visualize cluster_seed
# import random
# rgb = np.ones((len(centroids),3))*0.15
# cluster_seed = np.array(cluster_seed)
# for s in range(len(seeds_idx)):
# 	mask = np.zeros(len(centroids), dtype=bool)
# 	mask[np.where(cluster_seed==s)[0]] = 1
# 	rgb[mask] = [random.random(), random.random(),random.random()]

# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
# pcd.colors = o3d.utility.Vector3dVector(rgb)
# o3d.visualization.draw_geometries([pcd])


# %% Mesh Segmentation

# construct a new graph: G = {C, E}
# C = [[triangles to superfacet, largest region label]]
# E = [True or False, ...] (Adjacency of superfacets)
face_graph_triangles = []
face_graph_label = []
face_graph_adjacency = []
cluster_seed = np.array(cluster_seed)
label_region_F = np.array(label_region_F)

# C 
for seed in range(len(seeds_idx)):
	tri_idx = np.where(cluster_seed == seed)
	face_graph_triangles.append(list(tri_idx[0]))
l.append(np.bincount(label_region_F[tri_idx]).argmax()) 
# print("Number of Quadric surface: ", face_graph_label.count(0))
# print("Number of Blending surface: ", face_graph_label.count(1))
# print("Number of Irregularity: ", face_graph_label.count(2))

# E
# compute seeds distance and keep as neghbour the shortest 5
def find_nearest_facet(tri, seeds):
	seeds = np.asarray(seeds)
	dist_2 = np.sum((seeds - tri)**2, axis=1)
	num_neighbour = 5
	return np.argsort(dist_2)[::-1][:num_neighbour]

for seed_idx in seeds_idx:
	neighbour_facets = find_nearest_facet(centroids[seed_idx], centroids[seeds_idx])
	face_graph_adjacency.append(neighbour_facets)


#%% Geometric properties 
# compute properties for each superfacet as the average of all triangles
sf_K1 = []
sf_K2 = []
sf_K3 = [] # variance
sf_K4 = [] # k1 - k2

arr_K1_F = np.array(arr_K1_F)
arr_K2_F = np.array(arr_K2_F)

for idx in face_graph_triangles:
	sf_K1.append(np.mean(arr_K1_F[idx]))
	sf_K2.append(np.mean(arr_K2_F[idx]))
	sf_K3.append(np.var(arr_K1_F[idx]))
	sf_K4.append(sf_K1[-1] - sf_K2[-1])

#%% MRF Refined Segmentation on superfacets

shape_label = []

for i in range(len(sf_K1)):
	Pr1 = prob_gauss(sf_K1[i])*prob_gauss(sf_K3[i])*prob_gauss(sf_K4[i])
	Pr2 = prob_gauss(1-sf_K1[i])*prob_gauss(sf_K3[i])*prob_gauss(sf_K4[i])
	Pr3 = prob_gauss(1-sf_K1[i])*prob_gauss(sf_K3[i])*prob_gauss(1-sf_K4[i])
	Pr4 = prob_gauss(1-sf_K1[i])*prob_gauss(1-sf_K3[i])*prob_gauss(sf_K4[i])
	Pr5 = prob_gauss(1-sf_K1[i])*prob_gauss(1-sf_K3[i])*prob_gauss(1-sf_K4[i])
	Pr = [Pr1, Pr2, Pr3, Pr4, Pr5]
	shape_label.append(Pr.index(max(Pr)))

print("label 1: ", shape_label.count(0))
print("label 2: ", shape_label.count(1))
print("label 3: ", shape_label.count(2))
print("label 4: ", shape_label.count(3))
print("label 5: ", shape_label.count(4))

## UNCOMMENT TO VISUALLIZE !!!
shape_label = np.array(shape_label)
rgb = np.ones((len(centroids),3))*0.15
face_graph_triangles = np.array(face_graph_triangles, dtype=object)

mask_ = np.zeros((len(centroids),5), dtype=bool)
for i in range(4):
	#mask_ = np.zeros(len(centroids), dtype=bool)
	sf_label = np.where(shape_label==i)[0]
	tri_label = face_graph_triangles[sf_label]
	for j in tri_label:
		mask_[j,i] = 1
# black (plane)
rgb[mask_[:,1]] = [153./256,71./256,246./256] # purple (sphere)
rgb[mask_[:,2]] = [153./256,255./256,153./256] # green (cylinder)
rgb[mask_[:,3]] = [255./256,153./256,153./256] # reed (other quad. surface)
rgb[mask_[:,4]] = [153./256,204./256,255./256] # blu (cones)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])
sys.exit()



mask[np.where(cluster_seed==s)[0]] = 1

mask_1 = np.zeros(seed_num, dtype=bool)
mask_2 = np.zeros(seed_num, dtype=bool)
mask_3 = np.zeros(seed_num, dtype=bool)
mask_4 = np.zeros(seed_num, dtype=bool)
mask_0[np.where(label_region==0)[0]] = 1
mask_1[np.where(label_region==1)[0]] = 1
mask_2[np.where(label_region==2)[0]] = 1
mask_3[np.where(label_region==1)[0]] = 1
mask_4[np.where(label_region==2)[0]] = 1



rgb[mask_1] = [153./256,255./256,153./256]
rgb[mask_2] = [255./256,153./256,153./256]

mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries([mesh])


## Visualize cluster_seed
import random
rgb = np.ones((len(centroids),3))*0.15
cluster_seed = np.array(cluster_seed)
for s in range(len(seeds_idx)):
	mask = np.zeros(len(centroids), dtype=bool)
	mask[np.where(cluster_seed==s)[0]] = 1
	rgb[mask] = [random.random(), random.random(),random.random()]

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])