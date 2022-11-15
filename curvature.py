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

#%% Functions
"""
Fuction needed to compute principal curvatures

"""
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def get_area(a,b,c): # area of triangle using heron's formula
	v1 = get_vector(a,b)
	v2 = get_vector(a,c)
	v3 = get_vector(b,c)
	x = np.linalg.norm(v1)
	y = np.linalg.norm(v2)
	z = np.linalg.norm(v3)
	s = (x+y+z)/2.0

	area = (s*(s-x)*(s-y)*(s-z))**0.5
	return area

def check_obtuse(triangle,vertices,index): # checks if triangle is obtuse
	flag=0
	x = vertices[index]
	ind_arr = []
	ind_arr.append(index)
	
	arr = different_elements(tuple(triangle),tuple(ind_arr))
	y = vertices[int(arr[0])]
	z = vertices[int(arr[1])]

	v1 = get_vector(x,y)
	v2 = get_vector(x,z)
	angle = get_angle(v1,v2)
	if angle>np.pi/2.:
		flag=1
	v1 = get_vector(y,z)
	v2 = get_vector(y,x)
	angle = get_angle(v1,v2)
	if angle>np.pi/2.:
		flag=2
	v1 = get_vector(z,x)
	v2 = get_vector(z,y)
	angle = get_angle(v1,v2)
	if angle>np.pi/2.:
		flag=2

	return flag

def get_vector(x,y): # returns vector from x to y
	return y-x

def get_angle(x,y): # return tan of angle between vectors x and y
	x = x/np.linalg.norm(x,2)
	y = y/np.linalg.norm(y,2)
	cos = np.dot(x,y)
	
	return np.arccos(cos)
 
def get_neighbors_2(idx, triangles, vertices):
  neighbor_tiangles = np.where(triangles == idx)
  neighbor_vertices = []
  tri_idx = neighbor_tiangles[0]
  for i in tri_idx : neighbor_vertices.append(np.ndarray.tolist(triangles[i]))
  return np.array(neighbor_vertices)
  # this is for neighbor vertices (but they return a list with all triangles instead)
  # for i in tri_idx : neighbor_vertices = np.append(neighbor_vertices, triangles[tri_idx])

def get_neighbors(index,triangles): # get 1 ring neighborhood for ith vertex
	ring_neighbors = []
	for tri in triangles:
		if index in tri:
			ring_neighbors.append(tri)

	return np.array(ring_neighbors)

def common_elements(list1, list2): # get common elements of list1 and list2
	return list(set(list1) & set(list2))

def different_elements(list1, list2): # get uncommon elements of list1 and list2
	return list(set(list1) ^ set(list2))

def A_mixed(i,vertex,neighbors,vertices,triangles):

	A_mixed = 0

	if neighbors.dtype==object:
		return '#'

	summation = 0
	for j in range(neighbors.shape[0]):
		triangle = neighbors[j]

		flag1 = check_obtuse(triangle,vertices,i)

		if flag1==0: # not obtuse
		
			arr = np.delete(triangle,np.where(triangle==np.float64(i)))
			
			x1 = vertex
			x2 = vertices[int(arr[0])]
			x3 = vertices[int(arr[1])]

			v1 = get_vector(x1,x2)
			v2 = get_vector(x1,x3)
			v3 = get_vector(x2,x3)
			cot_alpha = 1.0/np.tan(get_angle(-v1,v3))

			cot_beta = 1.0/np.tan(get_angle(-v2,-v3))

			summation += (cot_alpha*np.linalg.norm(v2,2)**2 + cot_beta*np.linalg.norm(v1,2)**2)/8.0

		elif flag1==1:
			area = get_area(vertices[triangle[0]],vertices[triangle[1]],vertices[triangle[2]])
			summation += area/2.0
		else:
			area = get_area(vertices[triangle[0]],vertices[triangle[1]],vertices[triangle[2]])
			summation += area/4.0

	A_mixed += summation

	# print('A_mixed', A_mixed)
	return A_mixed

def mean_normal_curvature(i,vertex,A_mixed,neighbors,vertices,triangles):
	summation = np.array([0.,0.,0.])

	for j in range(neighbors.shape[0]):
		triangle = neighbors[j]
		
		arr = np.delete(triangle,np.where(triangle==np.float64(i)))
		
		x1 = vertex
		x2 = vertices[int(arr[0])]
		x3 = vertices[int(arr[1])]

		v1 = get_vector(x1,x2)
		v2 = get_vector(x1,x3)
		v3 = get_vector(x2,x3)
		cot_alpha = 1.0/np.tan(get_angle(-v1,v3))
		cot_beta = 1.0/np.tan(get_angle(-v2,-v3))

		summation += (cot_alpha*v2 + cot_beta*v1)

	K = summation/(2.0*A_mixed)
	return K

def gaussian_curvature(i,vertex,A_mixed,neighbors,vertices,triangles):

	summation = 0.
	for j in range(neighbors.shape[0]):
		triangle = neighbors[j]
		arr = np.delete(triangle,np.where(triangle==i))

		a = vertices[int(arr[0])]
		b = vertices[int(arr[1])]
		c = vertex

		v1 = get_vector(c,a)
		v2 = get_vector(c,b)

		theta = get_angle(v2,v1)

		summation += theta

	K_G = ((2.0*np.pi) - summation)/(A_mixed)
	return K_G

def mean_curvature(K):
	K_H = np.linalg.norm(K,2)/2.
	return K_H

def principal_curvature(K_H,K_G):
	delta = K_H*K_H - K_G
	
	if delta<0:
		delta=0

	K1 = K_H + delta**0.5
	K2 = K_H - delta**0.5

	return K1,K2


# %%
"""
Load a mesh model in format .obj
"""
#MODEL_NAME = 'castle_nut.obj'
MODEL_NAME = 'm02_roof_008_filtered.obj'
#MODEL_NAME = 'm02_roof_020.obj'
MODEL_NAME = 'm02_roof_020_filtered.obj'
#MODEL_NAME = 'roof_mesh_008.obj'
#MODEL_NAME = '604.obj'
MODEL_PATH = 'model/' + MODEL_NAME

if '.obj' in MODEL_PATH:
  mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
  mesh.compute_vertex_normals()
  vertices = np.asarray(mesh.vertices)
  VN = np.asarray(mesh.vertex_normals) # they are normalized
  triangles =  np.asarray(mesh.triangles)
else: print('No valid file extension')

print("numer of verices: ", len(vertices))
print("numer of triangles: ", len(triangles))

# %%

"""
Compute principal curvatures
"""
arr_K_G = []
arr_K_H = []
arr_K1 = []
arr_K2 = []
arr_K1_F = []
arr_K2_F = []

# check if curvatures where already computed       
if os.path.exists('curvature_backup/curvature_vertex_' + MODEL_NAME[:-4] + '.csv'):
	df = pd.read_csv('curvature_backup/curvature_vertex_' + MODEL_NAME[:-4] + '.csv')
	arr_K_H = df['Mean'].tolist()
	arr_K_G = df['Gaussian'].tolist()
	arr_K1 = df['Principal_1'].tolist()
	arr_K2 = df['Principal_2'].tolist() 
	df = pd.read_csv('curvature_backup/curvature_face_' + MODEL_NAME[:-4] + '.csv')
	arr_K1_F = df['Principal_1'].tolist()
	arr_K2_F = df['Principal_2'].tolist()

else:
	print('look for neighboring vertices for each vertex')
	neighbor_vertices = []
	for idx in tqdm(range(len(vertices))):
		neighbor_vertices.append([get_neighbors_2(idx, triangles, vertices)])
		# if idx%10000 == 0: print(f"neighbors found for triangles {idx}/{len(vertices)}")
	print('compute pricipal curvatures for each vertex')
	for i in tqdm(range(len(vertices))):
		#if i%2000 == 0: print('\nVertex: ' + str(i),'/',len(vertices))
		# neighbors = get_neighbors(np.copy(i),np.copy(triangles))
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

	
	
		
#%%
"""
Label regions
"""
label_region = []
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
  label_region.append(Pr.index(max(Pr)))

print("--------- End ---------")
print("Number of Quadric surface: ", label_region.count(0))
print("Number of Rolling-ball blending surface: ", label_region.count(1))
print("Number of irregularity: ", label_region.count(2))
  
#%%  
"""
Visualize point cloud with labels
"""
label_region = np.array(label_region)
mask_0 = np.zeros(len(vertices), dtype=bool)
mask_1 = np.zeros(len(vertices), dtype=bool)
mask_2 = np.zeros(len(vertices), dtype=bool)
mask_0[np.where(label_region==0)[0]] = 1
mask_1[np.where(label_region==1)[0]] = 1
mask_2[np.where(label_region==2)[0]] = 1

rgb = np.ones((len(vertices),3))
rgb[mask_0] = [153./256,204./256,255./256]
rgb[mask_1] = [153./256,255./256,153./256]
rgb[mask_2] = [255./256,153./256,153./256]

mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries([mesh])

#%%

