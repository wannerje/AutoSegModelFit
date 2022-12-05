import numpy as np
import math
import torch 
from dgl.geometry import farthest_point_sampler
from tqdm import tqdm
import csv
import os
import pandas as pd

import MeshVisualizer

# move centroids to tnesor 

def get_initial_seeds(centroids, a, visualize):
    
    # build tensor   
    centroids = np.array(centroids)
    centroids_tensor = torch.Tensor(centroids)
    centroids_tensor = torch.reshape(centroids_tensor, (1, len(centroids_tensor), 3))
    
    # chose number of seeds based on percentage
    seed_num = int(len(centroids) * a)

    # iterative farthest point optimization
    seeds_idx = farthest_point_sampler(centroids_tensor, npoints=seed_num, start_idx=0)
    seeds_idx = seeds_idx[0].tolist()

    if visualize: MeshVisualizer.visualize_initial_seeds(centroids, seeds_idx)

    return seeds_idx

def closest_seed(tri, seeds):
    seeds = np.asarray(seeds)
    dist_2 = np.sum((seeds - tri)**2, axis=1)
    return np.argmin(dist_2)

def get_cluster_seeds(centroids, initial_seeds, model_name, visualize= False):
    cluster_seeds = []
    data_backup = 'data_backup/' + model_name + '/'
    # data_backup + 'cluster_seeds.csv'
    if os.path.exists(data_backup + 'cluster_seeds.csv'):
        df = pd.read_csv(data_backup + 'cluster_seeds.csv', converters={'cluster_seeds': pd.eval})
        cluster_seeds = df['cluster_seeds'].tolist()
    else:  
        centroids = np.array(centroids)
        for centroid in tqdm(centroids):
            cluster_seeds.append(closest_seed(centroid, centroids[initial_seeds]))
        with open(data_backup + 'cluster_seeds.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['idx','cluster_seeds'])
            for i in range(len(cluster_seeds)):
                writer.writerow([i, cluster_seeds[i]])
    if visualize: MeshVisualizer.visualize_cluster_seeds(centroids, cluster_seeds, initial_seeds)
    return cluster_seeds

def superfacet_graph(initial_seeds, cluster_triangles_to_seeds, triangle_regions_label, centroids, vis=False):
    superfacet_triangles = []
    superfacet_regions_label = []
    cluster_triangles_to_seeds = np.array(cluster_triangles_to_seeds)
    for init_seed in range(len(initial_seeds)):
        tri_idx = np.where(cluster_triangles_to_seeds == init_seed)
        superfacet_triangles.append(list(tri_idx[0]))
        # avoid this if later
        if len(triangle_regions_label[tri_idx]) > 0:
            superfacet_regions_label.append(np.bincount(triangle_regions_label[tri_idx]).argmax()) 
    if vis: MeshVisualizer.superfacet_regions_label(centroids, superfacet_triangles,superfacet_regions_label)
    return superfacet_triangles, superfacet_regions_label

def find_nearest_facet(tri, seeds):
    # compute euclidean distance and keep neareset 5
	seeds = np.asarray(seeds)
	dist_2 = np.sum((seeds - tri)**2, axis=1)
	num_neighbour = 5
	return np.argsort(dist_2)[::-1][:num_neighbour]

def get_adjacent_superfacets(initial_seeds, centroids):
    superfacet_adjacency = []
    #initial_seeds = np.array(initial_seeds)
    centroids = np.array(centroids)
    for curr_seed in initial_seeds:
        neighbour_facets = find_nearest_facet(centroids[curr_seed], centroids[initial_seeds])
        superfacet_adjacency.append(neighbour_facets)
    return superfacet_adjacency


def get_geometry(k1, k2, w1, w2, superfacet_triangles):
    k1_sf = []
    k2_sf = []
    k3_sf = [] # variance
    k4_sf = [] # k1 - k2
    w1_sf = []
    w2_sf = []
    k1 = np.array(k1)
    k2 = np.array(k2)
    for idx in superfacet_triangles:
        k1_sf.append(np.mean(k1[idx]))
        k2_sf.append(np.mean(k2[idx]))
        k3_sf.append(np.var(k1[idx]))
        k4_sf.append(k1_sf[-1] - k2_sf[-1])
        w1_sf.append(np.mean(w1[idx]))
        w2_sf.append(np.mean(w2[idx]))
    return np.array(k1_sf), np.array(k2_sf), k3_sf, k4_sf, np.array(w1_sf), np.array(w2_sf)

def prob_gauss(x, sigma):
  return math.exp(-0.5*(x/sigma)**2)  #/ (math.sqrt(2*math.pi) * sigma)

def gauss_partition(k1_sp, k3_sp, k4_sp, sigma, centroids, superfacet_triangles, print_label= False, vis= False):

    Pr_region = []
    shape_label = []

    for i in range(len(k1_sp)):
        Pr1 = prob_gauss(k1_sp[i], sigma)*prob_gauss(k3_sp[i], sigma)*prob_gauss(k4_sp[i], sigma)
        Pr2 = prob_gauss(1-k1_sp[i], sigma)*prob_gauss(k3_sp[i], sigma)*prob_gauss(k4_sp[i], sigma)
        Pr3 = prob_gauss(1-k1_sp[i], sigma)*prob_gauss(k3_sp[i], sigma)*prob_gauss(1-k4_sp[i], sigma)
        Pr4 = prob_gauss(1-k1_sp[i], sigma)*prob_gauss(1-k3_sp[i], sigma)*prob_gauss(k4_sp[i], sigma)
        Pr5 = prob_gauss(1-k1_sp[i], sigma)*prob_gauss(1-k3_sp[i], sigma)*prob_gauss(1-k4_sp[i], sigma)
        Pr = [Pr1, Pr2, Pr3, Pr4, Pr5]
        Pr_region.append(Pr)
        shape_label.append(Pr.index(max(Pr)))
    if print_label:
        print("label 1: ", shape_label.count(0))
        print("label 2: ", shape_label.count(1))
        print("label 3: ", shape_label.count(2))
        print("label 4: ", shape_label.count(3))
        print("label 5: ", shape_label.count(4))
    if vis: MeshVisualizer.superfacet_region_probability(shape_label, centroids, superfacet_triangles)
    return Pr_region, shape_label

def topo_smooth_constraint(W_sf, adj_sf, epsilon, label_region_sf):
    V_ij = []
    for sf in range(len(adj_sf)):
        """ iterate over all sf"""
        V_row = []
        for adj in adj_sf[sf]:
            """ iterate over all adj in sf's """
            constraint = 0
            distance = epsilon * np.linalg.norm(W_sf[sf]-W_sf[adj])
            for l in range(5):
                """ for each gess of label """
                if l == label_region_sf[adj]:
                    constraint += min(1, distance)
                else:
                    constraint += 1
            V_row.append(constraint)
        V_ij.append(V_row)
    return V_ij
        
def build_cost_matrix(Pr_region_sf, V_sf_ij, beta):
    """
    U(l) = sum_i(D_i) + beta * sum_ij(V_ij)  (shape: 3 x N)
    """
    D_triangle_i = np.ones_like(Pr_region_sf)
    D_triangle_i -= Pr_region_sf
    return np.transpose(D_triangle_i + beta * np.reshape(V_sf_ij, (len(V_sf_ij),5)))

def minimize_cost_matrix(Cost_matrix_sf, centroids, superfacet_triangles, visualize= False):
    print('sf: ', Cost_matrix_sf.shape)
    index_labels_sf = np.argmin(Cost_matrix_sf, axis=0)
    if visualize: MeshVisualizer.region_label_sf(index_labels_sf, centroids, superfacet_triangles)
    return np.array(index_labels_sf)