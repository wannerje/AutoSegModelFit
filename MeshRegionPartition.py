import math
import numpy as np
from tqdm import tqdm

import MeshVisualizer

#%% Variable and param
Pr_triangles = []


#%% Functions utils

def prob_gauss(x, sigma):
  return math.exp(-0.5*(x/sigma)**2)  #/ (math.sqrt(2*math.pi) * sigma)

def normalize_curvature(K1,K2):
	K = [K1, K2]
	K_sum_positive =  sum([abs(ele) for ele in K])
	K = [float(i)/K_sum_positive for i in K]
	return K[0], K[1]

# def normalize_array_row(arr):
#     row_sums = arr.sum(axis=1)
#     normalized = arr / row_sums[:, np.newaxis]
#     return normalized
#%% 

def gauss_partition(arr_k1,arr_k2, sigma):
    label_region_F = []
    Pr_triangles = []
    for i in range(len(arr_k1)):
        # k1, k2 = normalize_curvature(arr_k1[i], arr_k2[i])
        k1 = arr_k1[i]
        k2 = arr_k2[i]
        Pr1_F = prob_gauss(k1, sigma)*prob_gauss(k2, sigma)
        Pr2_F = prob_gauss(1-k1, sigma)*prob_gauss(k2, sigma)
        Pr3_F = prob_gauss(1-k1, sigma)*prob_gauss(1-k2, sigma)
        Prob = [Pr1_F, Pr2_F, Pr3_F]
        #Prob = [float(i)/sum(Prob) for i in Prob]
        Pr_triangles.append(Prob)
        label_region_F.append(Prob.index(max(Prob)))
    return Pr_triangles, label_region_F

"""
Vij = min(1, ||Wi - Wj||)
"""
import sys
def topo_smooth_constraint(adj_tri, W, region_label_tri, epsilon): 
    # V_ij = []
    # for tri in range(len(adj_tri)):
    #     """ iterate over all tri"""
    #     V_row = []
    #     for adj in adj_tri[tri]:
    #         """ iterate over all adj in sf's """
    #         constraint = 0
    #         distance = epsilon * np.linalg.norm(W[tri]-W[adj])
    #         for l in range(3):
    #             """ for each gess of label """
    #             if l == region_label_tri[adj]:
    #                 constraint += min(1, distance)
    #             else:
    #                 constraint += 1
    #         V_row.append(constraint)
    #     V_ij.append(V_row)
    # return V_ij


    V = []   
    for i in range(len(adj_tri)):
        V_row = []
        for adj_idx in adj_tri[i]:
            distance = epsilon * np.linalg.norm(W[i]-W[adj_idx])
            V_row.append(min(1, distance))
        V.append(sum(V_row))
    return V


def build_cost_matrix(Pr_triangles_i, V_triangle_ij, beta):
    """
    U(l) = sum_i(D_i) + beta * sum_ij(V_ij)  (shape: 3 x N)
    """
    D_triangle_i = np.ones_like(Pr_triangles_i)
    D_triangle_i -= Pr_triangles_i
    return np.transpose(D_triangle_i + beta * np.reshape(V_triangle_ij, (len(V_triangle_ij),1)))
    

   
from scipy.optimize import linear_sum_assignment

def linear_sum_assignement(cost):
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind

def minimize_cost_matrix(Cost_matrix, centroids, visualize):
    """ shape (3 x N) """
    index_labels = np.argmin(Cost_matrix, axis=0)
    if visualize: MeshVisualizer.region_label(index_labels, centroids)
    return np.array(index_labels)
    

