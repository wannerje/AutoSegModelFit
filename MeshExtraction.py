import numpy as np
import open3d as o3d 
import trimesh
import os
import pandas as pd
import csv
from ast import literal_eval
from tqdm import tqdm


"""
    Args: 
        string path to the model 
    Retur:
        Vertices, triangles  
"""


import sys


def model_extraction(path, model_name):

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles =  np.asarray(mesh.triangles)

    if os.path.exists('curvature_backup/adjacent_triangles_' + model_name[:-4] + '.csv'):
        #df = pd.read_pickle('curvature_backup/adjacent_triangles_' + model_name[:-4] + '.csv')
        #df = pd.read_csv('curvature_backup/adjacent_triangles_' + model_name[:-4] + '.csv', converters={'full_set': literal_eval})
        df = pd.read_csv('curvature_backup/adjacent_triangles_' + model_name[:-4] + '.csv', converters={'full_set':pd.eval, 'centroids': pd.eval})
        adjacent_list = df['full_set'].tolist()
        centroids = df['centroids'].tolist()
        
    else:
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        adjacent_pairs = trimesh.graph.face_adjacency(mesh=mesh_tri)
        adjacent_pairs = adjacent_pairs[adjacent_pairs[:, 0].argsort()]
        adjacent_list = []
        centroids = []
        for i in tqdm(range(len(triangles))):
            
            # centroids       
            temp = 0
            for ver in triangles[i]:
                temp += vertices[ver]
            centroids.append(list(temp/3))

            # adjacent
            index = np.where(adjacent_pairs[:,0]==i)[0]
            full_set = np.take(adjacent_pairs[:,1], index)
            adjacent_list.append(full_set.tolist())
        
        with open('curvature_backup/adjacent_triangles_' + model_name[:-4] + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            #adjacent_list = np.array(adjacent_list, dtype=object)
            writer.writerow(['tri_idx','full_set', 'centroids'])
            for i in range(len(adjacent_list)):
                #adjacent_list[i] = np.delete(adjacent_list[i],[-1,0])
                #writer.writerow([i, [x.split(',') for x in adjacent_list[i].tolist()]])
                writer.writerow([i, adjacent_list[i], centroids[i]])
      

    return vertices, triangles, adjacent_list, centroids, mesh
    

    