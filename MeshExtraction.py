import numpy as np
import open3d as o3d 
import trimesh
import pymeshlab

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

class ModelExtraction(object):
    def __init__(self, model_name, filtering=False):
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), 'model', model_name + '.obj')
        self.filering = filtering
        self.data_backup = 'data_backup/' + model_name + '/'

    def load_model(self):
        mesh = o3d.io.read_triangle_mesh(self.model_path)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    
    def find_adj_pairs(self, vertices, triangles):
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        adjacent_pairs = trimesh.graph.face_adjacency(mesh=mesh_tri)
        return adjacent_pairs[adjacent_pairs[:, 0].argsort()]
        

    def compute_geometry(self):
        if self.filering:
            self.apply_filter()
        vertices, triangles = self.load_model()
        adjacent_pairs = self.find_adj_pairs(vertices, triangles)

        print('\n MeshExtraction -> compute \n')
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
       
        self.save_data(vertices, triangles, adjacent_list, centroids)
        return vertices, triangles, adjacent_list, centroids

    def save_data(self, vertices, triangles, adjacent_list, centroids):
        
        with open(self.data_backup + 'triangles.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['tri_idx', 'triangles', 'adjacent', 'centroids'])
            for i in range(len(adjacent_list)):
                writer.writerow([i, triangles[i].tolist(), adjacent_list[i], centroids[i]])

        with open(self.data_backup + 'vertices.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['ver_idx','coord'])
            for i in range(len(vertices)):
                writer.writerow([i, vertices[i].tolist()])

        
    def load_data(self):
        print('\n load data \n')
        df = pd.read_csv(self.data_backup + 'triangles.csv', converters={'adjacent':pd.eval, 'centroids': pd.eval})
        triangles = df['triangles'].tolist()
        adjacent_list = df['adjacent'].tolist()
        centroids = df['centroids'].tolist()
        print('\n load data \n')
        df = pd.read_csv(self.data_backup + 'vertices.csv', converters={'coord':pd.eval})
        vertices = df['coord'].tolist()
        return vertices, triangles, adjacent_list, centroids
    
    def apply_filter(self):
        if os.path.exists(self.model_path[:-4] + '_filtered.obj'):
            None
        else:
            mesh_in = pymeshlab.MeshSet()
            mesh_in.load_new_mesh(self.model_path)
            mesh_in.apply_coord_laplacian_smoothing(stepsmoothnum=20)
            mesh_in.save_current_mesh('model/' + self.model_name + '_filtered.obj') 
        self.model_path = self.model_path[:-4] + '_filtered.obj'
    
    def extraction(self):
        if os.path.exists(self.data_backup) and len(os.listdir(self.data_backup)) > 0:
            # very slow compared to computation
            # return self.load_data()
            return self.compute_geometry()
        elif os.path.exists(self.data_backup):
            return self.compute_geometry()
        else:
            print('please create folder with \'model_name\' in data_backup')


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
    

    