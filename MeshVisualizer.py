import open3d as o3d
import numpy as np
import random

def region_label(label_region, centroids):
    centroids = np.array(centroids)
    label_region = np.array(label_region)
    label_region = np.array(label_region)
    mask_0 = np.zeros(len(centroids), dtype=bool)
    mask_1 = np.zeros(len(centroids), dtype=bool)
    mask_2 = np.zeros(len(centroids), dtype=bool)
    mask_0[np.where(label_region==0)[0]] = 1
    mask_1[np.where(label_region==1)[0]] = 1
    mask_2[np.where(label_region==2)[0]] = 1

    rgb = np.ones((len(centroids),3))*0.15
    rgb[mask_0] = [0./256,76./256,153./256]
    rgb[mask_1] = [153./256,255./256,153./256]
    rgb[mask_2] = [255./256,153./256,153./256]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

def region_label_sf(label_region, centroids, superfacet_triangles):
    centroids = np.array(centroids)
    label_region = np.array(label_region)
    superfacet_triangles = np.array(superfacet_triangles, dtype=object)

    mask_ = np.zeros((len(centroids),5), dtype=bool)
    for i in range(4):
        #mask_ = np.zeros(len(centroids), dtype=bool)
        sf_label = np.where(label_region==i)[0]
        tri_label = superfacet_triangles[sf_label]
        for j in tri_label:
            mask_[j,i] = 1
    # black (plane) #
    rgb = np.ones((len(centroids),3))*0.15 
    rgb[mask_[:,0]] = [0./256,76./256,153./256] # blue (quadric)
    rgb[mask_[:,1]] = [153./256,71./256,246./256] # purple (sphere)
    rgb[mask_[:,2]] = [153./256,255./256,153./256] # green (cylinder)
    rgb[mask_[:,3]] = [255./256,153./256,153./256] # reed (other quad. surface)
    rgb[mask_[:,4]] = [153./256,204./256,255./256] # blu (cones)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

    
def visualize_initial_seeds(centroids, initial_seeds):
    rgb = np.ones((len(centroids),3))*0.15 # /256
    mask = np.zeros(len(centroids), dtype=bool)
    mask[initial_seeds] = 1
    rgb[mask] = [255./256,153./256,153./256] 
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

def visualize_cluster_seeds(centroids, cluster_seeds, initial_seeds):
    # ## Visualize cluster_seed
    rgb = np.ones((len(centroids),3))*0.15
    cluster_seeds = np.array(cluster_seeds)
    for s in range(len(initial_seeds)):
        mask = np.zeros(len(centroids), dtype=bool)
        mask[np.where(cluster_seeds==s)[0]] = 1
        rgb[mask] = [random.random(), random.random(),random.random()]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

def superfacet_regions_label(centroids, superfacet_triangles, superfacet_regions_label):
    
    superfacet_regions_label = np.array(superfacet_regions_label)
    superfacet_triangles = np.array(superfacet_triangles, dtype=object)
 
    mask_ = np.zeros((len(centroids),5), dtype=bool)
    for i in range(2):
        #mask_ = np.zeros(len(centroids), dtype=bool)
        sf_label = np.where(superfacet_regions_label==i)[0]
        tri_label = superfacet_triangles[sf_label]
        for j in tri_label:
            mask_[j,i] = 1
    
    # black (plane)
    rgb = np.ones((len(centroids),3))*0.15
    rgb[mask_[:,0]] = [0./256,76./256,153./256] # blue (quadric)
    rgb[mask_[:,1]] = [153./256,255./256,153./256] # green (blending)
    rgb[mask_[:,2]] = [255./256,153./256,153./256] # reed (irregularity)
    #rgb[mask_[:,4]] = [153./256,204./256,255./256] # blu (cones)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

def superfacet_region_probability(shape_label, centroids, face_graph_triangles):
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
    # black (plane) # 
    rgb[mask_[:,0]] = [0./256,76./256,153./256] # blue (quadric)
    rgb[mask_[:,1]] = [153./256,71./256,246./256] # purple (sphere)
    rgb[mask_[:,2]] = [153./256,255./256,153./256] # green (cylinder)
    rgb[mask_[:,3]] = [255./256,153./256,153./256] # reed (other quad. surface)
    rgb[mask_[:,4]] = [153./256,204./256,255./256] # blu (cones)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])