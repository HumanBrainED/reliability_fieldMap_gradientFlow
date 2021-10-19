import os
import cifti
import numpy as np
import pandas as pd
from brainspace.mesh.mesh_io import read_surface

fpath = os.path.dirname(os.path.abspath(__file__))

# Colormaps
gradientFlow_cmap = np.load(fpath + '/misc/cmaps/gradientFlow_cmap.npy',allow_pickle=True).item()
ICC_cmap = np.load(fpath + '/misc/cmaps/ICC_cmap.npy',allow_pickle=True).item()

# Load Yeo 7 network assignments to Glasser 360 parcellation
allparcels = np.r_[pd.read_csv(fpath + '/misc/Glasser2016_labels/181Yeo7matchlh.csv').values[1:,2],
                   pd.read_csv(fpath + '/misc/Glasser2016_labels/181Yeo7matchrh.csv').values[1:,2]]

# Load region to vertex mapping to convert 360 parcels to vertices
plabel = fpath + '/misc/Glasser2016_labels/HCP_MMP_P210_10k.dlabel.nii'
glasserlabel,(ax1,ax2) = cifti.read(plabel)

# Load 10k gifti surfaces:
lsurf = read_surface(fpath + '/misc/surfaces/Conte69.L.very_inflated.10k_fs_LR.surf.gii')
rsurf = read_surface(fpath + '/misc/surfaces/Conte69.R.very_inflated.10k_fs_LR.surf.gii')

def array2mat(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.triu_indices(nodes,1)] = data
    mat += mat.T
    return mat

def parcel2vert(glasserlabel,theta_img):
    numverts = glasserlabel.shape[1]
    if len(theta_img.shape) >1:
        numparcels = len(theta_img)
    else:
        numparcels = 1
        theta_img = np.reshape(theta_img,[-1,len(theta_img)])
    nparc = np.arange(numparcels)
    data = np.zeros([numparcels,numverts])
    for parcel in nparc:
        for plabel in range(int(np.max((glasserlabel)))):
            p_idx = np.where(glasserlabel[0,:]==plabel+1)[0]
            data[parcel,p_idx] = theta_img[parcel,plabel]
    return data
