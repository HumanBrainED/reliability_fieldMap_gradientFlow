import numpy as np
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

# Colormaps
gradientFlowCmaps = np.load('../misc/cmaps/gradientFlowCmap.npy',allow_pickle=True).item()
yeo_colors = np.load('../misc/cmaps/yeoColors.npy',allow_pickle=True)

# Load Yeo 7 network assignments to Glasser 360 parcellation
# Matching done in-house
# post online.
# Change to path input so user can input their own parcellation-to-network files
allparcels = np.r_[pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchlh.csv').values[1:,2],
                   pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchrh.csv').values[1:,2]]


def plot_surface(vdata,lsurf,rsurf,numVertices,data_range,cmap,alpha,darkness,cbar,symmetric_cmap,outpath,plotname):
    vmin,vmax = data_range
    for side in ['left','right']:
        if side == 'left':
            surf_array = vdata[0,:numVertices]
            surf = lsurf
        else:
            surf_array = vdata[0,numVertices:]
            surf = rsurf
        for view in ['lateral','medial']:
            plt.figure(figsize=(10,15))
            plt.rcParams['axes.facecolor'] = 'white'
            _ = plotting.plot_surf(surf, surf_array, 
                                   hemi=side, view=view,
                                   bg_on_data = True,
                                   cmap = cmap, colorbar=cbar, vmin=vmin, vmax=vmax, 
                                   avg_method='median',alpha=alpha, darkness=darkness,
                                   symmetric_cmap=symmetric_cmap)

def array2mat(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.triu_indices(nodes,1)] = data
    mat += mat.T
    matmask = np.zeros([nodes,nodes])
    matmask[mat<=0] = np.nan
    matmask[mat>0] = 1
    return mat,matmask

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
