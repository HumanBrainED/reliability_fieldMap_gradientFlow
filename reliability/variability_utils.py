import cifti
import numpy as np
import pandas as pd

# Colormaps
gradientFlowCmaps = np.load('../misc/cmaps/gradientFlowCmap.npy',allow_pickle=True).item()
yeo_colors = np.load('../misc/cmaps/yeoColors.npy',allow_pickle=True)

# Load Yeo 7 network assignments to Glasser 360 parcellation
# Matching done in-house
# post online.
# Change to path input so user can input their own parcellation-to-network files
allparcels = np.r_[pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchlh.csv').values[1:,2],
                   pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchrh.csv').values[1:,2]]

# Load region to vertex mapping to convert 360 parcels to vertices
plabel = '../misc/Glasser2016_labels/HCP_MMP_P210_10k.dlabel.nii'
glasserlabel,(ax1,ax2) = cifti.read(plabel)

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
