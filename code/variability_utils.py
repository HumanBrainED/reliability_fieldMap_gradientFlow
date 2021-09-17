import numpy as np
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

# Circular colormap for vector angles. Warm and cold colors for positive and negative change.
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def vector_cmap():
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [ c('lightskyblue'), c('white'),.125, c('white') ,c('pink'),
         .25,c('pink'),c('fuchsia'),.31,c('fuchsia'),c('deeppink'),c('darkred'),.374,c('red'),
         c('darkorange'),c('orange'),.5,c('orange'), c('navajowhite'), c('white'),
         .625,c('white'), c('lightgreen'),.75,c('lightgreen') ,c('springgreen'), c('green'),
         c('darkgreen'),.875, c('blue'),c('mediumblue'),c('darkblue'), c('deepskyblue')])   
    return rvb

def vector_cmap_p():
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [ c('lightskyblue'), c('white'),.125, c('white') ,c('pink'),
         .25,c('pink'),c('fuchsia'),.31,c('fuchsia'),c('deeppink'),c('darkred'),.374,c('red'),
         c('darkorange'),c('orange'),.5,c('orange'), c('navajowhite'), c('white'),
         .625,c('white'), c('lightgreen'),.75,c('lightgreen') ,c('springgreen'), c('green'),
         c('darkgreen'),.875, c('blue'),c('mediumblue'),c('darkblue'), c('deepskyblue')])   
    return rvb

def vector_cmap_n():
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
        [ c('lightskyblue'), c('white'),.125, c('white') ,c('pink'),
         .25,c('pink'),c('fuchsia'),.31,c('fuchsia'),c('deeppink'),c('darkred'),.374,c('red'),
         c('darkorange'),c('orange'),.5,c('orange'), c('navajowhite'), c('white'),
         .625,c('white'), c('lightgreen'),.75,c('lightgreen') ,c('springgreen'), c('green'),
         c('darkgreen'),.875, c('blue'),c('mediumblue'),c('darkblue'), c('deepskyblue')])   
    return rvb

# Create colormap of Yeo 7 network colors
def get_yeo_colors():
    yeo_colors = np.array([
        (120,18,134),
        (70,130,180),
        (0,118,14),
        (196,58,250),
        (220,248,164),
        (230,148,34),
        (205,62,78),
        (0,0,0)
    ], dtype=float)
    yeo_colors /=255.
    return yeo_colors

# Load Yeo 7 network assignments to Glasser 360 parcellation
def get_yeo_parcels():
    lhparcels = pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchlh.csv').values[1:,2]
    rhparcels = pd.read_csv('../misc/Yeo7_to_Glasser360_labels/181Yeo7matchrh.csv').values[1:,2]
    allparcels = np.r_[lhparcels,rhparcels]
    return allparcels

# Get circular mean of angles
def getCircularMean(angles):
    n = len(angles)
    sineMean = np.divide(np.nansum(np.sin(np.radians(angles))), n)
    cosineMean = np.divide(np.nansum(np.cos(np.radians(angles))), n)
    vectorMean = np.arctan2(sineMean, cosineMean)
    return np.degrees(vectorMean)

# Upper triangle to symmetric matrix
def array2mat(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.triu_indices(nodes,1)] = data
    mat += mat.T
    matmask = np.zeros([nodes,nodes])
    matmask[mat<=0] = np.nan
    matmask[mat>0] = 1
    return mat,matmask

# Lower triangle to symmetric matrix (sometimes needed for R/MATLAB arrays)
def array2mat_tril(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.tril_indices(nodes,-1)] = data
    mat += mat.T
    matmask = np.zeros([nodes,nodes])
    matmask[mat<=0] = np.nan
    matmask[mat>0] = 1
    return mat,matmask

# Load ICC colormap
def icc_cmap():
    from matplotlib.image import imread
    from matplotlib.colors import LinearSegmentedColormap
    img = imread('../misc/cbars/ICC_CBAR.png')
    # img is 30 x 280 but we need just one col
    colors_from_img = img[:, 0, :]
    # commonly cmpas have 256 entries, but since img is 280 px => N=280
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors_from_img, N=280)
    def reverse_colourmap(cmap, name = 'my_cmap_r'):
        reverse = []
        k = []   
        for key in cmap._segmentdata:    
            k.append(key)
            channel = cmap._segmentdata[key]
            data = []
            for t in channel:                    
                data.append((1-t[0],t[2],t[1]))            
            reverse.append(sorted(data))    
        LinearL = dict(zip(k,reverse))
        my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
        return my_cmap_r
    my_cmap_r = reverse_colourmap(my_cmap)
    return my_cmap_r

# Load ICC model output data into dict
def load_data(task,csvDict):
    data = {}
    for varName in csvDict.keys():
        data[varName] = pd.read_csv(csvDict[varName]).values[:,1]
        data[varName][(data[varName].all() > 1) or (data[varName].all() < 0)] = np.nan
        # Get ratio of variation to total variation:
        # Failed inter-individual variation edges:
    interRatio = data['rawb']/data['vartotal']
    interRatio[interRatio>1] = np.nan
    interMask = np.where(~np.isnan(interRatio)==True)[0] # Mask of failed inter-individual variation failed edges
    # Failed intra-individual variation edges:
    intraRatio = data['raww']/data['vartotal']
    intraRatio[intraRatio>1] = np.nan
    intraMask = np.where(~np.isnan(intraRatio)==True)[0] # Mask of failed intra-individual variation failed edges
    # Mask of all failed edges including intra- and inter-individual variation:
    data['totmask'] = np.intersect1d(interMask,intraMask)
    # Mask of all failed edges including intra- and inter-individual variation:
    data['totmask'] = np.intersect1d(interMask,intraMask)
    return data

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
#             if outpath:
#                 plt.savefig('%s/%s_%s_%s.png' % (outpath,plotname,side,view),dpi=300)
#             plt.close()

def plot_surface_comparisons(taskcombos,data,parcellation, surfaces, numparcels, alpha, darkness, data_range,cmap,outpath):
    import sys
    import cifti
    import matplotlib.pyplot as plt
    sys.path.append('../code')
    from gradient_flow_vectors import calc_icc_vectors_mean, convertAngle,calc_icc_vectors
    lsurf = surfaces[0]
    rsurf = surfaces[1]
    for taskcombo in taskcombos:
        for posNeg in ['positive','negative']:
            task1 = taskcombo[0]
            task2 = taskcombo[1]
            # Vector angles:
            mask1 = data[task1]['totmask']
            mask2 = data[task2]['totmask']
            bothMask = np.intersect1d(mask1,mask2)
            icc0 = np.nanmean(array2mat(data[task1]['icc'],447),0)[0]
            icc1 = np.nanmean(array2mat(data[task2]['icc'],447),0)[0]
            x0 = np.nanmean(array2mat(data[task1]['raww'],447),0)[0]
            y0 = np.nanmean(array2mat(data[task1]['rawb'],447),0)[0]
            x1 = np.nanmean(array2mat(data[task2]['raww'],447),0)[0]
            y1 = np.nanmean(array2mat(data[task2]['rawb'],447),0)[0]
    #             df = calc_icc_vectors_mean(x0,y0,x1,y1,icc0,icc1,task1,task2)
            df = calc_icc_vectors(x0,y0,x1,y1,icc0,icc1,task1,task2)

            plotname =  '%s-%s_%s_vectors' % (task2,task1,posNeg)

            converted_angles = np.array([convertAngle(df['theta0'][i],df['xdiff'][i]) for i in range(len(df['theta0']))])

            angVerts = parcel2vert(parcellation,converted_angles)
            posNegMask = parcel2vert(parcellation,icc1-icc0)
            meandICC = np.mean(posNegMask,0)
            ###################
            # Do this better: #
            ###################
            if posNeg == 'negative':
                angVerts[(angVerts>=45) & (angVerts<=225)] = 45
                angVerts[angVerts==0] = 45
                meandICC[meandICC> 0]  = 0
            elif posNeg == 'positive':
                angVerts[angVerts>=225] = 45
                angVerts[angVerts<=45] = 45
                angVerts[angVerts==0] = 45
                angVerts[np.isnan(angVerts)] = 45
                meandICC[meandICC< 0]  = 0

            # Plot angles:
            numVertices = int(angVerts.shape[1]/2.)
            symmetric_cmap = False
            plot_surface(angVerts,lsurf,rsurf,numVertices,data_range,cmap,alpha,
                         darkness,symmetric_cmap,False,outpath,'%s-%s_%s_dICC_angle' % (task2,task1,posNeg))
            # ICC Diff:
            examplecifti, (ax1,ax2) = cifti.read('../misc/surfaces/100206.sulc.10k_fs_LR.dscalar.nii')
            examplecifti = np.reshape(np.mean(posNegMask,0),[1,posNegMask.shape[1]])
            iccDiffVector = np.reshape(meandICC,[1,posNegMask.shape[1]])
            numVertices = int(iccDiffVector.shape[1]/2.)
            symmetric_cmap = True
            plot_surface(iccDiffVector,lsurf,rsurf,numVertices,(-.2,.2),'bwr',alpha,
                         darkness,symmetric_cmap,True,outpath,'%s-%s_%s_iccDiff_070121' % (task2,task1,posNeg))
    #         cifti.write('%s/%s-%s_iccDiff_070121.dscalar.nii' % (outpath,task2,task1),examplecifti,[ax1,ax2])

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