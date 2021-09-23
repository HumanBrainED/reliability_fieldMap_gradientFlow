import numpy as np
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from scipy import stats
from seaborn.utils import iqr, _kde_support, remove_na

from six import string_types
try:
    import statsmodels.nonparametric.api as smnp
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
    
def load_data(tasks):
    data = {}
    taskname = {}
    for task in tasks:
#         data[task] = {'raww':[],'rawb':[],'icc':[],'vartotal':[],'totmask':[],'edges':[]}
        data[task] = {'raww':[],'rawb':[],'icc':[],'vartotal':[],'totmask':[]}
        data[task]['raww'] = pd.read_csv('../tutorial/example_data/icc_output/%s/%s_within_vector.csv' % (task,task)).values[:,1]
        data[task]['rawb'] = pd.read_csv('../tutorial/example_data/icc_output/%s/%s_between_vector.csv' % (task,task)).values[:,1]
        data[task]['icc'] = pd.read_csv('../tutorial/example_data/icc_output/%s/%s_icc_vector.csv' % (task,task)).values[:,1]
        data[task]['vartotal'] = pd.read_csv('../tutorial/example_data/icc_output/%s/%s_vartotal_vector.csv' % (task,task)).values[:,1]
#         data[task]['edges'] = np.loadtxt('../data/hcp_trt/icc_input/%s/y.txt' % (task))
        taskname[task] = task
        between_ratio = data[task]['rawb']/data[task]['vartotal']
        between_ratio[between_ratio>1] = np.nan
    #     between_utri = np.asarray(between_ratio[np.triu_indices(len(between_ratio),1)])
        bmask = np.where(~np.isnan(between_ratio)==True)[0]
        within_ratio = data[task]['raww']/data[task]['vartotal']
        within_ratio[within_ratio>1] = np.nan
    #     within_utri = np.asarray(within_ratio[np.triu_indices(len(within_ratio),1)])
        wmask = np.where(~np.isnan(within_ratio)==True)[0]
        data[task]['totmask'] = np.intersect1d(bmask,wmask)
    return data

def load_network_data(data,tasks,net_conds,networkcolors,allparcels):
    
    netmat = {}
    for net_cond in net_conds:
        netmat[net_cond] = {}
        for cond in tasks:
            netmat[net_cond][cond] = {}
            for num,net in enumerate(['Visual','Somatomotor',
                                      'Dorsal Attention', 'Ventral Attention', 
                                      'Limbic','Frontoparietal','Default']):
                netmat[net_cond][cond][net] = {'within':{},'between':{},'totmask':{},'icc':{}}

    for net_cond in netmat.keys():
        for cond in tasks:
            mats = {}
            results = {}
            csvdir = '../data/hcp_trt/icc_output/%s' % (cond)
            for metric in ['icc', 'within', 'between', 'vartotal']:
                results[metric] = pd.read_csv('%s/%s_%s_vector.csv' % (csvdir, cond, metric)).values[:,1]
                mats[metric] = np.zeros([360,360])

                mats[metric][np.triu_indices(360,k=1)] = results[metric]
                mats[metric] = mats[metric].T
                mats[metric][np.triu_indices(360,k=1)] = results[metric]

            for num,net in enumerate(['Visual','Somatomotor','Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal','Default']):
                colork = networkcolors[net]
                parcelnum = num + 1
                netmask = np.where(allparcels == parcelnum)[0]
                netmask = np.intersect1d(netmask,
                                         data[cond]['totmask'])
                netbetweenmask = np.where(allparcels!=parcelnum)[0]
                netbetweenmask = np.intersect1d(netbetweenmask,
                                               data[cond]['totmask'])
                if net_cond == 'within':
                    # Get diagonal of networks:
                    netmat[net_cond][cond][net]['within'] = mats['within'][netmask,:]
                    netmat[net_cond][cond][net]['within'] = netmat[net_cond][cond][net]['within'][:,netmask]
                    netmat[net_cond][cond][net]['between'] = mats['between'][netmask,:]
                    netmat[net_cond][cond][net]['between'] = netmat[net_cond][cond][net]['between'][:,netmask]
                    netmat[net_cond][cond][net]['icc'] = mats['icc'][netmask,:]
                    netmat[net_cond][cond][net]['icc'] = netmat[net_cond][cond][net]['icc'][:,netmask]
                    # Get upper triangle of network diagonals:
                    netmat[net_cond][cond][net]['between'] = np.asarray(netmat[net_cond][cond][net]['between'][np.triu_indices(len(netmat[net_cond][cond][net]['between']),1)])
                    netmat[net_cond][cond][net]['within'] = np.asarray(netmat[net_cond][cond][net]['within'][np.triu_indices(len(netmat[net_cond][cond][net]['within']),1)])
                    netmat[net_cond][cond][net]['icc'] = np.asarray(netmat[net_cond][cond][net]['icc'][np.triu_indices(len(netmat[net_cond][cond][net]['icc']),1)])
                else:
    #                     print('between')
                    # Get row of network:
                    netmat[net_cond][cond][net]['within'] = mats['within'][netmask,:]
                    netmat[net_cond][cond][net]['between'] = mats['between'][netmask,:]
                    netmat[net_cond][cond][net]['icc'] = mats['icc'][netmask,:]
                    # Get column values of network NOT within network:
                    netmat[net_cond][cond][net]['within'] = netmat[net_cond][cond][net]['within'][:,netbetweenmask].flatten()
                    netmat[net_cond][cond][net]['between'] = netmat[net_cond][cond][net]['between'][:,netbetweenmask].flatten()
                    netmat[net_cond][cond][net]['icc'] = netmat[net_cond][cond][net]['icc'][:,netbetweenmask].flatten()

            # Create mask of within/between variation values:
            bmask = np.where(netmat[net_cond][cond][net]['between']>=0)[0]
            wmask = np.where(netmat[net_cond][cond][net]['within']>=0)[0]
            # Get common values and mask out nan/0 and mismatched values:
            totmask = np.intersect1d(bmask,wmask)
            netmat[net_cond][cond][net]['within'] = netmat[net_cond][cond][net]['within'][totmask]
            netmat[net_cond][cond][net]['between'] = netmat[net_cond][cond][net]['between'][totmask]
            netmat[net_cond][cond][net]['icc'] = netmat[net_cond][cond][net]['icc'][totmask]
    return netmat
    
# Calculate KDE for field maps:
def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T, bw_method=bw)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = ("Cannot specify a different bandwidth for each dimension "
               "with the scipy backend. You should install statsmodels.")
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z

# KDE plot function:
def _bivariate_kdeplot(xx1, yy1, z1scale, filled, fill_lowest,
                       kernel, bw, gridsize, cut, clip,
                       axlabel, cbar, cbar_ax, cbar_kws, ax, **kwargs):
    from seaborn.palettes import color_palette, light_palette, dark_palette, blend_palette
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]
    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)
    scout, = ax.plot([], [])
    default_color = scout.get_color()
    scout.remove()
    color = kwargs.pop("color", default_color)
#     cmap = kwargs.pop("cmap", None)
    cmap = cbar_kws['cmap']
    label = kwargs.pop("label", None)
    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx1, yy1, z1scale, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels
    if cbar:
        cbar_kws = {} if cbar_kws is None else cbar_kws
        ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)
    # Label the axes
    if hasattr(xx1, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(yy1, "name") and axlabel:
        ax.set_ylabel(y.name)
    if label is not None:
        legend_color = cmap(.95) if color is None else color
        if filled:
            ax.fill_between([], [], color=legend_color, label=label)
        else:
            ax.plot([], [], color=legend_color, label=label)
    return ax


def single_fieldmap(cond1w,cond1b,t1color,outpath,lines):  
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
#     from .reliability_plot_functions import _bivariate_kdeplot 
#     from .reliability_plot_functions import _scipy_bivariate_kde 
    bw='scott'
    gridsize=300
    cut=10
    clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    shade=True
    filled=True
    fill_lowest=False
    xx1, yy1, z1 = _scipy_bivariate_kde(cond1w, cond1b, bw, gridsize, cut, clip)
    scaler = float(1000)
    z1scale = scaler*z1/np.sum(z1)
    normalized = (z1scale-np.min(z1scale))/(np.max(z1scale)-np.min(z1scale))
    shade=True
    vertical=False
    kernel="gau",
    bw="scott"
    gridsize=300
    cut=10
    clip=None
    legend=True
    cumulative=False
    shade_lowest=False
#     cbar=True
    cbar=False
    cbar_ax=None
    cbar_kws={'cmap':t1color}
    our_cmap = plt.get_cmap(t1color)
    cmap_max = 1.01
    norm = mcolors.Normalize(vmin=0, vmax=cmap_max)    
    proxy_mappable = mpl.cm.ScalarMappable(cmap=our_cmap, norm=norm)
    proxy_mappable.set_array(normalized)   

    sns.set_style('white')    
    plt.figure(figsize=(12,10))
    ax=plt.gca()
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 1
    sns.set(font_scale=3)
    ax.axes.set_xlim([0,0.02])
    ax.axes.set_ylim([0,0.02])
#     ax.axes.set_xlim([0,0.1])
#     ax.axes.set_ylim([0,0.1])
    plt.xticks(fontweight='bold',fontsize=20)
    plt.yticks(fontweight='bold',fontsize=20)
#     ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, shade_lowest, kernel, bw, gridsize, cut, clip, legend, cbar, cbar_ax, cbar_kws, ax,vmin=0,vmax=cmap_max,levels=np.arange(0,cmap_max+.5,0.25))
    ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, shade_lowest, kernel, bw, gridsize, cut, clip, legend, cbar, cbar_ax, cbar_kws, ax,vmin=0,vmax=cmap_max,levels=5,alpha=0.5)
    if lines == 'True':
        ax.plot([1,0],[1,0],color='black',alpha=0.3)
        for iccline in [0.2,0.4,0.6,0.8]:
            ax.plot([1,0],[iccline,0],color='black',alpha=0.3)
            ax.plot([iccline,0],[1,0],color='black',alpha=0.3)   
    ax = plt.contour(xx1,yy1,normalized,5, colors = contourcolors[cond1])
    plt.colorbar(proxy_mappable, boundaries=np.arange(0,cmap_max,.1), spacing='proportional', orientation='vertical', pad=.01)
    plt.savefig(outpath)
    plt.show()

# Convert all angles from 0-90 to 0-360
def ang2deg(df):
    numnodes0 = 1 # Parcel-wise discriminability so only 1 array of 360
    numnodes1 = len(df['theta0'])
    for kk in ['x_star','y_star','xdiff','ydiff','theta0']:
        df[kk] = np.reshape(df[kk],[-1,numnodes1])
    theta_deg = np.zeros([numnodes0,numnodes1])
    task1 = df['task1name']
    task2 = df['task2name']
    for i in np.arange(0,numnodes0,1):
        for j in np.arange(0,numnodes1,1):
            x0_n = df['x_star'][i,j]; y0_n = df['y_star'][i,j]
            xdiff = df['xdiff'][i,j]; ydiff = df['ydiff'][i,j]
            xstart = x0_n+i-x0_n; ystart = y0_n+j-y0_n
            newang = df['theta0'][i,j]
            theta_deg[i,j] = newang - 180*(np.min([np.sign(xdiff),0]))
            if theta_deg[i,j] < 0:
                theta_deg[i,j] += 360
    return theta_deg

# Calculate within and between distance difference and angle from discriminability.
def calc_disc_vectors(x0,y0,x1,y1,disc0,disc1,task1name,task2name):
    ddiff = disc1-disc0
    x_star = np.zeros(x0.shape)
    y_star = np.zeros(y0.shape)
    xdiff = x1-x0
    ydiff = y1-y0
    newang = np.degrees(np.arctan(ydiff/xdiff))
#     theta0 = andg2deg(newang)
    
    df = {'x0': x0,
          'y0': y0,
          'x1': x1,
          'y1': y1,
          'x_star':x_star,
    'y_star': y_star,
    'xdiff': xdiff,
    'ydiff': ydiff,
    'disc': ddiff,
    'theta0': newang,
    'task1name': task1name,
    'task2name': task2name}
    return df

# Calculate normalized vectors for ICC difference across tasks.
def calc_icc_vectors(x0,y0,x1,y1,icc0,icc1,task1name,task2name):
    import math
    if icc0 is None and icc1 is None:
        icc0 = b0/(b0+w0)
        icc1 = b1/(b1+w1)
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
    else:
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
    vv = np.sqrt((x1-x0)**2+(y1-y0)**2)
    angle0 = np.arctan(y0/x0)
    rot = math.radians(45) - angle0
    # Rotate vector by angle of ICC:
    # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
    # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
    x0_n = np.cos(rot)*x0 - np.sin(rot)*y0
    y0_n = np.sin(rot)*x0 + np.cos(rot)*y0
    x1_n = np.cos(rot)*x1 - np.sin(rot)*y1
    y1_n = np.sin(rot)*x1 + np.cos(rot)*y1
    xdiff = x1_n-x0_n; ydiff = y1_n-y0_n
    newang = np.degrees(np.arctan(ydiff/xdiff))
#     theta0 = ang2deg(newang)
    
    df = {'x0': x0,
          'y0': y0,
          'x1': x1,
          'y1': y1,
          'icc0': icc0,
          'icc1': icc1,
          'x_star':x0_n,
    'y_star': y0_n,
    'xdiff': xdiff,
    'ydiff': ydiff,
    'dICC': dICC2,
    'theta0': newang,
    'task1name': task1name,
    'task2name': task2name}
    return df

def calc_icc_vectors_mean(x0,y0,x1,y1,icc0,icc1,task1name,task2name):
    import math
    if icc0 is None and icc1 is None:
        icc0 = b0/(b0+w0)
        icc1 = b1/(b1+w1)
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
    else:
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
    vv = np.sqrt((x1-x0)**2+(y1-y0)**2)
    angle0 = np.arctan(y0/x0)
    rot = math.radians(45) - angle0
    # Rotate vector by angle of ICC:
    # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
    # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
    x0_n = np.cos(rot)*x0 - np.sin(rot)*y0
    y0_n = np.sin(rot)*x0 + np.cos(rot)*y0
    x1_n = np.cos(rot)*x1 - np.sin(rot)*y1
    y1_n = np.sin(rot)*x1 + np.cos(rot)*y1
    xdiff = x1_n-x0_n; ydiff = y1_n-y0_n
#     newang = np.degrees(np.arctan(np.nanmean(ydiff,0)/np.nanmean(xdiff,0)))
    newang = np.degrees(np.arctan(np.nanmean(ydiff,0)/np.nanmean(xdiff,0)))
#     theta0 = ang2deg(newang)
    
    df = {'x0': x0,
          'y0': y0,
          'x1': x1,
          'y1': y1,
          'icc0': icc0,
          'icc1': icc1,
          'x_star':x0_n,
    'y_star': y0_n,
    'xdiff': xdiff,
    'ydiff': ydiff,
    'dICC': dICC2,
    'theta0': newang,
    'task1name': task1name,
    'task2name': task2name}
    return df

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

def get_yeo_parcels():
    lhparcels = pd.read_csv('../data/yeo_labels/181Yeo7matchlh.csv').values[1:,2]
    rhparcels = pd.read_csv('../data/yeo_labels/181Yeo7matchrh.csv').values[1:,2]
    allparcels = np.r_[lhparcels,rhparcels]
    return allparcels

# Function to plot vectors
def vector_plot(df, alpha, vector_type, task1,task2, yeo_colors, allparcels, outname):
    import matplotlib.pyplot as plt
    cond0 = df['task1name']
    cond1 = df['task2name']
    icc0 = df['icc0']
    icc1 = df['icc1']
    numnodes = df['theta0'].shape[0]
    fig, ax = plt.subplots(figsize=(10,10),dpi=300)  
    ax.set_facecolor('xkcd:white')
    for i in range(numnodes):
        if vector_type == 'raw':
            # Non-normalized vectors so we remove the "optimal change" and plot "raw change":
            xstart = df['x0'][i]; xend = df['x1'][i]
            ystart = df['y0'][i]; yend = df['y1'][i]
            ax.set_xlim([0,0.02])
            ax.set_ylim([0,0.02])
            xticks = np.arange(0,0.0201,0.0025)
            yticks = np.arange(0,0.0201,0.0025)
        elif vector_type == 'norm':
            xstart = df['x0'][i]; xend = df['x_star'][i]+df['xdiff'][i]
            ystart = df['y0'][i]; yend = df['y_star'][i]+df['ydiff'][i]
            ax.set_xlim([0,0.02])
            ax.set_ylim([0,0.02])
            xticks = np.arange(0,0.0201,0.0025)
            yticks = np.arange(0,0.0201,0.0025)
        elif vector_type == 'norm_0':
            xstart = 0; xend = df['xdiff'][i]
            ystart = 0; yend = df['ydiff'][i]
            ax.set_xlim([-0.01,0.01])
            ax.set_ylim([-0.01,0.01])
            xticks = np.round(np.arange(-0.01,0.01,0.005),3)
            yticks = np.round(np.arange(-0.01,0.01,0.005),3)
            
        xdiff = df['xdiff'][i]
        ydiff = df['ydiff'][i]
        scale = df['dICC'][i]
        degrees = df['theta0'][i]
        if np.isnan(degrees) == False:
            if icc1[i] - icc0[i] <= 0:
                color_head = yeo_colors[allparcels[i]-1]
                color_tail = yeo_colors[allparcels[i]-1]
            else:
                color_tail = yeo_colors[allparcels[i]-1]
                color_head = yeo_colors[allparcels[i]-1]
            prop_head = dict(arrowstyle = "-|>,head_width=0.3,head_length=.4",
                      shrinkA=5,shrinkB=5,color = color_head, 
                           alpha=alpha,linewidth=1)
            prop_line = dict(arrowstyle="wedge,tail_width=1, shrink_factor=5", 
                           shrinkA=5, shrinkB=5, color = color_tail,
                           alpha=alpha, width = 1)
            prop_head = dict(arrowstyle = "-|>,head_width=0.4,head_length=.8",
                      shrinkA=5,shrinkB=5,color = color_head, 
                           alpha=alpha,linewidth=3)
            prop_line = dict(arrowstyle="wedge,tail_width=3, shrink_factor=5", 
                           shrinkA=5, shrinkB=5, color = color_tail,
                           alpha=alpha, width = 2)
            arr1 = ax.annotate("", xy=(xend, yend), xycoords = 'data', 
                  xytext=(xstart, ystart), textcoords='data',
                  arrowprops=prop_head, annotation_clip=False)
            arr1.arrow_patch.set_clip_box(ax.bbox)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=15)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=15)
    plt.plot([-1,1],[-1,1],color = 'gray',linestyle = '--')
    # plt.title('%s-%s' % (cond1,cond0))
    # if outname:
    plt.savefig(outname)
    plt.title('%s-%s' % (cond1,cond0))
    if outname == True:
        plt.savefig(outname)
    plt.show()
        
def array2mat(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.triu_indices(nodes,1)] = data
    mat += mat.T
    matmask = np.zeros([nodes,nodes])
    matmask[mat<=0] = np.nan
    matmask[mat>0] = 1
    return mat,matmask

def array2mat_tril(data,nodes):
    mat = np.zeros([nodes,nodes])
    mat[np.tril_indices(nodes,-1)] = data
    mat += mat.T
    matmask = np.zeros([nodes,nodes])
    matmask[mat<=0] = np.nan
    matmask[mat>0] = 1
    return mat,matmask

def icc_cmap():
    from matplotlib.image import imread
    from matplotlib.colors import LinearSegmentedColormap
    img = imread('../figures/ICC_CBAR.png')
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

# Loop through networks:
def network_polar():
    for num,net in enumerate(['Visual','Somatomotor',
    'Dorsal Attention', 'Ventral Attention', 'Limbic', 
    'Frontoparietal','Default']):
#     for num,net in enumerate(['Visual']):
        task0name = '%s' % (cond0)
        task1name = '%s' % (cond1)

        # Parcel index:
        p_ind = np.where(allparcels == num+1)[0]
        netx0 = matx0[:,p_ind].flatten()
        nety0 = maty0[:,p_ind].flatten()
        netx1 = matx1[:,p_ind].flatten()
        nety1 = maty1[:,p_ind].flatten()
        neticc0 = maticc0[:,p_ind].flatten()
        neticc1 = maticc1[:,p_ind].flatten()

        # Calculate vector parameterss
        df = calc_icc_vectors(np.array(netx0),np.array(nety0),np.array(netx1),np.array(nety1),
                              np.array(neticc0),np.array(neticc1),task0name,task1name)
        theta = ang2deg(df)[0]
        theta = theta[~np.isnan(theta)]
        # Setting bins:
        bins = np.arange(0,360,9)
        a = np.histogram(theta,bins)
        # Set frequency:
        height = a[0]/np.sum(a[0])
        deg_ind = np.radians(a[1][1:])
        width = .05
        rmax = .1

        # Plot angular histo:
        ax = plt.subplot(111, projection='polar')
        ax.set_rlim(0, rmax)
        ax.set_rticks(np.round(np.arange(rmax/4., rmax+0.01, rmax/4.),3))
        ax.set_rlabel_position(-90)
        ax.bar(x=deg_ind, height=height, width=width, 
               bottom=0, alpha=0.5, color = yeo_colors[num])
        ax.bar(x=np.radians([45,135,225,315]), height=10, width=0, 
               bottom=0, alpha=1, tick_label=['No\nChange','+ Optimal','No\nChange','- Optimal'], 
               color = 'k', linestyle = '--')
        ax.tick_params(axis='both', which='major', pad=20)
        ax.spines['polar'].set_visible(False)
        plt.title('SOC - REST Concat\n%s' % net,pad=10)
        plt.tight_layout()
        plt.savefig('../figures/polar/%s-%s_%s_bin-5deg.png' % net,dpi=300)
        plt.show()
        
def getCircularMean(angles):
    n = len(angles)
    sineMean = np.divide(np.nansum(np.sin(np.radians(angles))), n)
    cosineMean = np.divide(np.nansum(np.cos(np.radians(angles))), n)
    vectorMean = np.arctan2(sineMean, cosineMean)
    return np.degrees(vectorMean)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Load ICC gradient flow colormap
def icc_gradient_flow_cmap():
    from matplotlib.image import imread
    from matplotlib.colors import LinearSegmentedColormap
    img = imread('../misc/cbars/ICC_gradient_flow_cbar.png')
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

def warm_cold_gradient_flow_cmap():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    ######################################
    # Positive, warm gradient flow cmap: #
    ######################################
    cmap = icc_gradient_flow_cmap()
    # Cmap sections for degrees used to section:
    cmapN = cmap.N
    ind45 = int(cmapN*(45/360))
    ind135 = int(cmapN*(135/360))
    ind180 = int(cmapN*(180/360))
    ind225 = int(cmapN*(225/360))

    # Cut out cold colors (0-45 and 225-360 degrees):
    warm_cmap = truncate_colormap(cmap, 0,1,cmapN)
    warm_cmap = warm_cmap(np.arange(ind45,ind225))
    warm_cmap_colors = warm_cmap

    # Set up white colors:
    white_colors_bottom = plt.cm.Greys_r(np.ones(ind45))
    white_colors_top = plt.cm.Greys_r(np.ones(cmapN - ind225))

    # Combine them and build a new colormap
    warm_cmap_whites = np.vstack((white_colors_bottom,warm_cmap_colors,white_colors_top))
    # warm_cmap_whites = np.vstack((white_colors_bottom,warm_cmap_colors))
    warm_cmap_whites = mcolors.LinearSegmentedColormap.from_list('my_colormap', warm_cmap_whites)

    # Get both sides of cold, negative colors (0-45 and 225-360 degrees):
    # 0-45 degrees:
    cmap = icc_gradient_flow_cmap()
    # cold_cmap_045 = truncate_colormap(cmap, 0,0.125)
    cold_cmap_045 = truncate_colormap(cmap, 0,1)
    cold_cmap_045 = cold_cmap_045(np.arange(0,ind45))
    cold_cmap_045_colors = cold_cmap_045
    # 225-360 degrees
    # cold_cmap_225_360 = truncate_colormap(cmap, 0.625,1)
    cold_cmap_225_360 = truncate_colormap(cmap, 0,1)
    cold_cmap_225_360 = cold_cmap_225_360(np.arange(ind225,cmapN))
    cold_cmap_225_360_colors = cold_cmap_225_360
    white_colors_middle = plt.cm.Greys_r(np.ones(ind180))

    # combine them and build a new colormap
    cold_cmap_whites = np.vstack((cold_cmap_045,white_colors_middle,cold_cmap_225_360))
    cold_cmap_whites = mcolors.LinearSegmentedColormap.from_list('my_colormap', cold_cmap_whites)
    return warm_cmap_whites,cold_cmap_whites



