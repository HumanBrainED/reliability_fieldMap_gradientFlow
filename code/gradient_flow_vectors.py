import numpy as np 
import matplotlib.pyplot as plt
from variability_utils import *

def convertAngle(angle,xdiff):
    newAngle = angle - 180*(np.min([np.sign(xdiff),0]))
    if newAngle < 0:
        newAngle += 360
    return newAngle

# Calculate normalized vectors for ICC difference across tasks.
def calc_icc_vectors(x0,y0,x1,y1,icc0,icc1,task1name,task2name,mean=False):
    import math
    if icc0 is None and icc1 is None:
        icc0 = b0/(b0+w0)
        icc1 = b1/(b1+w1)
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
    else:
        dICC = np.abs(icc1-icc0)
        dICC2 = icc1-icc0
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
    if mean == False:
        newang = np.degrees(np.arctan(ydiff/xdiff))
    else:
        newang = np.degrees(np.arctan(np.nanmean(ydiff,0)/np.nanmean(xdiff,0)))
        
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

def pah(theta,bin_threshold,vector_cmap,title,outpath):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    # Setting bins:
    bins = np.arange(0,361,bin_threshold)
    hist = np.histogram(theta,bins)

    # Set frequency:
    height = hist[0]/np.sum(hist[0])     # bar height
    deg_ind = np.radians(hist[1][1:]) # index in radians
    width = .1                     # linewidth
    rmax = np.max(height)*(1+0.1)  # max height for plot - input for 0.1 (10% more than max height)

    # color list:
    rvbColors = vector_cmap(np.linspace(0, 1, len(deg_ind)))

    # Plot angular histo:
    ax = plt.subplot(111, projection='polar')
    ax.set_rlim(0, rmax)
    ax.set_rticks(np.round(np.arange(rmax/4., rmax+0.01, rmax/4.),3))
    ax.set_rlabel_position(-90)
    ax.bar(x=deg_ind, height=height, width=width, 
           bottom=0, alpha=1, color = rvbColors, edgecolor = 'black',lw=0.2)
    ax.bar(x=np.radians([45,135,225,315]), height=10, width=0, 
           bottom=0, alpha=1, tick_label=['No\nChange','+ Optimal','No\nChange','- Optimal'], 
           color = 'k')
    ax.tick_params(axis='both', which='major', pad=20)
    ax.spines['polar'].set_visible(False)
    if title:
        plt.title(title,pad=10)
    plt.tight_layout()
    if outpath:
        plt.savefig('%s/%s_%s_gradient_flow_histogram.png' % (outpath,cond1,cond0),dpi=300)
    plt.show()

            
# Gradient flow angles plotting change in individual variation between tasks:
def plot_gradient_flow_vector(df, alpha, vector_type, task1,task2, yeo_colors, allparcels, outname):
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
    
    