import numpy as np 
import matplotlib.pyplot as plt
from variability_utils import *

# take out matrix to outside func()
# Convert all angles from 0-90 to 0-360
# def ang2deg(df):
#     numnodes0 = 1 # Parcel-wise discriminability so only 1 array of 360
#     numnodes1 = len(df['theta0'])
#     for kk in ['x_star','y_star','xdiff','ydiff','theta0']:
#         df[kk] = np.reshape(df[kk],[-1,numnodes1])
#     theta_deg = np.zeros([numnodes0,numnodes1])
#     task1 = df['task1name'] # delete list
#     task2 = df['task2name'] # delete list
#     for i in np.arange(0,numnodes0,1):
#         for j in np.arange(0,numnodes1,1):
#             x0_n = df['x_star'][i,j]; y0_n = df['y_star'][i,j]
#             xdiff = df['xdiff'][i,j]; ydiff = df['ydiff'][i,j]
#             xstart = x0_n+i-x0_n; ystart = y0_n+j-y0_n
#             newang = df['theta0'][i,j]
#             theta_deg[i,j] = newang - 180*(np.min([np.sign(xdiff),0]))
#             if theta_deg[i,j] < 0:
#                 theta_deg[i,j] += 360
#     return theta_deg

def convertAngle(angle,xdiff):
    newAngle = angle - 180*(np.min([np.sign(xdiff),0]))
    if newAngle < 0:
        newAngle += 360
    return newAngle

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


# Gradient flow angular histogram plot:
def gradient_flow_histogram(tasks,data,vector_type,num_parc,
                            bin_threshold,vector_cmap,title,outpath):
    
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    for ind1,cond0 in enumerate(tasks):
        for ind2,cond1 in enumerate(tasks[ind1+1:len(tasks)]):
            task0name = cond0
            task1name = cond1
            # Read in data:
            icc1 = data[cond1]['icc'].astype(float)[:64620]
            icc0 = data[cond0]['icc'].astype(float)[:64620]
            x1 = data[cond1]['raww'].astype(float)[:64620]
            x0 = data[cond0]['raww'].astype(float)[:64620]
            y1 = data[cond1]['rawb'].astype(float)[:64620]
            y0 = data[cond0]['rawb'].astype(float)[:64620]

            # Parcel-wise so create matrices and masks
            maticc1,icc1mask = array2mat(icc1,num_parc)
            maticc0,icc0mask = array2mat(icc0,num_parc)
            matx1,matx1mask = array2mat(x1,num_parc)
            matx0,matx0mask = array2mat(x0,num_parc)
            maty1,maty1mask = array2mat(y1,num_parc)
            maty0,maty0mask = array2mat(y0,num_parc)

            # Mask containing good edges for both conditions
            finalmask = icc0mask*icc1mask*matx1mask*matx0mask*maty1mask*maty0mask
            maticc1 = maticc1*finalmask
            maticc0 = maticc0*finalmask
            matx1 = matx1*finalmask
            matx0 = matx0*finalmask
            maty1 = maty1*finalmask
            maty0 = maty0*finalmask

            # All upper triangle edges:
            netx0 = matx0[np.triu_indices(num_parc,1)].flatten()
            nety0 = maty0[np.triu_indices(num_parc,1)].flatten()
            netx1 = matx1[np.triu_indices(num_parc,1)].flatten()
            nety1 = maty1[np.triu_indices(num_parc,1)].flatten()
            neticc0 = maticc0[np.triu_indices(num_parc,1)].flatten()
            neticc1 = maticc1[np.triu_indices(num_parc,1)].flatten()

            # Calculate vector parameters:
            df = calc_icc_vectors(np.array(netx0),np.array(nety0),np.array(netx1),np.array(nety1),
                                  np.array(neticc0),np.array(neticc1),task0name,task1name)
            
            theta = np.array([convertAngle(df['theta0'][i],df['xdiff'][i]) for i in range(len(df['theta0']))])
#             theta = ang2deg(df)[0]
            theta = theta[~np.isnan(theta)]

            # Setting bins:
            bins = np.arange(0,361,bin_threshold)
            a = np.histogram(theta,bins)

            # Set frequency:
            height = a[0]/np.sum(a[0])     # bar height
            deg_ind = np.radians(a[1][1:]) # index in radians
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
    return df
            
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
    
    