import math
import numpy as np 
import matplotlib as mpl

# Calculate normalized vectors for ICC difference across tasks.
def calc_icc_vectors(x0,y0,icc0,x1,y1,icc1,task1name,task2name):
    
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
    ang = np.degrees(np.arctan(ydiff/xdiff))
    
    # Convert angles from 0 to 180 and 0 to -180 to 0-360:
    newAngle = ang - len(xdiff)*[180]*np.minimum(0,np.sign(xdiff))
    newAngle[newAngle<0] += 360
    
    df = {'x0': x0,
          'y0': y0,
          'x1': x1,
          'y1': y1,
          'icc0': icc0,
          'icc1': icc1,
          'x0_star':x0_n,
          'y0_star': y0_n,
          'x1_star':x1_n,
          'y1_star':y1_n,
          'xdiff': xdiff,
          'ydiff': ydiff,
          'dICC': dICC2,
          'theta0': newAngle,
          'task1name': task1name,
          'task2name': task2name}
    return df

def pah(theta,bin_threshold,vector_cmap,title,outpath):
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.rcParams["axes.edgecolor"] = "0.15"
    mpl.pyplot.rcParams["axes.linewidth"]  = 1.25
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
    ax = mpl.pyplot.subplot(111, projection='polar')
    ax.set_rlim(0, rmax)
    ax.set_rticks(np.round(np.arange(rmax/4., rmax+0.01, rmax/4.),3))
    ax.set_rlabel_position(-90)
    # Histogram portion:
    ax.bar(x=deg_ind, height=height, width=width, 
           bottom=0, alpha=1, color = rvbColors, edgecolor = 'black',lw=0.2)
    # Lines to show optimal/suboptimal angle direction:
    ax.bar(x=np.radians([45,135,225,315]), height=10, width=0, 
           bottom=0, alpha=1, tick_label=['No\nChange','+ Optimal','No\nChange','- Optimal'], 
           color = 'k')
    ax.tick_params(axis='both', which='major', pad=20)
    ax.spines['polar'].set_visible(False)
    if title:
        mpl.pyplot.title(title,pad=10)
    mpl.pyplot.tight_layout()
    if outpath:
        mpl.pyplot.savefig('%s' % (outpath),dpi=300)
    mpl.pyplot.show()


            