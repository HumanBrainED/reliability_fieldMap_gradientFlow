import numpy as np
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from scipy import stats
from seaborn.utils import iqr, _kde_support, remove_na

from six import string_types

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


def single_fieldmap(cond1w,cond1b,t1color,taskcmaps,outpath,lines):  
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    
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
    cbar_kws={'cmap':taskcmaps[cond1]}
    our_cmap = plt.get_cmap(taskcmaps[cond1])
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
    plt.xticks(fontweight='bold',fontsize=20)
    plt.yticks(fontweight='bold',fontsize=20)
    ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, shade_lowest, kernel, bw, gridsize, cut, clip, legend, cbar, cbar_ax, cbar_kws, ax,vmin=0,vmax=cmap_max,levels=5,alpha=0.5)
    if lines == 'True':
        ax.plot([1,0],[1,0],color='black',alpha=0.3)
        for iccline in [0.2,0.4,0.6,0.8]:
            ax.plot([1,0],[iccline,0],color='black',alpha=0.3)
            ax.plot([iccline,0],[1,0],color='black',alpha=0.3)   
    ax = plt.contour(xx1,yy1,normalized,5, colors = contourcolors[cond1])
    plt.colorbar(proxy_mappable, boundaries=np.arange(0,cmap_max,.1), spacing='proportional', orientation='vertical', pad=.01)
    if outpath == True:
        plt.savefig(outpath)
    plt.show()