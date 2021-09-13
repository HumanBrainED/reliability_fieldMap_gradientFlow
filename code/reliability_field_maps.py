import numpy as np
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
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
    
# Plot a single field map:
def field_map(tasks,data,taskcolors,taskcmaps,alpha,lines,outpath):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    for cond1 in tasks:
        print(cond1)
        t1color = taskcolors[cond1]
        exec('colors1 = plt.cm.%s(np.linspace(0,1,128))' % taskcmaps[cond1])
        cond1w = data[cond1]['raww'].copy()
        cond1b = data[cond1]['rawb'].copy()
        wmask1 = np.where((cond1w != 0) & (cond1w <= 0.015))[0]
        bmask1 = np.where((cond1b != 0) & (cond1b <= 0.015))[0]
        cond1totmask = np.intersect1d(bmask1,wmask1)

        # Mask b/w and w/in values for each condition
    #     bothmask = np.intersect1d(cond1totmask,cond2totmask)
        bothmask = np.intersect1d(data[cond1]['totmask'],data[cond1]['totmask'])
        cond1w = cond1w[bothmask]
        cond1b = cond1b[bothmask]

#         # Setting to top X percentile
#         perc = np.percentile(data[cond1]['icc'][bothmask],percnum)
#         percmask = np.where(data[cond1]['icc'][bothmask]>perc)[0]
#         cond1w = cond1w[percmask]
#         cond1b = cond1b[percmask]

        bw='scott'
        gridsize=100
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
        gridsize=100
        cut=10
        clip=None
        legend=True
        cumulative=False
        shade_lowest=False
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
        ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, shade_lowest, kernel, bw, gridsize, cut, clip, legend, cbar, cbar_ax, cbar_kws, ax,vmin=0,vmax=cmap_max,levels=5,alpha=alpha)
        
        ax.plot([1,0],[1,0],color='black',alpha=0.3)
        for iccline in [0.2,0.4,0.6,0.8]:
            ax.plot([1,0],[iccline,0],color='black',alpha=0.3)
            ax.plot([iccline,0],[1,0],color='black',alpha=0.3)
        if lines == True:
            ax = plt.contour(xx1,yy1,normalized,5, colors = contourcolors[cond1])
        cbar = plt.colorbar(proxy_mappable, boundaries=np.arange(0,cmap_max,.1), spacing='proportional', orientation='vertical', pad=.01)
        cbar.set_label('Density',labelpad=20)
        if outpath == True:
            plt.savefig(outpath)
        plt.show()


# Plot field map for each condition in taskcombos in 1 plot for comparison:
def field_map_overlay(taskcombos,data,taskcolors,taskcmaps,alpha,lines,outpath):
    for taskcombo in taskcombos:
        plt.figure(figsize=(10,10))
        sns.set_style('white')
        ax=plt.gca()
        mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams['font.size'] = 1
        sns.set(font_scale=3)
        ax.axes.set_xlim([0,0.025])
        ax.axes.set_ylim([0,0.025])
        plt.xticks(fontweight='bold',fontsize=20)
        plt.yticks(fontweight='bold',fontsize=20)
        ###

        for cond1 in taskcombo:
            print(cond1)
            t1color = taskcolors[cond1]
            exec('colors1 = plt.cm.%s(np.linspace(0,1,128))' % taskcmaps[cond1])
            cond1w = data[cond1]['raww'].copy()
            cond1b = data[cond1]['rawb'].copy()
            wmask1 = np.where((cond1w != 0) & (cond1w <= 0.015))[0]
            bmask1 = np.where((cond1b != 0) & (cond1b <= 0.015))[0]
            cond1totmask = np.intersect1d(bmask1,wmask1)

            # Mask b/w and w/in values for each condition
            bothmask = np.intersect1d(data[cond1]['totmask'],data[cond1]['totmask'])
            cond1w = cond1w[bothmask]
            cond1b = cond1b[bothmask]
            #### Edit end:

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
            shade=False
            vertical=False
            kernel="gau",
            bw="scott"
            gridsize=300
            cut=10
            clip=None
            legend=True
            cumulative=False
            shade_lowest=False
            cbar=False
            cbar_ax=None
            cbar_kws={'cmap':taskcmaps[cond1]}
            our_cmap = plt.get_cmap(taskcmaps[cond1])
            cmap_max = 1.01
            norm = mcolors.Normalize(vmin=0, vmax=cmap_max)    
            proxy_mappable = mpl.cm.ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(normalized)   
            ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, 
                                    shade_lowest, kernel, bw, gridsize, 
                                    cut, clip, legend, cbar, cbar_ax, cbar_kws, 
                                    ax,vmin=0,vmax=cmap_max,levels=5,alpha=alpha,
                                   linewidths=5)
        if lines == True:
            ax.plot([1,0],[1,0],color='black',alpha=0.3)
            for iccline in [0.2,0.4,0.6,0.8]:
                ax.plot([1,0],[iccline,0],color='black',alpha=0.3)
                ax.plot([iccline,0],[1,0],color='black',alpha=0.3)    
        if outpath == True:
            plt.savefig('../figures/shortpaper/fieldmaps/%s_%s_perc%s_fieldmap_nogsr_front_contour_070121_time_1200-600.png' % (taskcombo[0],taskcombo[1],percnum),dpi=300)
        plt.show()        
        
