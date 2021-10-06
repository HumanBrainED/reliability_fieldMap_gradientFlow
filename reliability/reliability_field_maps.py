import numpy as np
from scipy import stats
import matplotlib as mpl

import seaborn as sns
from seaborn.utils import _kde_support
from six import string_types
#####################
# Why is this here
#####################
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

def plot_field_map(x,y,taskcolor,taskcmap,alpha,lines,outpath,thr=0.0001,gridsize=300,
                     overlay=False,cbar_option=True,figSize=(12,10),xyLim=95,shade=True,addContourLines=True,
                  plotstyle='all',bins=500):
    # Set X,Y lims:
    if int(xyLim):
        xperc = np.percentile(x,xyLim)
        yperc = np.percentile(y,xyLim)
        xy_lim = np.max([xperc,yperc])
        xyVals = (0,xy_lim)
    elif type(xyLim) == 'tuple':
        xyVals = (xyLim[0],xyLim[1])
    
    # KDE options
    bw='scott'
    gridsize=gridsize
    cut=10
    clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    
    # Kde distribution:
    xx1, yy1, z1 = _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Scaling and normalization so that field maps are comparable:
    z1scale = z1/np.sum(z1)
    normalized = (z1scale-np.min(z1scale))/(np.max(z1scale)-np.min(z1scale))
    
    for ps in plotstyle:
        print('Creating %s plot' % ps)
    
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 1
    sns.set(font_scale=3)
    
    if 'kde' in plotstyle or plotstyle == 'all':
        mpl.pyplot.figure(figsize=(figSize[0],figSize[1]))
        sns.set_style('white')
        ax=mpl.pyplot.gca()
        ax.axes.set_xlim([xyVals[0],xyVals[1]])
        ax.axes.set_ylim([xyVals[0],xyVals[1]])
        mpl.pyplot.xticks(fontweight='bold',fontsize=20)
        mpl.pyplot.yticks(fontweight='bold',fontsize=20)
        mpl.pyplot.xlabel('Intra-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        mpl.pyplot.ylabel('Inter-individual Variation',labelpad=20,fontweight='bold',fontsize=20)

        # Set colorbar for scaled density plot:
        cbar_kws={'cmap':taskcmap}
        our_cmap = mpl.pyplot.get_cmap(taskcmap)
        cmap_max = 1.00001
        norm = mpl.colors.Normalize(vmin=0, vmax=cmap_max)    
        proxy_mappable = mpl.cm.ScalarMappable(cmap=our_cmap, norm=norm)
        proxy_mappable.set_array(normalized)  
        
        # KDE plot options:
        legend=True
        cumulative=False
        shade=shade
        shade_lowest=False
        cbar=False
        cbar_ax=None
        filled=True
        fill_lowest=False
        vertical=False
        kernel="gau"   
        
        # KDE plot:
        ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, 
                                shade_lowest, kernel, bw, gridsize, 
                                cut, clip, legend, cbar, cbar_ax, cbar_kws, 
                                ax,vmin=0,vmax=cmap_max,levels=5,alpha=alpha,
                               linewidths=5)
        ax.set_aspect('equal')
        if addContourLines == True:
            ax = mpl.pyplot.contour(xx1,yy1,normalized,5,colors = taskcolor)
        if cbar_option == True:
            cbar = mpl.pyplot.colorbar(proxy_mappable, boundaries=np.arange(0,cmap_max,.1), spacing='proportional', orientation='vertical', pad=.01)
            cbar.set_label('Density',labelpad=20)
       
        if lines == True:
            mpl.pyplot.plot([1,0],[1,0],color='black',alpha=0.3,zorder=0)
            for iccline in [0.2,0.4,0.6,0.8]:
                mpl.pyplot.plot([1,0],[iccline,0],color='black',alpha=0.3)
                mpl.pyplot.plot([iccline,0],[1,0],color='black',alpha=0.3)
        #######################################
        ### Output figure handle not saving ###
        #######################################
        if outpath == True:
            mpl.pyplot.savefig('../figures/shortpaper/fieldmaps/%s_%s_perc%s_fieldmap_nogsr_front_contour_070121_time_1200-600.png' % (taskcombo[0],taskcombo[1],percnum),dpi=300)
        mpl.pyplot.show()   
    
    if 'nokde' in plotstyle or plotstyle == 'all':
        fig, ax = mpl.pyplot.subplots(figsize=(figSize[0],figSize[1]))
        ax.set_facecolor('white')
        mpl.pyplot.hist2d(x, y, bins=(bins, bins), cmap=taskcmap,density=True)
        mpl.pyplot.xlim([0,xyVals[1]])
        mpl.pyplot.ylim([0,xyVals[1]])
        mpl.pyplot.xticks(np.round(np.arange(xyVals[0],xyVals[1]*1.1,np.max(xyVals)/4.),4),fontweight='bold',fontsize=15)
        mpl.pyplot.yticks(np.round(np.arange(xyVals[0],xyVals[1]*1.1,np.max(xyVals)/4.),4),fontweight='bold',fontsize=15)
        mpl.pyplot.xlabel('Intra-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        mpl.pyplot.ylabel('Inter-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        cbar = mpl.pyplot.colorbar()
        cbar.set_label('Density',labelpad=20)
        mpl.pyplot.show()
        
    if 'scatter' in plotstyle or plotstyle == 'all':
        fig, ax = mpl.pyplot.subplots(figsize=(figSize[1],figSize[1]))
        ax.set_facecolor('white')
        mpl.pyplot.scatter(x,y, color=taskcolor,marker='o',s=10,linewidth=1,edgecolor='k',alpha=0.3)
        mpl.pyplot.xlim([0,xyVals[1]])
        mpl.pyplot.ylim([0,xyVals[1]])
        mpl.pyplot.xticks(np.round(np.arange(xyVals[0],xyVals[1]*1.1,np.max(xyVals)/4.),4),fontweight='bold',fontsize=15)
        mpl.pyplot.yticks(np.round(np.arange(xyVals[0],xyVals[1]*1.1,np.max(xyVals)/4.),4),fontweight='bold',fontsize=15)
        mpl.pyplot.xlabel('Intra-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        mpl.pyplot.ylabel('Inter-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        mpl.pyplot.show()
    
    return xx1,yy1,normalized


# Plot field map for each condition in taskcombos in 1 plot for comparison:
def plot_field_map_overlay(taskcombos,data,taskcolors,taskcmaps,alpha,lines,outpath,
                      cbar_option=True,figSize=(12,10),xlim=(0,0.025),ylim=(0,0.025),
                     shade=True,thr=0.0001):
    for taskcombo in taskcombos:
        mpl.pyplot.figure(figsize=(figSize[0],figSize[1]))
        sns.set_style('white')
        ax=mpl.pyplot.gca()
        mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams['font.size'] = 1
        sns.set(font_scale=3)
        ax.axes.set_xlim([xlim[0],xlim[1]])
        ax.axes.set_ylim([ylim[0],ylim[1]])
        mpl.pyplot.xticks([],fontweight='bold',fontsize=20)
        mpl.pyplot.yticks([],fontweight='bold',fontsize=20)
        mpl.pyplot.xlabel('Intra-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        mpl.pyplot.ylabel('Inter-individual Variation',labelpad=20,fontweight='bold',fontsize=20)
        ###

        for num,cond1 in enumerate(taskcombo):
            t1color = taskcolors[cond1]
            exec('colors1 = mpl.pyplot.cm.%s(np.linspace(0,1,128))' % taskcmaps[cond1])
            cond1w = data[cond1]['raww'].copy()
            cond1b = data[cond1]['rawb'].copy()

            # Mask b/w and w/in values for each condition
            bothmask = data[cond1]['totmask']*data[cond1]['totmask']
            cond1w = cond1w[bothmask]
            cond1b = cond1b[bothmask]
            #### Edit end:

            bw='scott'
            gridsize=100
            cut=10
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
            legend=True
            cumulative=False
            shade=shade
            shade_lowest=False
            cbar=False
            cbar_ax=None
            filled=True
            fill_lowest=False
            vertical=False
            kernel="gau"
            
            # Kde distribution:
            xx1, yy1, z1 = _scipy_bivariate_kde(cond1w, cond1b, bw, gridsize, cut, clip)
            
            # Scaling and normalization so that field maps are comparable:
            scaler = float(1000) # Take this out and just make sure line 225 is float
            z1scale = scaler*z1/np.sum(z1)
            normalized = (z1scale-np.min(z1scale))/(np.max(z1scale)-np.min(z1scale))
            
            # Reset clip for actual kdeplot:
            clip=None
            
            # Set colorbar for scaled density plot:
            cbar_kws={'cmap':taskcmaps[cond1]}
            our_cmap = mpl.pyplot.get_cmap(taskcmaps[cond1])
            cmap_max = 1.00001
            norm = mpl.colors.Normalize(vmin=0, vmax=cmap_max)    
            proxy_mappable = mpl.cm.ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(normalized)   
            ax = _bivariate_kdeplot(xx1, yy1, normalized, shade, 
                                    shade_lowest, kernel, bw, gridsize, 
                                    cut, clip, legend, cbar, cbar_ax, cbar_kws, 
                                    ax,vmin=0,vmax=cmap_max,levels=5,alpha=alpha,
                                   linewidths=5)
            ax.set_aspect('equal')
            if cbar_option == True:
                cbar = mpl.pyplot.colorbar(proxy_mappable, boundaries=np.arange(0,cmap_max,.1), spacing='proportional', orientation='vertical', pad=.01)
#                 if (len(taskcombo) >= 1) & (num == len(taskcombo)-1):
                if num == 0:
                    cbar.set_label('Density',labelpad=20)
                    
        if lines == True:
            mpl.pyplot.plot([1,0],[1,0],color='black',alpha=0.3,zorder=0)
            for iccline in [0.2,0.4,0.6,0.8]:
                mpl.pyplot.plot([1,0],[iccline,0],color='black',alpha=0.3,zorder=len(taskcombos)+1)
                mpl.pyplot.plot([iccline,0],[1,0],color='black',alpha=0.3,zorder=len(taskcombos)+1) 
                
        if outpath == True:
            mpl.pyplot.savefig('../figures/shortpaper/fieldmaps/%s_%s_perc%s_fieldmap_nogsr_front_contour_070121_time_1200-600.png' % (taskcombo[0],taskcombo[1],percnum),dpi=300)
        mpl.pyplot.show()   
