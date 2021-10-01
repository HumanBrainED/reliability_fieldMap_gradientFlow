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
        df = calc_icc_vectors(x0,y0,x1,y1,icc0,icc1,task1,task2)

        plotname =  '%s-%s_%s_vectors' % (task2,task1,posNeg)
        converted_angles = np.array([convertAngle(df['theta0'][i],df['xdiff'][i]) for i in range(len(df['theta0']))])
        angVerts = parcel2vert(parcellation,converted_angles)
        posNegMask = parcel2vert(parcellation,icc1-icc0)
        meandICC = np.mean(posNegMask,0)
        numVertices = int(angVerts.shape[1]/2.)
        symmetric_cmap = False
        parcellation_mask = np.where(glasserlabel[0,:] == 0)[0]
        angVerts[0,parcellation_mask] = np.nan

warm_cmap_whites,cold_cmap_whites = warm_cold_gradient_flow_cmap()
# testarray = np.ones([1,20484]) * 300
# testarray[0,parcellation_mask] = 45
numVertices = int(angVerts.shape[1]/2.)
symmetric_cmap = False
plot_surface(angVerts,lsurf,rsurf,numVertices,data_range,warm_cmap_whites,alpha,
                         darkness,symmetric_cmap,False,outpath,'%s-%s_%s_dICC_angle' % (task2,task1,posNeg))
