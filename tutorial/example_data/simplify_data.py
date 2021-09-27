import numpy as np
import glob
sys.path.append('../code')
from variability_utils import *

path = '../tutorial/example_data/icc_output'
tasks = ['2subsets_1200TRx1seg_REST1_LR_nogsr','2subsets_1200TRx1seg_REST1_LR_gsr']
data = {}
for task in tasks:
    csvDict = {'rawb':glob.glob('%s/%s/*inter*.csv' % (path,task))[0],'raww':glob.glob('%s/%s/*intra*.csv' % (path,task))[0],
               'icc':glob.glob('%s/%s/*icc*.csv' % (path,task))[0],'vartotal':glob.glob('%s/%s/*vartotal*.csv' % (path,task))[0]}
    data[task] = load_data(task,csvDict)
data['REST_nogsr'] = data[tasks[0]]; del data[tasks[0]]
data['REST_gsr'] = data[tasks[1]]; del data[tasks[1]]
for task in ['REST_nogsr','REST_gsr']:
    data[task]['totmask_cortex'] = data[task]['totmask'][data[task]['totmask']<64620] 
    data[task]['raww_masked'] = data[task]['raww'][:64620][data[task]['totmask_cortex']]
    data[task]['rawb_masked'] = data[task]['rawb'][:64620][data[task]['totmask_cortex']]
    data[task]['icc_masked'] = data[task]['icc'][:64620][data[task]['totmask_cortex']]
    data[task]['vartotal_masked'] = data[task]['vartotal'][:64620][data[task]['totmask_cortex']]
for i in data[task].keys():
    print(i,data[task][i].shape)
np.save('../tutorial/example_data/tutorial_data.npy',data)