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


# Load Data:
data = np.load('../tutorial/example_data/tutorial_data.npy',allow_pickle=True).item()
tasks = [task for task in data.keys()]
print('Task conditions in data variable: %s' % tasks)
print('Dictionary keys within each task condition: %s' % [data_keys for data_keys in data[tasks[0]].keys()])
newdata = data.copy()
for task in tasks:
    for m in ['raww_masked', 'rawb_masked', 'icc_masked', 'vartotal_masked']:
        del newdata[task][m]
    newmask = np.zeros(len(newdata[task]['raww']))
    newmask[newdata[task]['totmask_cortex']] = 1
    newdata[task]['totmask'] = newmask[:64620]
    del newdata[task]['totmask_cortex']
    for mm in ['rawb', 'raww', 'icc', 'vartotal']:
        newdata[task][mm] = newdata[task][mm][:64620]
np.save('../tutorial/example_data/tutorial_data.npy',newdata)