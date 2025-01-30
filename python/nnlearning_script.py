import sys
import gspread
import oauth2client.service_account #import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds']
creds = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(sys.argv[5], scope)

print()

import os
import tables
import datetime
import math
import numpy as np
import json
import xml.etree.ElementTree as ET
list_channels = []
tree = ET.parse(sys.argv[6])
root = tree.getroot()
for br1Elem in root:
    if br1Elem.tag != 'spikeDetection':
        continue
    for br2Elem in br1Elem:
        if br2Elem.tag != 'channelGroups':
            continue
        for br3Elem in br2Elem:
            if br3Elem.tag != 'group':
                continue
            group=[];
            for br4Elem in br3Elem:
                if br4Elem.tag != 'channels':
                    continue
                for br5Elem in br4Elem:
                    if br5Elem.tag != 'channel':
                        continue
                    group.append(int(br5Elem.text))
            list_channels.append(group)
for br1Elem in root:
    if br1Elem.tag != 'acquisitionSystem':
        continue
    for br2Elem in br1Elem:
        if br2Elem.tag == 'samplingRate':
            samplingRate  = float(br2Elem.text)
        if br2Elem.tag == 'nChannels':
            nChannels = int(br2Elem.text)
for br1Elem in root:
    if br1Elem.tag != 'programs':
        continue
    for br2Elem in br1Elem:
        if br2Elem.tag != 'program':
            continue
        for br3Elem in br2Elem:
            if br3Elem.tag == 'name' and br3Elem.text=='ndm_hipass':
                for br3Elem2 in br2Elem:
                    if br3Elem2.tag != 'parameters':
                        continue
                    for br4Elem in br3Elem2:
                        if br4Elem.tag != 'parameter':
                            continue
                        for br5Elem in br4Elem:
                            if br5Elem.tag == 'name' and br5Elem.text=='windowHalfLength':
                                for br5Elem2 in br4Elem:
                                    if br5Elem2.tag != 'value':
                                        continue
                                    windowHalfLength = int(br5Elem2.text)
                            else:
                                continue
            else:
                continue



import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy import ndimage



class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        for i in range(len(self.path)):
            sys.path.insert(0, self.path[i])

    def __exit__(self, exc_type, exc_value, traceback):
        for i in range(len(self.path)):
            sys.path.remove(self.path[i])

python_sys_path = [sys.argv[4]]
with add_path(python_sys_path):
    mobsNN = __import__("mobs_nndecoding")
    mobsBAYES = __import__("mobs_bayesiandecoding")
    mobs_NB = __import__("mobs_networkbuilding")
    mobsFILT = __import__("mobs_filtering")
modules = {"mobsNN":mobsNN, "mobsBAYES":mobsBAYES, "mobs_NB":mobs_NB, "mobsFILT":mobsFILT}


def timedelta_to_ms(timedelta):
    ms = 0
    ms = ms + 3600*24*1000*timedelta.days
    ms = ms + 1000*timedelta.seconds
    ms = ms + timedelta.microseconds/1000
    return ms
    
def clear_clusters(Cluster_selection, clusters):
    return [np.multiply(clusters[tetrode],Cluster_selection[tetrode])
            for tetrode in range(len(Cluster_selection))]

def cleanProbas(probas):
    for i in range(len(probas)):
        temp = probas[i,:,:]
        temp[temp<np.max(temp/3)] = 0
        probas[i,:,:] = temp

def translatePosition(grpR, grpC, Bins):
    Rsize = np.mean(Bins[0][1:Bins[0].size] - Bins[0][0:Bins[0].size-1])
    Csize = np.mean(Bins[1][1:Bins[1].size] - Bins[1][0:Bins[1].size-1])

    RPos = np.array([Bins[0][0] + grpR[n]*Rsize for n in range(len(grpR))])
    CPos = np.array([Bins[1][0] + grpC[n]*Csize for n in range(len(grpC))])

    return RPos, CPos

def next_col(sheet):
    str_list = list(filter(None, sheet.row_values(1)))
    return (len(str_list)+1)
    
def next_row(sheet):
    str_list = list(filter(None, sheet.col_values(1)))
    return (len(str_list)+1)

def save_data(sheet, data):
    cell_list = sheet.range(1, next_col(sheet), 1+len(data), next_col(sheet))
    idx = 0
    for cell in cell_list:
        if idx == 0:
            cell.value = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        else:
            cell.value = data[idx-1]
        idx = idx +1
    sheet.update_cells(cell_list)
    
def write_log(sheet, dict):
    nxrow = next_row(sheet)
    data = {}
    idx = 0
    past_keys = sheet.col_values(1)
    for i in range(len(past_keys)):
        if i==0:
            continue
        else:
            if past_keys[i] in dict.keys():
                data[str(idx)] = dict[past_keys[i]]
            else:
                data[str(idx)] = None
            idx = idx + 1
    for key in dict.keys():
        if key in past_keys:
            continue
        else:
            sheet.update_cell(next_row(sheet),1,key)
            data[str(idx)] = dict[key]
            idx = idx + 1
    
    cell_list = sheet.range(1, next_col(sheet), 1+len(data), next_col(sheet))
    idx = 0
    for cell in cell_list:
        if idx == 0:
            cell.value = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        else:
            cell.value = data[str(idx-1)]
        idx = idx +1
    sheet.update_cells(cell_list)

def save_learning(sheet, data, log):
    save_data(sheet.worksheet("Data"), data)
    write_log(sheet.worksheet("Log"), log)












### Header

xml_path = sys.argv[6]
project_path = sys.argv[1]
prefix_results = project_path + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
os.makedirs(prefix_results)
NN_dir = prefix_results + '/'

n_tetrodes = len(list_channels) # max number !
speed_cut = 9#3
bandwidth = 5#3.5
masking_factor = 20
randomize_learning = 0   # fraction of cluster labels that will be shuffled before learning
cluster_modifier = 1   # factor to modify the number of clusters before learning

# start_time = int(float(sys.argv[3]))
# stop_time = int(float(sys.argv[2]))
start_time = 1180
stop_time = 3300
learning_time = 90*(stop_time-start_time)//100
nSteps = 20000
time_bin = 0.02   # in seconds


log_entry = {'prefix': prefix_results, 
            'speed_cut' : speed_cut,
            'bandwidth' : bandwidth,
            'masking_factor' : masking_factor,
            'start_time' : start_time,
            'stop_time' : stop_time,
            'learning_time' : learning_time, 
            'steps' : nSteps, 
            'time_bin' : time_bin, 
            'rand_param' : randomize_learning,
            'cluster_modifier' : cluster_modifier,
            'list_channels' : str(list_channels),
            'window half length' : windowHalfLength}


















### Learning

efficiencies = modules['mobsNN'].neural_encode(modules, xml_path, NN_dir, list_channels, samplingRate,
    start_time        = start_time,
    stop_time         = start_time + learning_time,
    rand_param        = randomize_learning,
    cluster_modifier  = cluster_modifier,
    nSteps            = nSteps)
print('efficiencies : ', efficiencies)
log_entry.update({'efficiencies' : str(efficiencies)})

bayes_matrices = modules['mobsBAYES'].build_kernel_densities(modules, project_path, xml_path, list_channels, samplingRate,
    speed_cut        = speed_cut,
    start_time       = start_time,
    stop_time        = start_time + learning_time,
    end_time         = stop_time,
    bandwidth        = bandwidth,
    kernel           = 'gaussian',
    masking_factor   = masking_factor,
    cluster_modifier = cluster_modifier,
    rand_param       = randomize_learning)
np.save(prefix_results+'_bayes.npy', bayes_matrices)
Occupation = bayes_matrices['Occupation']
MRF = bayes_matrices['Marginal rate functions']
RF = bayes_matrices['Rate functions']
RF_flat = [RF[tetrode][clu] for tetrode in range(len(RF)) for clu in range(len(RF[tetrode]))]

np.save(prefix_results+'_Occupation.npy', Occupation)
np.save(prefix_results+'_MRF.npy', MRF)
np.save(prefix_results+'_RF.npy', RF_flat)


















### Decoding



CLOCK1 = datetime.datetime.now()
spikes_info = {'spikes' : [], 'spike_time' : [], 'thresholds' : []}
# spikes_info = modules['mobsFILT'].find_spikes(xml_path, windowHalfLength, nChannels, list_channels, samplingRate)

guessed_clusters_info = modules['mobsNN'].neural_decode(spikes_info, xml_path, project_path, NN_dir, list_channels, samplingRate,
    speed_cut        = speed_cut,
    start_time       = start_time + learning_time,
    stop_time        = stop_time,
    cluster_modifier = cluster_modifier)

# print("\nSpikes have been sorted.")
# for i in range(len(guessed_clusters_info['clusters'])):
#     print("On group "+str(i)+" there are "+str(np.shape(guessed_clusters_info['clusters'][i])[1])+" clusters.")

bayes_matrices = np.load(prefix_results+'_bayes.npy').item()

position_proba, position, nSpikes = modules['mobsBAYES'].bayesian_decoding(project_path, time_bin, guessed_clusters_info, bayes_matrices,
    start_time     = start_time + learning_time,
    stop_time      = stop_time, 
    masking_factor = masking_factor)


Occupation = bayes_matrices['Occupation']
np.savez(prefix_results+'_simDecoding', Occupation, position_proba, position, spikes_info['thresholds'])

CLOCK2 = datetime.datetime.now()

duration = timedelta_to_ms(CLOCK2 - CLOCK1)
n_bin = math.floor((stop_time - start_time)/time_bin)
print('Calculation over. Mean time per bin : %.3f ms' % (duration/n_bin))






















### Plotting

bayes_matrices = np.load(prefix_results+'_bayes.npy').item()
Bins = bayes_matrices['Bins']
Results = np.load(prefix_results+'_simDecoding.npz')
Occupation = Results['arr_0']
position_proba = Results['arr_1']
position = Results['arr_2'].tolist()
OccupationG = Occupation>(np.amax(Occupation)/masking_factor)



X_proba = [np.sum(position_proba[n,:,:], axis=1) for n in range(len(position_proba))]
Y_proba = [np.sum(position_proba[n,:,:], axis=0) for n in range(len(position_proba))]
position_guessed = []
position_maxlik = [np.unravel_index(position_proba[n].argmax(), position_proba[n].shape) for n in range(len(position_proba))]


X_true = [position[n][0] for n in range(len(position))]
Y_true = [position[n][1] for n in range(len(position))]
X_guessed = [np.average( Bins[0], weights=X_proba[n] ) for n in range(len(X_proba))]
Y_guessed = [np.average( Bins[1], weights=Y_proba[n] ) for n in range(len(Y_proba))]
X_err = [np.abs(X_true[n] - X_guessed[n]) for n in range(len(X_true))]
Y_err = [np.abs(Y_true[n] - Y_guessed[n]) for n in range(len(Y_true))]
Error = [np.sqrt(X_err[n]**2 + Y_err[n]**2) for n in range(len(X_err))]
X_maxlik = [position_maxlik[n][0] for n in range(len(position_maxlik))]
Y_maxlik = [position_maxlik[n][1] for n in range(len(position_maxlik))]
X_standdev = np.array([np.std(np.sum(position_proba[n], 1)) for n in range(len(position_proba))])
Y_standdev = np.array([np.std(np.sum(position_proba[n], 0)) for n in range(len(position_proba))])
Standdev   = np.sqrt(np.power(X_standdev,2) + np.power(Y_standdev,2))

print("mean error is "+ str(np.mean(Error)))
tri = np.argsort(Standdev)
Errornp = np.array(Error)
Selected_errors = np.array([ 
        np.mean(Errornp[ tri[0:1*len(tri)//10] ]), 
        np.mean(Errornp[ tri[1*len(tri)//10:2*len(tri)//10] ]),
        np.mean(Errornp[ tri[2*len(tri)//10:3*len(tri)//10] ]),
        np.mean(Errornp[ tri[3*len(tri)//10:4*len(tri)//10] ]),
        np.mean(Errornp[ tri[4*len(tri)//10:5*len(tri)//10] ]),
        np.mean(Errornp[ tri[5*len(tri)//10:6*len(tri)//10] ]),
        np.mean(Errornp[ tri[6*len(tri)//10:7*len(tri)//10] ]),
        np.mean(Errornp[ tri[7*len(tri)//10:8*len(tri)//10] ]),
        np.mean(Errornp[ tri[8*len(tri)//10:9*len(tri)//10] ]),
        np.mean(Errornp[ tri[9*len(tri)//10:len(tri)]       ]) ])
print("----Selected errors----")
print(Selected_errors)
std_bins = np.array([
        Standdev[tri[0]], 
        Standdev[tri[1*len(tri)//10]], 
        Standdev[tri[2*len(tri)//10]], 
        Standdev[tri[3*len(tri)//10]], 
        Standdev[tri[4*len(tri)//10]], 
        Standdev[tri[5*len(tri)//10]], 
        Standdev[tri[6*len(tri)//10]], 
        Standdev[tri[7*len(tri)//10]], 
        Standdev[tri[8*len(tri)//10]], 
        Standdev[tri[9*len(tri)//10]], 
        Standdev[tri[len(tri)-1]] ])
log_entry.update({"std_bins":str(std_bins)})




outjsonStr = {};
outjsonStr['encodingPrefix'] = prefix_results
outjsonStr['mousePort'] = 0

outjsonStr['nGroups'] = len(list_channels)
for group in range(len(list_channels)):
    outjsonStr['group'+str(group)]={}
    outjsonStr['group'+str(group)]['nChannels'] = len(list_channels[group])
    outjsonStr['group'+str(group)]['nClusters'] = np.shape(guessed_clusters_info['clusters'][group])[1]
    for chnl in range(len(list_channels[group])):
        outjsonStr['group'+str(group)]['channel'+str(chnl)]=list_channels[group][chnl]
    if os.path.isfile(xml_path[:len(xml_path)-3] + 'clu.' + str(group+1)):
        outjsonStr['group'+str(group)]['networkPath'] = prefix_results + '/network_t' + str(group+1)
    else:
        outjsonStr['group'+str(group)]['networkPath'] = ''

outjsonStr['windowHalfLength'] = windowHalfLength
outjsonStr['gammaChannel']     = 0
outjsonStr['gammaThreshold']   = 0.0
outjsonStr['thedelChannel']    = 0
outjsonStr['thedelThreshold']  = 0.0
outjsonStr['wakePin']          = 15
outjsonStr['remPin']           = 14

outjsonStr['nStimConditions'] = 1
outjsonStr['stimCondition0'] = {}
outjsonStr['stimCondition0']['stimPin'] = 13
outjsonStr['stimCondition0']['lowerX'] = 0.0
outjsonStr['stimCondition0']['higherX'] = 0.0
outjsonStr['stimCondition0']['lowerY'] = 0.0
outjsonStr['stimCondition0']['higherY'] = 0.0
outjsonStr['stimCondition0']['lowerDev'] = 0.0
outjsonStr['stimCondition0']['higherDev'] = 0.0

outjson = json.dumps(outjsonStr, indent=4)
with open(sys.argv[6][:len(sys.argv[6])-4]+'.json',"w") as json_file:
    json_file.write(outjson)







# ERROR & STD
fig, ax = plt.subplots(figsize=(50,10))
ax2 = plt.subplot2grid((2,3),(0,0))
ax2.plot(X_err, X_standdev, 'b.')
ax2.axvline(x=np.mean(X_err), linewidth=1, color='k')
ax2.axhline(y=np.mean(X_standdev), linewidth=1, color='k')

ax3 = plt.subplot2grid((2,3),(1,0), sharex=ax2, sharey=ax2)
ax3.plot(Y_err, Y_standdev, 'b.')
ax3.axvline(x=np.mean(Y_err), linewidth=1, color='k')
ax3.axhline(y=np.mean(Y_standdev), linewidth=1, color='k')

ax1 = plt.subplot2grid((2,3),(0,1), rowspan=2, colspan=2)
ax1.plot(Error, Standdev, 'b.')
ax1.axvline(x=np.mean(Error), linewidth=1, color='k')
ax1.axhline(y=np.mean(Standdev), linewidth=1, color='k')

plt.show(block=True)



# Histogram of errors
fig2, axb = plt.subplots(figsize=(50,10))
axb.set_title('Histogram of errors', size=30)
axb.hist(Error, 100, edgecolor='k')
plt.show(block=True)






# # # MOVIE
fig, ax = plt.subplots(figsize=(50,10))
best_bins = np.argsort(Selected_errors)
first_bin  = [std_bins[best_bins[0]], std_bins[best_bins[0]]+1]
second_bin = [std_bins[best_bins[1]], std_bins[best_bins[1]]+1]
# frame_selection = range(len(Standdev))
frame_selection = np.union1d(
                        np.intersect1d(np.where(Standdev[:] > first_bin[0])[0],
                                       np.where(Standdev[:] < first_bin[1])[0]),
                        np.intersect1d(np.where(Standdev[:] > second_bin[0])[0],
                                       np.where(Standdev[:] < second_bin[1])[0]))

ax1 = plt.subplot2grid((2,2),(0,1), rowspan=2)
im1 = ax1.imshow(position_proba[0][:,:], animated=True, extent=[Bins[1][0],Bins[1][-1],Bins[0][-1],Bins[0][0]])
# im1 = ax1.imshow(position_proba[0][:,:], norm=LogNorm(vmin=0.00001, vmax=1), animated=True, extent=[Bins[1][0],Bins[1][-1],Bins[0][-1],Bins[0][0]])

im2, = ax1.plot([position[0][1]],[position[0][0]],marker='o', markersize=15, color="red")
im2b, = ax1.plot([Y_guessed[0]],[X_guessed[0]],marker='P', markersize=15, color="green")
im3 = ax1.contour(Bins[1], Bins[0], OccupationG)
cmap = fig.colorbar(im1, ax=ax1)

# X
ax2 = plt.subplot2grid((2,2),(0,0))
plot11 = ax2.plot(X_true, linewidth=2, color='r')
plot12 = ax2.plot(X_guessed, linewidth=1, color='b')
plot14 = ax2.plot(frame_selection, [X_guessed[frame_selection[n]] for n in range(len(frame_selection))], 'ko', markersize=10)
# plot12b = ax2.plot(X_maxlik, linewidth=1, color='y')
paint1 = ax2.fill_between(range(len(X_true)) , np.subtract(X_guessed,X_standdev) , np.add(X_guessed,X_standdev))
plot13 = ax2.axvline(linewidth=3, color='k')
plt.xlim(-200,200)

# Y
ax3 = plt.subplot2grid((2,2),(1,0), sharex=ax2, sharey=ax2)
plot21 = ax3.plot(Y_true, linewidth=2, color='r')
plot22 = ax3.plot(Y_guessed, linewidth=1, color='b')
plot24 = ax3.plot(frame_selection, [Y_guessed[frame_selection[n]] for n in range(len(frame_selection))], 'ko', markersize=10)
# plot22b = ax3.plot(Y_maxlik, linewidth=1, color='y')
paint2 = ax3.fill_between(range(len(Y_true)) , np.subtract(Y_guessed,Y_standdev) , np.add(Y_guessed,Y_standdev))
plot23 = ax3.axvline(linewidth=3, color='k')
plt.xlim(-200,200)

def updatefig(frame, *args):
    global position_proba, position, OccupationG, frame_selection, X_guessed, Y_guessed
    reduced_frame = frame % len(frame_selection)
    selected_frame = frame_selection[reduced_frame]
    im1.set_array(position_proba[selected_frame][:,:])
    im2.set_data([position[selected_frame][1]],[position[selected_frame][0]])
    im2b.set_data([Y_guessed[selected_frame]],[X_guessed[selected_frame]])
    plt.xlim(-200+selected_frame,200+selected_frame)
    plot13.set_xdata(selected_frame)
    plot23.set_xdata(selected_frame)
    return im1,im3,im2,im2b, plot11, plot12, plot13, plot14, plot21, plot22, plot23, plot24#, plot12b, plot22b

ani = animation.FuncAnimation(fig,updatefig,interval=100, save_count=len(frame_selection))
if len(frame_selection)<len(position_proba)/4:
    ani.save(prefix_results+'_Movie.mp4')
fig.show()








client = gspread.authorize(creds)
spreadsheet = client.open_by_key('1Wj7GgzwttypnX9zqIKleYa_zgkAQ4hf1122JWU5FDic')
save_learning(spreadsheet, Selected_errors, log_entry)
print("Results and log saved.")
