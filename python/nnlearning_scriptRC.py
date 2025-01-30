import sys
# import gspread
# import oauth2client.service_account #import ServiceAccountCredentials
# scope = ['https://spreadsheets.google.com/feeds']
# creds = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(sys.argv[5], scope)

print()

import os
import tables
import datetime
import math
import numpy as np
import json
import xml.etree.ElementTree as ET
list_channels = []
try:
    tree = ET.parse(os.path.expanduser("~/Documents/dataset/RatCatanese/rat122-20090731.xml"))
except:
    sys.exit(4)
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
from scipy.stats import gaussian_kde



class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        for i in range(len(self.path)):
            sys.path.insert(0, self.path[i])

    def __exit__(self, exc_type, exc_value, traceback):
        for i in range(len(self.path)):
            sys.path.remove(self.path[i])

python_sys_path = [os.path.expanduser("~/Dropbox/Kteam/PrgMatlab/python")]
with add_path(python_sys_path):
    mobsNN = __import__("mobs_nndecoding4")
    mobs_NB = __import__("mobs_networkbuilding")
modules = {"mobsNN":mobsNN, "mobs_NB":mobs_NB}


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
getDir = lambda path: path if path[-1]=='/' else getDir(path[:-1])
xml_path = os.path.expanduser("~/Documents/dataset/RatCatanese/rat122-20090731.xml")
project_path = getDir(xml_path)
prefix_results = project_path + "mobsEncoding_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
# try:
#     os.makedirs(prefix_results)
# except:
#     pass
NN_dir = prefix_results + '/'

n_tetrodes = len(list_channels) # max number !
speed_cut = 9
bandwidth = 5
masking_factor = 20

# start_time = int(float(sys.argv[3]))
# stop_time = int(float(sys.argv[2]))
start_time = 1180
stop_time = 3300
learning_time = 90*(stop_time-start_time)//100
nSteps = 20000
time_bin = 0.036   # in seconds


log_entry = {'prefix': prefix_results, 
            'speed_cut' : speed_cut,
            'bandwidth' : bandwidth,
            'masking_factor' : masking_factor,
            'start_time' : start_time,
            'stop_time' : stop_time,
            'learning_time' : learning_time, 
            'steps' : nSteps, 
            'time_bin' : time_bin, 
            'list_channels' : str(list_channels),
            'window half length' : windowHalfLength}


















# ### Learning

# clu_path = xml_path[:len(xml_path)-3]
# Data = modules['mobsNN'].extract_data(clu_path, list_channels, start_time, start_time + learning_time, stop_time,
#                                         speed_cut, samplingRate, 
#                                         masking_factor, 'gaussian', bandwidth)
NN_dir = os.path.expanduser('~/Documents/dataset/RatCatanese/mobsEncoding_2019-05-06_15:10/')
# np.save(NN_dir+'_data.npy', Data)
# # sys.exit(0)

Data = np.load(NN_dir+'_data.npy', allow_pickle=True).item()


# print('shuffling spike time')
# modules['mobsNN'].shuffle_spike_time(Data, 2.0)
# print('spike time shuffled')

efficiencies = modules['mobsNN'].build_position_decoder(modules, Data, NN_dir, nSteps)

print('efficiencies : ', efficiencies)
# # log_entry.update({'efficiencies' : str(efficiencies)})






















# ## Decoding



# CLOCK1 = datetime.datetime.now()


position_proba, position, nSpikes,timeDeltas = modules['mobsNN'].decode_position(modules, Data, NN_dir, 
                                                            start_time + learning_time, stop_time, time_bin)


# CLOCK2 = datetime.datetime.now()

Occupation = Data['Occupation']
timesInMs = np.array([timedelta_to_ms(timeDeltas[n+1] - timeDeltas[n]) for n in range(len(timeDeltas)-1)])
np.savez(NN_dir+'_simDecoding', Occupation=Occupation, position_proba=position_proba, position=position, times=timesInMs, nSpikes=nSpikes)


# duration = timedelta_to_ms(CLOCK2 - CLOCK1)
n_bin = math.floor(((stop_time - start_time)-learning_time)/time_bin)
# print('Calculation over. Mean time per bin : %.3f ms' % (duration/n_bin))
# print('Mean number of spikes per bin : %.3f' % (np.sum(nSpikes)/n_bin))

# sys.exit(0)




















### Plotting

Bins = Data['Bins']
# NN_dir = os.path.expanduser('~/Documents/dataset/RatCatanese/mobsEncoding_2019-09-30_shuffledSpikes/')
Results = np.load(NN_dir+'_simDecoding.npz')
Occupation = Results['Occupation']
position_proba = Results['position_proba']
position = Results['position'].tolist()
times = Results['times']
nSpikes = Results['nSpikes']
# Occupation = Results['arr_0']
# position_proba = Results['arr_1']
# position = Results['arr_2'].tolist()
OccupationG = Occupation>(np.amax(Occupation)/masking_factor)
for n in range(len(position_proba)):
    if np.sum(position_proba[n,:,:])==0.0:
        position_proba[n,:,:] = np.ones([45,45]) / np.sum([45,45])


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
X_standdev = np.sqrt([np.sum([X_proba[n][x]*(Bins[0][x]-X_guessed[n])**2 for x in range(Bins[0].size)]) for n in range(len(position_proba))])
Y_standdev = np.sqrt([np.sum([Y_proba[n][y]*(Bins[1][y]-Y_guessed[n])**2 for y in range(Bins[1].size)]) for n in range(len(position_proba))])
Standdev   = np.sqrt(np.power(X_standdev,2) + np.power(Y_standdev,2))






### points of lowest speed : bottom right of U shape points to reward locations
def distance (pos1,pos2):
    return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
speed = [distance(position[n],position[n+1]) for n in range(len(position)-1)]
speed.append(speed[-1])
speed = np.array(speed)
tri = np.argsort(speed)
Errornp = np.array(Error)
selectionHigh = np.where(np.logical_and(
    speed < speed[tri[2*len(tri)//6]],
    Standdev > 40))
selectionLowRight = np.where(np.logical_and(
    speed < speed[tri[2*len(tri)//6]],
    np.logical_and(
        Standdev < 40,
        Errornp > 100)))
selectionLowLeft = np.where(np.logical_and(
    speed < speed[tri[2*len(tri)//6]],
    np.logical_and(
        Standdev < 40,
        Errornp < 100)))
fig, ax = plt.subplots(figsize=(10,15))
ax1 = plt.subplot2grid((3,2),(0,0))
ax1.scatter(Errornp[selectionHigh], Standdev[selectionHigh], s=5, c='black')
ax1.scatter(Errornp[selectionLowLeft], Standdev[selectionLowLeft], s=5, c='xkcd:light gray')
ax1.scatter(Errornp[selectionLowRight], Standdev[selectionLowRight], s=5, c='xkcd:light gray')
ax1.set_ylabel('width of the probability function')
ax1.set_xlabel('true error')

ax2 = plt.subplot2grid((3,2),(1,0), sharex=ax1, sharey=ax1)
ax2.scatter(Errornp[selectionLowLeft], Standdev[selectionLowLeft], s=5, c='black')
ax2.scatter(Errornp[selectionHigh], Standdev[selectionHigh], s=5, c='xkcd:light gray')
ax2.scatter(Errornp[selectionLowRight], Standdev[selectionLowRight], s=5, c='xkcd:light gray')
ax2.set_ylabel('width of the probability function')
ax2.set_xlabel('true error')

ax5 = plt.subplot2grid((3,2),(2,0), sharex=ax1, sharey=ax1)
ax5.scatter(Errornp[selectionLowRight], Standdev[selectionLowRight], s=5, c='black')
ax5.scatter(Errornp[selectionHigh], Standdev[selectionHigh], s=5, c='xkcd:light gray')
ax5.scatter(Errornp[selectionLowLeft], Standdev[selectionLowLeft], s=5, c='xkcd:light gray')
ax5.set_ylabel('width of the probability function')
ax5.set_xlabel('true error')

# plt.tight_layout()
# # plt.savefig(os.path.expanduser('~/Pictures/_tempFig.png'), bbox_inches='tight')

X_guessednp = np.array(X_guessed)
Y_guessednp = np.array(Y_guessed)
positionnp = np.array(position)
ax3 = plt.subplot2grid((3,2),(0,1))
ax3.scatter(Y_guessednp[selectionHigh], X_guessednp[selectionHigh], marker='P',color='green',label='guessed position')
ax3.scatter(positionnp[selectionHigh,1], positionnp[selectionHigh,0], marker='o',color='red',label='true position')
ax3.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="red")
ax3.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="red")
n = 135; nn=2*n
ax3.plot([nn-125,nn-125,nn-115,nn-90,nn-90,nn-115,nn-125], [100,215,225,220,185,100,100], color="red")
ax3.legend()
a,b = plt.ylim()
plt.ylim(b,a)

ax4 = plt.subplot2grid((3,2),(1,1))
ax4.scatter(Y_guessednp[selectionLowLeft], X_guessednp[selectionLowLeft], marker='P',color='green',label='guessed position')
ax4.scatter(positionnp[selectionLowLeft,1], positionnp[selectionLowLeft,0], marker='o',color='red',label='true position')
ax4.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="red")
ax4.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="red")
n = 135; nn=2*n
ax4.plot([nn-125,nn-125,nn-115,nn-90,nn-90,nn-115,nn-125], [100,215,225,220,185,100,100], color="red")
ax4.legend()
a,b = plt.ylim()
plt.ylim(b,a)

ax6 = plt.subplot2grid((3,2),(2,1))
ax6.scatter(Y_guessednp[selectionLowRight], X_guessednp[selectionLowRight], marker='P',color='green',label='guessed position')
ax6.scatter(positionnp[selectionLowRight,1], positionnp[selectionLowRight,0], marker='o',color='red',label='true position')
ax6.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="red")
ax6.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="red")
n = 135; nn=2*n
ax6.plot([nn-125,nn-125,nn-115,nn-90,nn-90,nn-115,nn-125], [100,215,225,220,185,100,100], color="red")
ax6.legend()
a,b = plt.ylim()
plt.ylim(b,a)

plt.suptitle('Points of lowest speed (33%) : \nbottom right of U shape points to reward locations', size=20)
# plt.savefig(os.path.expanduser('~/Dropbox/Mobs_member/Thibault/Poster Chicago Thibault/replays.png'), bbox_inches='tight')
plt.show(block=True)





print("mean error is "+ str(np.mean(Error)))
tri = np.argsort(Standdev)
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
outjsonStr['encodingPrefix'] = NN_dir + 'mobsGraph'
outjsonStr['mousePort'] = 0

outjsonStr['nGroups'] = Data['nGroups']
idx=0
for group in range(len(list_channels)):
    if os.path.isfile(xml_path[:len(xml_path)-3] + 'clu.' + str(group+1)):
        outjsonStr['group'+str(group-idx)]={}
        outjsonStr['group'+str(group-idx)]['nChannels'] = len(list_channels[group])
        outjsonStr['group'+str(group-idx)]['nClusters'] = Data['clustersPerGroup'][group-idx]
        for chnl in range(len(list_channels[group])):
            outjsonStr['group'+str(group-idx)]['channel'+str(chnl)]=list_channels[group][chnl]
    else:
        idx+=1

outjsonStr['windowHalfLength'] = windowHalfLength

outjsonStr['nStimConditions'] = 1
outjsonStr['stimCondition0'] = {}
outjsonStr['stimCondition0']['stimPin'] = 14
outjsonStr['stimCondition0']['lowerX'] = 0.0
outjsonStr['stimCondition0']['higherX'] = 0.0
outjsonStr['stimCondition0']['lowerY'] = 0.0
outjsonStr['stimCondition0']['higherY'] = 0.0
outjsonStr['stimCondition0']['lowerDev'] = 0.0
outjsonStr['stimCondition0']['higherDev'] = 0.0

outjson = json.dumps(outjsonStr, indent=4)
with open(os.path.expanduser('~/Documents/dataset/RatCatanese/rat122-20090731.json'),"w") as json_file:
    json_file.write(outjson)







# ERROR & STD
def scattWithError(metric, label):
    fig, ax1 = plt.subplots(figsize=(15,15))
    # ax2 = plt.subplot2grid((2,3),(0,0))
    # ax2.plot(X_err, X_standdev, 'b.')
    # ax2.axvline(x=np.mean(X_err), linewidth=1, color='k')
    # ax2.axhline(y=np.mean(X_standdev), linewidth=1, color='k')
    # ax2.set_title('X')

    # ax3 = plt.subplot2grid((2,3),(1,0), sharex=ax2, sharey=ax2)
    # ax3.plot(Y_err, Y_standdev, 'b.')
    # ax3.axvline(x=np.mean(Y_err), linewidth=1, color='k')
    # ax3.axhline(y=np.mean(Y_standdev), linewidth=1, color='k')
    # ax3.set_title('Y')

    # ax1 = plt.subplot2grid((2,3),(0,1), rowspan=2, colspan=2)
    xy = np.vstack([Error, metric])
    z = gaussian_kde(xy)(xy)
    ax1.scatter(Error, metric, c=z, s=13, label='error and width of probability function for each time bin')
    # ax1.axvline(x=np.mean(Error), linewidth=1, color='k')
    # ax1.axhline(y=np.mean(metric), linewidth=1, color='k')
    # ax1.set_title('Distance')
    ax1.set_ylabel(label, fontsize=25)
    ax1.set_xlabel('true error', fontsize=25)
    nBins = 20
    _, edges = np.histogram(Error, nBins)
    histIdx = []
    for bin in range(nBins):
        temp=[]
        for n in range(len(Error)):
            if Error[n]<=edges[bin+1] and Error[n]>edges[bin]:
                temp.append(n)
        histIdx.append(temp)
    err=np.array([
        [np.median(metric[histIdx[n]])-np.percentile(metric[histIdx[n]],30) for n in range(nBins)],
        [np.percentile(metric[histIdx[n]],70)-np.median(metric[histIdx[n]]) for n in range(nBins)]])
    ax1.errorbar(
        [(edges[n+1]+edges[n])/2 for n in range(nBins)],
        [np.median(metric[histIdx[n]]) for n in range(nBins)], c='xkcd:cherry red', 
        yerr = err, 
        label=r'$median \pm 20 percentile$',
        linewidth=3)
    ax1.legend(loc="upper right", fontsize=25)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
scattWithError(Standdev, 'width of the probability function')
plt.ylim(0,120)
# plt.savefig(NN_dir+'_stdFig.png', bbox_inches='tight')
# plt.savefig(os.path.expanduser('~/Dropbox/Mobs_member/Thibault/Poster Chicago Thibault/classicError.png'), bbox_inches='tight')
plt.show()
scattWithError(speed, 'Speed')
plt.ylim(0,4)
# plt.savefig(os.path.expanduser('~/Dropbox/Mobs_member/Thibault/Poster Chicago Thibault/classicErrorSpeed.png'), bbox_inches='tight')
plt.show()



# Histogram of errors
fig2, axb = plt.subplots(figsize=(10,7))
axb.set_title('Histogram of errors', size=30)
axb.hist(Error, 100, edgecolor='k')
plt.savefig(NN_dir+'_errFig.png', bbox_inches='tight')
plt.show()


# # Overview
# best_bins = np.argsort(Selected_errors)
# frame_selection = range(len(Standdev))
# frame_selection = np.union1d(
#                         np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[0]],
#                                                 Standdev[:] < std_bins[best_bins[0]+1]))[0],
#                         np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[1]],
#                                                 Standdev[:] < std_bins[best_bins[1]+1]))[0])

# fig, ax = plt.subplots(figsize=(15,9))
# ax1 = plt.subplot2grid((2,1),(0,0))
# ax1.plot(np.array(position)[:,0], label='true X')
# # ax1.plot(X_guessed, label='guessed X')
# ax1.plot(frame_selection, np.array(X_guessed)[frame_selection], label='guessed X')
# ax1.legend()
# ax1.set_title('position X')

# ax2 = plt.subplot2grid((2,1),(1,0), sharex=ax1)
# ax2.plot(np.array(position)[:,1], label='true Y')
# # ax2.plot(Y_guessed[:], label='guessed Y')
# ax2.plot(frame_selection, np.array(Y_guessed)[frame_selection], label='guessed Y')
# ax2.legend()
# ax2.set_title('position Y')
# plt.savefig(NN_dir+'_overview.png', bbox_inches='tight')
# plt.show(block=True)

# Overview as figure
temp = Standdev
temp2 = temp.argsort()
thresh = temp[temp2[int(len(temp2)*0.2)]]
selection = np.array(Standdev)<thresh
# frames = np.where(selection)[0]
lw=5
fig, ax1 = plt.subplots(figsize=(20,12))
ax1.plot(np.where(selection)[0]*0.036, np.array(X_guessed)[selection]+200, color='xkcd:dark pink', markersize=10, label=None, linewidth=lw)
ax1.plot([n*0.036 for n in range(len(position))], np.array(position)[:,0]+200, label=None, color='k', linewidth=lw)
ax1.axis('off')

ax2 = ax1
ax2.plot(np.where(selection)[0]*0.036, np.array(Y_guessed)[selection], color='xkcd:dark pink', markersize=10, label='inferred position', linewidth=lw)
ax2.plot([n*0.036 for n in range(len(position))], np.array(position)[:,1], label='true position', color='k', linewidth=lw)
ax2.legend(loc="lower right", fontsize=25)
ax2.axis('off')
plt.xlim(100,170)
# plt.savefig(os.path.expanduser('~/Dropbox/Mobs_member/Thibault/Poster Chicago Thibault/classicResults.png'), bbox_inches='tight')
plt.show(block=True)




# # # MOVIE
n=170
best_bins = np.argsort(Selected_errors)
# frame_selection = range(len(Standdev))
frame_selection = np.union1d(
    np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[0]],
                            Standdev[:] < std_bins[best_bins[0]+1]))[0],
    np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[1]],
                            Standdev[:] < std_bins[best_bins[1]+1]))[0])
fig, ax = plt.subplots(figsize=(50,30))

ax1 = ax
# ax1 = plt.subplot2grid((2,2),(0,1), rowspan=2)
im1 = ax1.imshow(position_proba[frame_selection[n]][:,:], animated=True, extent=[Bins[1][0],Bins[1][-1],Bins[0][-1],Bins[0][0]])
# im1 = ax1.imshow(position_proba[0][:,:], norm=LogNorm(vmin=0.00001, vmax=1), animated=True, extent=[Bins[1][0],Bins[1][-1],Bins[0][-1],Bins[0][0]])

im2, = ax1.plot([position[frame_selection[n]][1]],[position[frame_selection[n]][0]],marker='o', markersize=35, color="red")
temp = np.array(position)
ax1.plot(temp[frame_selection[n]-80:frame_selection[n],1], temp[frame_selection[n]-80:frame_selection[n],0], 'r',linewidth=3)
# im2b, = ax1.plot([Y_guessed[frame_selection[n]]],[X_guessed[frame_selection[n]]],marker='P', markersize=15, color="green")
# im3 = ax1.contour(Bins[1], Bins[0], OccupationG)
# cmap = fig.colorbar(im1, ax=ax1)
im3 = ax1.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="white",linewidth=5)
im4 = ax1.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="white",linewidth=5)
x = 135; xx=2*x
im4 = ax1.plot([xx-125,xx-125,xx-115,xx-90,xx-90,xx-115,xx-125], [100,215,225,220,185,100,100], color="white",linewidth=5)
plt.xlim(50,210)
plt.ylim(252,40)
ax1.axis("off")
# plt.suptitle("n="+str(n))
# plt.savefig(os.path.expanduser('~/Dropbox/Mobs_member/Thibault/Poster Chicago Thibault/thetaCycle.png'), bbox_inches='tight')
plt.show(block = True)

sys.exit(0)

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

# ani = animation.FuncAnimation(fig,updatefig,interval=100, save_count=len(frame_selection))
# if len(frame_selection)<len(position_proba)/4:
#     ani.save(NN_dir+'_Movie.mp4')
# fig.show()





# try:
#     client = gspread.authorize(creds)
#     spreadsheet = client.open_by_key('1Wj7GgzwttypnX9zqIKleYa_zgkAQ4hf1122JWU5FDic')
#     save_learning(spreadsheet, Selected_errors, log_entry)
#     print("Results and log saved.")
# except:
#     print('not sending to google server.')
#     sys.exit(2)

