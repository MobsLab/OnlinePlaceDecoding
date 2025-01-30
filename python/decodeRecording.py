import sys

print()

dat_path = '/home/mobshamilton/Documents/dataset/Mouse-797/intanDecodingSession_181109_190106/'
xml_path = '/home/mobshamilton/Documents/dataset/Mouse-797/ERC-Mouse-797-09112018-Hab_SpikeRef.xml'
project_path = '/home/mobshamilton/Documents/dataset/Mouse-797/'
prefix_results = '/home/mobshamilton/Documents/dataset/Mouse-797/2018-11-12_10:31'
NN_dir = '/home/mobshamilton/Documents/dataset/Mouse-797/2018-11-09_17:09/'




import os
import tables
import datetime
import math
import numpy as np
import json
import xml.etree.ElementTree as ET
list_channels = []
tree = ET.parse(xml_path)
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

python_sys_path = ['/home/mobshamilton/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding/python']
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













### Header


n_tetrodes = len(list_channels) # max number !
bandwidth = 3.5
masking_factor = 20

time_bin = 0.04   # in seconds

















Results = np.load(prefix_results+'_simDecoding.npz')
Occupation = Results['arr_0']
position_proba = Results['arr_1']
position = Results['arr_2'].tolist()
thresholds = Results['arr_3'].tolist()



### Decoding


CLOCK1 = datetime.datetime.now()
spikes_info = modules['mobsFILT'].extract_spikes(dat_path, nChannels, list_channels, samplingRate, thresholds)

guessed_clusters_info = modules['mobsNN'].neural_decode(spikes_info, xml_path, project_path, NN_dir, list_channels, samplingRate)


bayes_matrices = np.load(prefix_results+'_bayes.npy').item()

position_proba = modules['mobsBAYES'].simpleDecode(time_bin, guessed_clusters_info, bayes_matrices,
    masking_factor = masking_factor)


Occupation = bayes_matrices['Occupation']
np.savez(dat_path+'_simDecoding', Occupation, position_proba, spikes_info['thresholds'])

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
frame_selection = range(len(Standdev))
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



