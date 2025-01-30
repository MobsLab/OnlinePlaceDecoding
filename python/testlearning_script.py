import sys

import os
import tables
import datetime
import math
import numpy as np
import json


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy import ndimage


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









### Header

project_path = '/home/mobshamilton/Documents/dataset/RatCatanese/mobsEncoding_2018-12-12_16:13/'
# prefix_results = '/home/mobshamilton/Documents/dataset/RatCatanese/NNdecoding/testing'
# project_path = '/home/mobshamilton/Documents/dataset/Mouse-797/2018-11-09_17:09'
prefix_results = project_path
NN_dir = prefix_results + '/'

masking_factor = 20
time_bin = 0.02   # in seconds










### Plotting
print('loading')
bayes_matrices = np.load(prefix_results+'_data.npy').item()
Bins = bayes_matrices['Bins']
Results = np.load(prefix_results+'_simDecoding.npz')
Occupation = Results['arr_0']
position_proba = Results['arr_1']
position = Results['arr_2'].tolist()
print('done.')

xx, yy = np.meshgrid(Bins[0],Bins[1], indexing='ij')
OccupationG = Occupation>(np.amax(Occupation)/masking_factor)
# noSpikes = np.where(np.array(nSpikes)!=0)[0]




# nEvaluation = np.array([np.log(np.divide(position_proba[n], 
#                                 np.sqrt((xx-position[n][0])**2 + (yy-position[n][1])**2))) 
#                                 for n in range(len(position_proba))])
# evaluation = np.divide(np.mean(nEvaluation, 0), OccupationG)
# MAevaluation = np.ma.divide(np.mean(nEvaluation, 0), OccupationG) # used to normalize ? : evaluation -= np.mean(MAevaluation); evaluation /= N;
# selection_value = np.sort(evaluation.flatten())
# selection_value = selection_value[19*len(selection_value)//20] # objective value taken from normalization ?
# blobs = evaluation > selection_value
# groups, ngroups = ndimage.label(blobs, structure=[[1,1,1],
#                                                   [1,1,1],
#                                                   [1,1,1]])
# grpR, grpC = np.vstack(ndimage.center_of_mass(evaluation, groups, np.arange(ngroups) + 1)).T
# grpR, grpC = translatePosition(grpR, grpC, Bins)

# stimPlaces = []
# stimCenters = []
# for grp in range(ngroups):
#     temp = np.where(groups == grp + 1)
#     # if temp[0].size > 6:
#     temp2 = np.zeros([Bins[0].size, Bins[1].size])
#     temp2[temp] = 1
#     stimPlaces.append( temp2 )
#     stimCenters.append( [grpR[grp], grpC[grp]] )
# stimProbas = []
# stimDistance = []
# stimSelection = []
# for grp in range(len(stimPlaces)):
#     stimProbas.append( np.array([np.sum(np.multiply(position_proba[n,:,:], stimPlaces[grp])) for n in range(len(position_proba))]) )
#     stimDistance.append( np.array([np.sqrt((stimCenters[grp][0] - position[n][0])**2 + (stimCenters[grp][1] - position[n][1])**2) for n in range(len(position_proba))]) )
#     thresholds = np.arange(0,100)/100
#     stimSelection.append( np.array([np.mean(stimDistance[grp][np.where(stimProbas[grp]>thresholds[t])]) for t in range(len(thresholds))]) )
#     plt.plot(stimSelection[grp], label='group '+str(grp))
# plt.axhline(20);
# plt.legend();
# plt.show();





# ## WE SHOULD TEST A PREDICTOR HERE !!!! 


# cleanProbas(position_proba)

X_proba = [np.sum(position_proba[n,:,:], axis=1) for n in range(len(position_proba))]
Y_proba = [np.sum(position_proba[n,:,:], axis=0) for n in range(len(position_proba))]
X_proba = [X_proba[n] if np.sum(X_proba[n]) > 0 else np.ones(45)/45 for n in range(len(position_proba))]
Y_proba = [Y_proba[n] if np.sum(Y_proba[n]) > 0 else np.ones(45)/45 for n in range(len(position_proba))]
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

X_standdev = np.sqrt([np.sum([X_proba[n][x]*(Bins[0][x]-X_guessed[n])**2 for x in range(Bins[0].size)]) for n in range(len(position_proba))])
Y_standdev = np.sqrt([np.sum([Y_proba[n][y]*(Bins[1][y]-Y_guessed[n])**2 for y in range(Bins[1].size)]) for n in range(len(position_proba))])
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
stuffbins = axb.hist(Error, 100, alpha=1, edgecolor='k')
plt.show(block=True)


# Overview
best_bins = np.argsort(Selected_errors)
frame_selection = range(len(Standdev))
frame_selection = np.union1d(
                        np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[0]],
                                                Standdev[:] < std_bins[best_bins[0]+1]))[0],
                        np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[1]],
                                                Standdev[:] < std_bins[best_bins[1]+1]))[0])

fig, ax = plt.subplots(figsize=(15,9))
ax1 = plt.subplot2grid((2,1),(0,0))
ax1.plot(np.array(position)[:,0], label='true X')
# ax1.plot(X_guessed, label='guessed X')
ax1.plot(frame_selection, np.array(X_guessed)[frame_selection], label='guessed X')
ax1.legend()
ax1.set_title('position X')

ax2 = plt.subplot2grid((2,1),(1,0), sharex=ax1)
ax2.plot(np.array(position)[:,1], label='true Y')
# ax2.plot(Y_guessed[:], label='guessed Y')
ax2.plot(frame_selection, np.array(Y_guessed)[frame_selection], label='guessed Y')
ax2.legend()
ax2.set_title('position Y')
plt.show(block=True)

bbbbb
# # # MOVIE
fig, ax = plt.subplots(figsize=(50,50))
best_bins = np.argsort(Selected_errors)
frame_selection = range(len(Standdev))
frame_selection = np.union1d(
                        np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[0]],
                                                Standdev[:] < std_bins[best_bins[0]+1]))[0],
                        np.where(np.logical_and(Standdev[:] >= std_bins[best_bins[1]],
                                                Standdev[:] < std_bins[best_bins[1]+1]))[0])

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

save_len = len(position_proba)
ani = animation.FuncAnimation(fig,updatefig,interval=50, save_count=save_len)
# ani.save(project_path+'_Movie.mp4')
fig.show()



