import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce




filePath = '/home/mobshamilton/Documents/dataset/RatCatanese/mobsEncoding3/'

print('loading data')
Data = np.load(filePath + '_data.npy').item()
print('done.')

# tf.reset_default_graph()
# saver = tf.train.import_meta_graph(filePath + 'mobsGraph.meta')
# 
# spike2 = tf.get_default_graph().get_tensor_by_name("group2/spikeEncoder/x:0")
# kp2    = tf.get_default_graph().get_tensor_by_name("group2/spikeEncoder/keep_proba:0")
# label2 = tf.get_default_graph().get_tensor_by_name("group2/spikeEvaluator/probas:0")
# 
# rateMaps = tf.get_default_graph().get_tensor_by_name("bayesianDecoder/rateMaps:0")
# 
# positionProba = tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0")
# 
# spikes = Data['spikes_test'][2][:5,:,:]
# labels = Data['labels_test'][2][:5,:]
# 
# sumConstantTerms = np.sum(Data['Marginal_rate_functions'], axis=0)
# newRM = Data['Rate_functions']
# allRateMaps = [np.log(Data['Rate_functions'][group][clu] + np.min(Data['Rate_functions'][group][clu][Data['Rate_functions'][group][clu]!=0])) for group in range(Data['nGroups']) for clu in range(Data['clustersPerGroup'][group])]
# allRateMaps = np.array(allRateMaps)
# 
# tens = [spike2, kp2]
# dat = [spikes, 1.]
# 
# with tf.Session() as sess:
# 	saver.restore(sess, filePath + 'mobsGraph')
# 	RMarray = rateMaps.eval()
# 	
# 	guessedlabels = label2.eval({i:j for i,j in zip(tens,dat)})
# 
# print(labels)
# print(guessedlabels)


# bayes_matrices = np.load('/home/mobshamilton/Documents/dataset/Mouse-743-all/2018-12-05_12:55_bayes.npy').item()
# Rate_functions = bayes_matrices['Rate functions']
# log_RF = []
# for tetrode in range(np.shape(Rate_functions)[0]):
# 	temp = []
# 	for cluster in range(np.shape(Rate_functions[tetrode])[0]):
# 		temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster]!=0])))
# 	log_RF.append(temp)
	
	
# old = np.load('/home/mobshamilton/Documents/dataset/RatCatanese/2018-10-23_19:47_simDecoding.npz')
# new = np.load('/home/mobshamilton/Documents/dataset/RatCatanese/mobsEncoding_2018-12-11_20:33/_simDecoding.npz')
# 
# OccupationOld = old['arr_0']
# OccupationNew = new['arr_0']
# position_probaOld = old['arr_1']
# position_probaNew = new['arr_1']








bandwidth = 5

BinsX = Data['Bins'][0]
BinsY = Data['Bins'][1]

rateFunctionsPoles = []
for tet in range(len(Data['Rate_functions'])):
	lrf = []
	for clu in range(len(Data['Rate_functions'][tet])):
		x_proba = np.sum(Data['Rate_functions'][tet][clu], axis=1)
		y_proba = np.sum(Data['Rate_functions'][tet][clu], axis=0)
		x = np.average(Data['Bins'][0], weights=x_proba)
		y = np.average(Data['Bins'][1], weights=y_proba)
		lrf.append([x,y])
	rateFunctionsPoles.append(lrf)

spikes_train = Data['spikes_train'][0]
labels_train = Data['labels_train'][0]
extLabels_train = Data['extLabels_train'][0]


labels = [np.argmax(Data['labels_train'][tet], axis=1) for tet in range(len(Data['labels_train']))]
sortingLabels = [np.argsort(Data['extLabels_train'][tet], axis=1) for tet in range(len(Data['extLabels_train']))]
extLabels = [sortingLabels[tet][:,-1] for tet in range(len(Data['extLabels_train']))]
extLabels2 = [sortingLabels[tet][:,-2] for tet in range(len(Data['extLabels_train']))]
extLabels3 = [sortingLabels[tet][:,-3] for tet in range(len(Data['extLabels_train']))]

behaviouraldistances = []
behaviouraldistances2 = []
behaviouraldistances3 = []
randomdistances = []
for tet in range(len(extLabels)):
	nClu = len(Data['Rate_functions'][tet])
	nSteps = len(extLabels[tet])
	for event in range(nSteps):
		randlabel = np.random.randint(nClu)
		behaviouraldistances.append(np.sqrt((rateFunctionsPoles[tet][extLabels[tet][event]][0] - 
										rateFunctionsPoles[tet][labels[tet][event]][0])**2 +
										(rateFunctionsPoles[tet][extLabels[tet][event]][1] -
										rateFunctionsPoles[tet][labels[tet][event]][1])**2))
		behaviouraldistances2.append(np.sqrt((rateFunctionsPoles[tet][extLabels2[tet][event]][0] - 
										rateFunctionsPoles[tet][labels[tet][event]][0])**2 +
										(rateFunctionsPoles[tet][extLabels2[tet][event]][1] -
										rateFunctionsPoles[tet][labels[tet][event]][1])**2))
		behaviouraldistances3.append(np.sqrt((rateFunctionsPoles[tet][extLabels3[tet][event]][0] - 
										rateFunctionsPoles[tet][labels[tet][event]][0])**2 +
										(rateFunctionsPoles[tet][extLabels3[tet][event]][1] -
										rateFunctionsPoles[tet][labels[tet][event]][1])**2))
		randomdistances.append(np.sqrt((rateFunctionsPoles[tet][randlabel][0] - 
										rateFunctionsPoles[tet][labels[tet][event]][0])**2 +
										(rateFunctionsPoles[tet][randlabel][1] -
										rateFunctionsPoles[tet][labels[tet][event]][1])**2))


plt.figure(figsize=(20,20))
_, bins,_ = plt.hist(randomdistances, 150, alpha=0.5, edgecolor='k', label='random')
plt.hist(behaviouraldistances, bins, alpha=0.5, edgecolor='k', label='extracted')
plt.legend(loc='upper right')
plt.yscale('log', nonposy='clip')
plt.title('Distance between the true place field and the maximum likelihood placefield')
plt.show()

plt.figure(figsize=(20,20))
_, bins,_ = plt.hist(randomdistances, 150, alpha=0.5, edgecolor='k', label='random')
plt.hist(behaviouraldistances2, bins, alpha=0.5, edgecolor='k', label='extracted')
plt.legend(loc='upper right')
plt.yscale('log', nonposy='clip')
plt.title('Distance between the true place field and the second maximum likelihood placefield')
plt.show()

plt.figure(figsize=(20,20))
_, bins,_ = plt.hist(randomdistances, 150, alpha=0.5, edgecolor='k', label='random')
plt.hist(behaviouraldistances3, bins, alpha=0.5, edgecolor='k', label='extracted')
plt.legend(loc='upper right')
plt.yscale('log', nonposy='clip')
plt.title('Distance between the true place field and the third maximum likelihood placefield')
plt.show()