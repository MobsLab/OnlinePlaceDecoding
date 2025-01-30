import os
# import sys
import struct
# import tables
import sompy
import numpy as np
# # from sklearn.decomposition import PCA
# import matplotlib
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# # from tensorflow import keras
from tqdm import tqdm
from tfrbm import BBRBM
# from collections import Counter
from contextlib import ExitStack
import xml.etree.ElementTree as ET

class add_path():
	def __init__(self, path):
		self.path = path

	def __enter__(self):
		sys.path.insert(0, self.path)

	def __exit__(self, exc_type, exc_value, traceback):
		sys.path.remove(self.path)

networkTools = __import__("mobs_networkbuilding")


def next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def swapColumns(array, col1, col2):
	array[:, col1], array[:, col2] = array[:,col2], array[:,col1].copy()
def swapRows(array, row1, row2):
	array[row1,:], array[row2,:] = array[row2,:], array[row1,:].copy()
def swapBasisElement(array, idx1, idx2):
	swapColumns(array, idx1, idx2)
	swapRows(array, idx1, idx2)
def firstZeroInRow(array, row, starting=0 ,idx=None):
	if idx==None:
		idx = starting
	if idx == len(array):
		return row
	if array[row,idx]==0:
		return idx
	else:
		return firstZeroInRow(array, row, idx=idx+1)

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def prettyConfusion(confusion):
	if confusion.shape[0] is not confusion.shape[1]:
		raise ValueError
	for clu in range(len(confusion)):
		maxcoord = np.unravel_index(confusion[clu:,clu:].argmax(), confusion[clu:,clu:].shape)
		swapRows(confusion,clu,maxcoord[0]+clu)	
		swapColumns(confusion,clu,maxcoord[1]+clu)
	return






rep = lambda str : str if str[-1]=='/' or len(str)==1 else rep(str[:-1])
baseName = os.path.expanduser('~/Documents/dataset/RatCatanese/rat122-20090731.')
folder = rep(baseName)


list_channels = []
tree = ET.parse(baseName+'xml')
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
			group = []
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
			samplingRate = float(br2Elem.text)
		if br2Elem.tag == 'nChannels':
			nChannels = int(br2Elem.text)


accuracies = []
startT = 1180
stopT = 3300

# with tables.open_file(folder + '/nnBehavior.mat') as f:
# 	positions = f.root.behavior.positions
# 	speed = f.root.behavior.speed
# 	position_time = f.root.behavior.position_time
# 	positions = np.swapaxes(positions[:,:],1,0)
# 	speed = np.swapaxes(speed[:,:],1,0)
# 	position_time = np.swapaxes(position_time[:,:],1,0)

# for group in range(1):
# 	group = 9
for group in range(len(list_channels)):
	spikes = []
	spikesFlat = []
	if not (os.path.isfile(baseName + 'clu.' + str(group+1))):
		print('file '+baseName+'clu.'+str(group+1)+' does not exist')
		continue

	nChannels = len(list_channels[group])




	print('READING DATA')
	with ExitStack() as stack:
		fClu = stack.enter_context(open(baseName + 'clu.' +str(group+1), 'r'))
		fRes = stack.enter_context(open(baseName + 'res.' +str(group+1), 'r'))
		fSpk = stack.enter_context(open(baseName + 'spk.' +str(group+1), 'rb'))

		cluStr = fClu.readlines()
		resStr = fRes.readlines()
		length = len(resStr)
		nClusters = int(cluStr[0])

		spike_time = np.array([[float(resStr[n])/samplingRate] for n in range(length)])
		dataSelection = np.where(np.logical_and(
			spike_time[:,0] > startT,
			spike_time[:,0] < stopT))[0]

		labels          = np.array([[1. if int(cluStr[n+1])==l else 0.for l in range(nClusters)] for n in dataSelection])
		labelsSparse    = np.array([int(cluStr[n+1]) for n in dataSelection])
		# spike_positions = np.array([positions[np.argmin(np.abs(spike_time[n]-position_time)),:] for n in dataSelection])
		# spike_speed     = np.array([speed[np.min((np.argmin(np.abs(spike_time[n]-position_time)), len(speed)-1)),:] for n in dataSelection])

		fSpk.seek(dataSelection[0]*2*32*nChannels)
		spikeReader = struct.iter_unpack(str(32*nChannels)+'h', fSpk.read())
		for n in tqdm(range(len(dataSelection))):
			spike = np.reshape(np.array(next(spikeReader)), [32, nChannels])
			spike = np.transpose(spike) * 0.195 # microvolts
			spikes.append(spike)
			spikesFlat.append(spike.flatten())
			spikesFlat[-1] = (spikesFlat[-1] - np.min(spikesFlat[-1]))
			spikesFlat[-1] = spikesFlat[-1] / np.max(spikesFlat[-1])
		spikes = np.array(spikes, dtype=float)
		spikesFlat = np.array(spikesFlat, dtype=float)

	spikesTrain = spikes[0:len(spikes)*9//10,:,:]
	labelsTrain = labels[0:len(spikes)*9//10,:]
	labelsTrainSparse = labelsSparse[0:len(spikes)*9//10]

	spikesTest = spikes[len(spikes)*9//10:,:,:]
	labelsTest = labels[len(spikes)*9//10:,:]
	labelsTestSparse = labelsSparse[len(spikes)*9//10:]
	print()

	







	# print('BUILDING GRAPH')
	# with tf.name_scope("group"+str(group)):
	# 	with tf.name_scope("spikeEncoder"):

	# 		x = tf.placeholder(tf.float32, shape=[None, len(list_channels[group]), 32], name='x')
	# 		y = tf.placeholder(tf.float32, shape=[None, nClusters], name='y')
	# 		ySparse = tf.placeholder(tf.int32, shape=[None], name='ySparse')
	# 		keepProba = tf.placeholder(tf.float32, name='keepProba')

	# 		encoder = networkTools.encoder(x,y,keepProba, size = 64)

	# 	with tf.name_scope("spikeEvaluator"):

	# 		guesses = tf.argmax(encoder['Output'],1,name='guesses')
	# 		goodGuesses = tf.equal(tf.argmax(y,1), guesses)
	# 		accuracy = tf.reduce_mean(tf.cast(goodGuesses, tf.float32), name='accuracy')
	# 		confusionMatrix = tf.confusion_matrix(tf.argmax(y,1), guesses, name='confusion')

	# 		crossEntropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ySparse, logits=encoder['Output']))
	# 		crossTrain = tf.train.AdamOptimizer(0.00004).minimize(crossEntropy, name='trainer')
	# print()







	# print('TRAINING GRAPH')
	# nSteps = 20000
	# with tf.Session() as sess:
	# 	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

	# 	for i in tqdm(range(nSteps)):
	# 		batch = next_batch(50, spikesTrain, labelsTrainSparse)
	# 		crossTrain.run({x: batch[0], ySparse: batch[1], keepProba: 0.5})

	# 	efficiency, confusion = sess.run([accuracy, confusionMatrix], {x: spikesTest, y: labelsTest, keepProba: 1})
	# 	print('Accuracy : ', efficiency)
	# 	print('confusion : ')
	# 	print(confusion)
	# 	print()
	# # Accuracy :  0.9820879
	# # confusion : 
	# # [[ 1343     9    46     3    45    10     7    61    12    46]
	# #  [   12    62     0     0     0     0    47     0     0     0]
	# #  [    7     0  2506     1     0     4     0     2     0     0]
	# #  [    6     0    12    40     1     2     0     0     0     1]
	# #  [    2     0     0     0  1034     0     0     0     0     0]
	# #  [    0     0    10     0     0   703     0     0     0     2]
	# #  [    7     8     2     0     0     0  3927     1    13     0]
	# #  [   34     0     7     0     0     0     0  1393     0     0]
	# #  [    3     0     0     0     0     0     4     0   308     0]
	# #  [    3     0     4     0     0    19     1     0     0 13576]]
	# # Idea : check if the misclassified spikes are always the same and if they look like the shape of the other cluster


	







	print('REDUCING DIMENSIONS')
	try:
		del bbrbm
	except NameError:
		print('new network')
	# 0.0058
	nHidden = 5
	spikesFlat = spikesFlat.reshape([len(spikes), nChannels, 32]).reshape([len(spikes)*nChannels, 32])
	spikesTrain = spikesFlat[0:(len(spikesFlat)*9//10),:]
	bbrbm = BBRBM(n_visible=32, n_hidden=nHidden, learning_rate=0.1, momentum=0.95, use_tqdm=True)
	errs = bbrbm.fit(spikesTrain, n_epoches=10, batch_size=50)
	spikesReconstructed = bbrbm.reconstruct(spikesFlat)
	spikesReduced = bbrbm.transform(spikesFlat).reshape([len(spikes), nChannels*nHidden])
	spikesTrain = spikesReduced[0:len(spikesReduced)*9//10,:]
	spikesTest = spikesReduced[len(spikesReduced)*9//10:,:]
	print()



	# print('SUPERVISED KNN CLASSIFIER')
	# classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
	# classifier.fit(spikesTrain, labelsTrainSparse)

	# labelsPredicted = classifier.predict(spikesTest)
	# accuracy = np.mean([x==y for x,y in zip(labelsPredicted, labelsTestSparse)])
	# print('accuracy : ', accuracy)
	# confusion = confusion_matrix(labelsTestSparse, labelsPredicted)
	# print(confusion)
	# print()


	# print('PCA')
	# try:
	# 	del pca
	# except NameError:
	# 	pass
	# pca = PCA(n_components=5)
	# spikesTrain = pca.fit_transform(spikesTrain)
	# print()

#[0.51161569759810999, 0.41073125839204622, 0.29994891878086155, 0.26374375988341447, 0.59016244668300211, 0.5365180536846601, 0.51086294991012415, 0.6058682789151707, 0.63025414331069896, 0.42282411800224273, 0.22314349169070533, 0.37245850594172053]
	print('SOM')
	try:
		del som
	except NameError:
		pass
	mapsize = [100,100]
	som = sompy.SOMFactory.build(
		spikesTrain, 
		mapsize, 
		mask=None, 
		mapshape='planar', 
		lattice='rect', 
		normalization='var', 
		initialization='pca',
		neighborhood='gaussian',
		training='batch',
		name='sompy')
	som.train(n_job=5, verbose='info')
	cluMap = som.cluster(n_clusters=nClusters)
	bmu = som.find_bmu(spikesTrain,njb=5)
	labelsPredicted = [cluMap[int(bmu[0,n])] for n in range(bmu.shape[1])]
	print()


	# print('UNSUPERVISED KNN BASED CLASSIFIER')
	# import sklearn.neighbors
	# import sklearn.cluster
	from sklearn.metrics import confusion_matrix
	# try:
	# 	del model
	# except NameError:
	# 	pass
	# nNeighbors = 5
	# # classifier = sklearn.neighbors.NearestNeighbors(n_neighbors=nNeighbors, algorithm='ball_tree').fit(spikesTrain[:300,:])
	# # _, indices = classifier.kneighbors(spikesTrain[:300,:])

	# # Unsupervised accuracy 0.55 to 0.65, lr=0.5, nepoch=3,nneighbors=5, link=ward
	# # went to 0.70 once with lr=0.2 and nEpoch=10
	cutoff=len(spikesTrain)
	# sparseGraph = sklearn.neighbors.kneighbors_graph(spikesTrain[:cutoff,:], nNeighbors, include_self=False)
	# model = sklearn.cluster.AgglomerativeClustering(linkage='ward', connectivity=sparseGraph, n_clusters=nClusters)
	# model.fit(spikesTrain[:cutoff,:])
	#confusion = confusion_matrix(labelsTrainSparse[:cutoff], model.labels_)
	confusion = confusion_matrix(labelsTrainSparse[:cutoff], labelsPredicted)
	prettyConfusion(confusion)
	print('accuracy : ', np.trace(confusion)/np.sum(confusion))
	print(confusion)
	accuracies.append(np.trace(confusion)/np.sum(confusion))


	# # Reexpress sparseMatrix closer to block-diagonal form. Blocks of size nNeighbors
	# sparseMatrix = classifier.kneighbors_graph(spikesTrain[:300,:]).toarray()
	# np.fill_diagonal(sparseMatrix, 2)
	# print('Neighbors computed.')
	# basisElement = 0
	# pbar = tqdm(total=len(sparseMatrix))
	# while basisElement < len(sparseMatrix):
	# 	neighboringElements = [n for n in range(basisElement, len(sparseMatrix)) if sparseMatrix[basisElement,n]==1]
	# 	for n in neighboringElements:
	# 		swapBasisElement(sparseMatrix, n, firstZeroInRow(sparseMatrix, basisElement, starting=basisElement))
	# 	basisElement = basisElement + len(neighboringElements) + 1
	# 	pbar.update(len(neighboringElements)+1)
	# pbar.close()
	# plt.imshow(sparseMatrix)
	# plt.show()



	## Somewhat naïvely separates clusters. Not really working but close.
	# classifier = sklearn.cluster.SpectralClustering(n_clusters=10).fit(spikesTrain[:1000,:])
	# labelsPredicted = classifier.fit_predict(spikesTest[:1000,:])
	# print(confusion_matrix(labelsTestSparse[:1000], labelsPredicted))
	print()

print('accuracies : ', accuracies)







# results with BBRBM(n_visible=32, n_hidden=nHidden, learning_rate=0.1, momentum=0.95, use_tqdm=True).fit(spikesTrain, n_epoches=1, batch_size=50)
# and sklearn.neighbors.KNeighborsClassifier(n_neighbors=5) on each group (every spike of the dataset) :
# [0.78,0.82,0.77,0.73,0.89,0.78,0.92,0.92,0.97,0.77,0.79,0.83]

# learning_rate=0.5, n_epoches=3, data selected :
# [0.82,0.81,0.77,0.71,0.92,0.81,0.92,0.94,0.97,0.76,0.77,0.80]









# print('BUILD RECOVERY MODEL')
# spikesTrain = spikesReduced[0:len(spikesReduced)*9//10,:]
# spikesTest = spikesReduced[len(spikesReduced)*9//10:,:]

# model = keras.Sequential([
# 	keras.layers.Dense(100, input_shape=(nChannels*nHidden,), activation=tf.nn.leaky_relu),
# 	keras.layers.Dense(100, activation=tf.nn.leaky_relu),
# 	keras.layers.Dropout(.5),
# 	keras.layers.Dense(100, activation=tf.nn.leaky_relu),
# 	keras.layers.Dense(nClusters, activation=tf.nn.softmax)])

# model.compile(optimizer='adam',
# 	loss='sparse_categorical_crossentropy',
# 	metrics=['accuracy'])

# model.fit(spikesTrain, labelsTrainSparse, epochs=5)

# with open(os.path.expanduser('~/Documents/dataset/RatCatanese/rat122-20090731.keras.json'), 'w') as fout:
# 	fout.write(model.to_json())
# model.save_weights(os.path.expanduser('~/Documents/dataset/RatCatanese/rat122-20090731.keras.h5'), overwrite=True)

# print()
# test_loss, test_acc = model.evaluate(spikesTest, labelsTestSparse)
# print('test accuracy : ', test_acc)
# print()










