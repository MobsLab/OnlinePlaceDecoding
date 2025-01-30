import os
import sys
import datetime
import tables
import math
import random
import struct
import numpy as np
import tensorflow as tf
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import multiprocessing
from sklearn.neighbors import KernelDensity
from functools import reduce

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x/e_x.sum()

def next_batch(num, data, labels):
	""" Generates a random batch of matching data and labels """
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)
def shuffle(data, labels):
	return next_batch(len(data), data, labels)

def makeGaussian(size, fwhm = 3, center=None):
	""" Make a square gaussian kernel.
	size is the length of a side of the square
	fwhm is full-width-half-maximum, which
	can be thought of as an effective radius.
	"""
	
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]
	
	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]
	
	unnormalized = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
	return unnormalized / np.sum(unnormalized)

def kde2D(x, y, bandwidth, xbins=45j, ybins=45j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'epanechnikov')
	if ('edges' in kwargs):
		xx = kwargs['edges'][0]
		yy = kwargs['edges'][1]
	else:
		# create grid of sample locations (default: 45x45)
		xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]


	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T

	kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth)
	kde_skl.fit(xy_train)

	# score_samples() returns the log-likelihood of the samples
	z = np.exp(kde_skl.score_samples(xy_sample))
	zz = np.reshape(z, xx.shape)
	return xx, yy, zz/np.sum(zz)


class groupProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class groupContext(type(multiprocessing.get_context())):
    Process = groupProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class groupsPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = groupContext()
        super(groupsPool, self).__init__(*args, **kwargs)


def rateFunctions(clu_path, group, nChannels, 
	start_time, stop_time, end_time, 
	positions, position_time, speed, speed_cut, 
	Occupation_inverse, edges, bandwidth, kernel, samplingRate):
	learning_time = stop_time - start_time
	
	print('Starting data from group '+ str(group+1))
	if os.path.isfile(clu_path + 'clu.' +str(group+1)):
		with open(
					clu_path + 'clu.' + str(group+1), 'r') as fClu, open(
					clu_path + 'res.' + str(group+1), 'r') as fRes, open(
					clu_path + 'spk.' + str(group+1), 'rb') as fSpk:
			clu_str = fClu.readlines()
			res_str = fRes.readlines()
			nClusters = int(clu_str[0])

			spike_time      = np.array([[float(res_str[n])/samplingRate] for n in range(len(res_str))])
			dataSelection   = np.where(np.logical_and(
								spike_time[:,0] > start_time,
								spike_time[:,0] < end_time))[0]

			labels          = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(nClusters)] for n in dataSelection])
			spike_positions = np.array([positions[np.argmin(np.abs(spike_time[n]-position_time)),:] for n in dataSelection])
			spike_speed     = np.array([speed[np.min((np.argmin(np.abs(spike_time[n]-position_time)), len(speed)-1)),:] for n in dataSelection])

			
			n=0
			spikes = []
			fSpk.seek(dataSelection[0]*2*32*nChannels) #skip the first spikes, that are not going to be read
			spikeReader = struct.iter_unpack(str(32*nChannels)+'h', fSpk.read())
			for it in spikeReader:
				if n > len(dataSelection):
					break
				spike = np.reshape(np.array(it), [32,nChannels])
				spikes.append(np.transpose(spike))
				n = n+1
			spikes = np.array(spikes, dtype=float) * 0.195 # microvolts
			spike_time = spike_time[dataSelection]
	else:
		print('File ' + clu_path + 'clu.' + str(group+1) + ' not found.')
		return []
		


	trainingTimeSelection = np.where(np.logical_and(
		spike_time[:,0] > start_time, 
		spike_time[:,0] < stop_time))
	spikes_temp     =          spikes[trainingTimeSelection]
	labels_temp     =          labels[trainingTimeSelection]
	spike_speed     =     spike_speed[trainingTimeSelection]
	spike_positions = spike_positions[trainingTimeSelection]

	spikes = {'all':  spikes,
			  'train':spikes_temp[0:len(spikes_temp)*9//10,:,:],
			  'test': spikes_temp[len(spikes_temp)*9//10:len(spikes_temp),:,:]}
	labels = {'all':  labels,
			  'train':labels_temp[0:len(labels_temp)*9//10,:],
			  'test': labels_temp[len(labels_temp)*9//10:len(labels_temp),:]}
	

	### MARGINAL RATE FUNCTION
	selected_positions = spike_positions[np.where(spike_speed[:,0] > speed_cut)]
	_, _, MRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=edges, kernel=kernel)
	MRF[MRF==0] = np.min(MRF[MRF!=0])
	MRF         = MRF/np.sum(MRF)
	MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/learning_time

	### RATE FUNCTION

	with multiprocessing.Pool(np.shape(labels_temp)[1]) as p:
		Local_rate_functions = p.starmap( 
			rateFunction, 
			((label, labels_temp, 
			spike_positions, spike_speed, speed_cut, 
			bandwidth, edges, kernel, Occupation_inverse, learning_time) for label in range(np.shape(labels_temp)[1])))


	print('Finished data from group '+ str(group+1))
	return [nClusters, MRF, Local_rate_functions, spikes, spike_time, labels]


def rateFunction(label, labels, spike_positions, spike_speed, speed_cut, bandwidth, edges, kernel, Occupation_inverse, learning_time):
	selected_positions = spike_positions[np.where(np.logical_and( 
		spike_speed[:,0] > speed_cut, 
		labels[:,label] == 1))]
	if np.shape(selected_positions)[0]!=0:
		_, _, LRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=edges, kernel=kernel)
		LRF[LRF==0] = np.min(LRF[LRF!=0])
		LRF         = LRF/np.sum(LRF)
		return np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/learning_time
	else:
		return np.ones([45,45])








def extract_data(clu_path, list_channels, start_time, stop_time, end_time,
					speed_cut, samplingRate, 
					masking_factor, kernel, bandwidth):
	
	print('Extracting data.\n')

	with tables.open_file(os.path.dirname(clu_path) + '/nnBehavior.mat') as f:
		positions = f.root.behavior.positions
		speed = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		speed = np.swapaxes(speed[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)
	if stop_time == None:
		stop_time = position_time[-1]
	if bandwidth == None:
		bandwidth = (np.max(positions) - np.min(positions))/20


	### GLOBAL OCCUPATION
	if np.shape(speed)[0] != np.shape(positions)[0]:
		if np.shape(speed)[0] == np.shape(positions)[0] - 1:
			speed.append(speed[-1])
		elif np.shape(speed)[0] == np.shape(positions)[0] + 1:
			speed = speed[:-1]
		else:
			sys.exit(5)
	selected_positions = positions[np.where(np.logical_and.reduce( 
		[speed[:,0] > speed_cut, 
		position_time[:,0] > start_time, 
		position_time[:,0] < stop_time]))]
	xEdges, yEdges, Occupation = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, kernel=kernel)
	Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

	mask = Occupation > (np.max(Occupation)/masking_factor)
	Occupation_inverse = 1/Occupation
	Occupation_inverse[Occupation_inverse==np.inf] = 0
	Occupation_inverse = np.multiply(Occupation_inverse, mask)

	print('Behavior data extracted')



	totNClusters = 0
	nGroups = len(list_channels)
	channelsPerGroup = []
	clustersPerGroup = []

	spikes_all = []
	spikes_time = []
	spikes_train = []
	spikes_test = []
	labels_all = []
	labels_train = []
	labels_test = []

	Marginal_rate_functions = []
	Rate_functions = []

	### Extract
	undone_tetrodes = 0
	nCoresAvailable = multiprocessing.cpu_count() // 2 # We're mercifully leaving half of cpus for other processes.
	processingPools = [
		[pool*nCoresAvailable + group 
		for group in range(min(nCoresAvailable, nGroups-pool*nCoresAvailable))]
		for pool in range(nGroups//nCoresAvailable+1)]

	for pool in processingPools:
		if pool == []:
			continue
		with groupsPool(len(pool)) as p:
			Results = p.starmap(rateFunctions, 
				((clu_path, group, len(list_channels[group]), start_time, stop_time, end_time,
				positions, position_time, speed, speed_cut,
				Occupation_inverse, [xEdges, yEdges], bandwidth, kernel,
				samplingRate) for group in pool))

		for group in range(len(pool)):

			if Results[group] == []:
				undone_tetrodes += 1
				continue

			totNClusters += Results[group][0]
			clustersPerGroup.append(Results[group][0])
			channelsPerGroup.append(len(list_channels[group]))

			spikes_time.append(Results[group][4])
			spikes_all.append(Results[group][3]['all'])
			spikes_train.append(Results[group][3]['train'])
			spikes_test.append(Results[group][3]['test'])
			labels_all.append(Results[group][5]['all'])
			labels_train.append(Results[group][5]['train'])
			labels_test.append(Results[group][5]['test'])

			Marginal_rate_functions.append(Results[group][1])
			Rate_functions.append(Results[group][2])



	return {'nGroups':nGroups - undone_tetrodes, 'nClusters':totNClusters, 'clustersPerGroup':clustersPerGroup, 'channelsPerGroup':channelsPerGroup, 
				'positions':positions, 'position_time':position_time, 'speed':speed,  
				'spikes_all':spikes_all, 'spikes_time':spikes_time, 'labels_all':labels_all,
				'spikes_train':spikes_train, 'spikes_test':spikes_test, 'labels_train':labels_train, 'labels_test':labels_test, 
				'Occupation':Occupation, 'Mask':mask, 
				'Marginal_rate_functions':Marginal_rate_functions, 'Rate_functions':Rate_functions, 'Bins':[xEdges[:,0],yEdges[0,:]]}












def get_behaviour_labels(Data):
	
	behaviourLabels = []

	for group in range(Data['nGroups']):
		print()
		print('Calculating new labels for group '+str(group)+'.')
		grpBehaviourLabels = []
		for spk in range(len(Data['spikes_train'][group])):

			spkTime = Data['spikes_time'][group][spk][0]
			posIdx = np.argmin(np.abs(Data['position_time'] - spkTime))

			x = Data['positions'][posIdx, 0]
			y = Data['positions'][posIdx, 1]
			x = np.argmin(np.abs(x - Data['Bins'][0]))
			y = np.argmin(np.abs(y - Data['Bins'][1]))

			grpBehaviourLabels.append( 
				softmax(np.array(    [Data['Rate_functions'][group][clu][x,y] 
				for clu in range(Data['clustersPerGroup'][group])]    )) )

			if spk%300==0:
				sys.stdout.write('[%-30s] step : %d/%d' % ('='*(spk*30//len(Data['spikes_train'][group])),spk,len(Data['spikes_train'][group])))
				sys.stdout.write('\r')
				sys.stdout.flush()
		sys.stdout.write('[%-30s] step : %d/%d' % ('='*((spk+1)*30//len(Data['spikes_train'][group])),(spk+1),len(Data['spikes_train'][group])))
		sys.stdout.write('\r')
		sys.stdout.flush()

		behaviourLabels.append(np.array(grpBehaviourLabels))
	print()
	return behaviourLabels



def get_som_labels(Data):
	import sompy
	from tqdm import tqdm
	from tfrbm import BBRBM

	somLabels = []

	for group in range(Data['nGroups']):
		print('Calculating labels for group '+str(group)+'.')
		nChannels = Data['channelsPerGroup'][group]
		nClusters = Data['clustersPerGroup'][group]

		# RBM
		nHidden = 5
		spikesFlat = []
		for spk in tqdm(range(len(Data['spikes_train'][group]))):
			spikesFlat.append(Data['spikes_train'][group][spk,:,:].flatten())
			spikesFlat[-1] = (spikesFlat[-1] - np.min(spikesFlat[-1]))
			spikesFlat[-1] = spikesFlat[-1] / np.max(spikesFlat[-1])
		spikesFlat = np.array(spikesFlat, dtype=float).reshape([len(spikesFlat), nChannels, 32]).reshape([len(spikesFlat)*nChannels, 32])
		bbrbm = BBRBM(n_visible=32, n_hidden=nHidden, learning_rate=0.1, momentum=0.95, use_tqdm=True)
		errs = bbrbm.fit(spikesFlat, n_epoches=10, batch_size=50)
		spikesReconstructed = bbrbm.reconstruct(spikesFlat)
		spikesReduced = bbrbm.transform(spikesFlat).reshape([len(Data['spikes_train'][group]), nChannels*nHidden])
		print()

		# SOM
		mapsize = [100,100]
		som = sompy.SOMFactory.build(
			spikesReduced, 
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
		bmu = som.find_bmu(spikesReduced,njb=5)
		labelsPredicted = [cluMap[int(bmu[0,n])] for n in range(bmu.shape[1])]
		print()

		somLabels.append(labelsPredicted)

	return somLabels







def build_position_decoder(modules, Data, results_dir, nSteps):
	"""Trains one artificial neural network to guess position proba from spikes"""


	print('\nENCODING GRAPH\n')


	efficiencies = []
	convolutions = []
	n_tetrodes = Data['nGroups']
	# behaviourLabels = get_behavior_labels(Data)
	somLabels = get_som_labels(Data)

	sumConstantTerms = np.sum(Data['Marginal_rate_functions'], axis=0)
	allRateMaps = [np.log(Data['Rate_functions'][group][clu] + np.min(Data['Rate_functions'][group][clu][Data['Rate_functions'][group][clu]!=0])) 
					for group in range(n_tetrodes)
					for clu in range(Data['clustersPerGroup'][group])]
	allRateMaps = np.array(allRateMaps)


	##### BUILDING THE MODEL
	MOBSgraph = tf.Graph()
	with MOBSgraph.as_default():

		with tf.variable_scope("positionEncoder"):
			xBins                       = tf.constant(np.array(Data['Bins'][0]), shape=[45], name='xBins')
			yBins                       = tf.constant(np.array(Data['Bins'][1]), shape=[45], name='yBins')
			xTrue                       = tf.placeholder(tf.float64, name='xTrue')
			yTrue                       = tf.placeholder(tf.float64, name='yTrue')
			xIdx                        = tf.argmin( tf.abs(xBins - xTrue) )
			yIdx                        = tf.argmin( tf.abs(yBins - yTrue) )


		probasTensors = []
		for tetrode in range(n_tetrodes):

			with tf.variable_scope("group"+str(tetrode)+"-encoder"):

				x                   = tf.placeholder(tf.float32, shape=[None, Data['channelsPerGroup'][tetrode], 32],      name='x')
				# y                   = tf.placeholder(tf.float32, shape=[None, Data['clustersPerGroup'][tetrode]],          name='y')
				ySparse             = tf.placeholder(tf.int64,   shape=[None],                                             name='ySparse')

			spikeEncoder, ops = modules['mobs_NB'].layeredEncoder(x,Data['clustersPerGroup'][tetrode], Data['channelsPerGroup'][tetrode], size=200)
			convolutions.append(ops)

			with tf.variable_scope("group"+str(tetrode)+"-evaluator"):

				probas              = tf.nn.softmax(spikeEncoder, name='probas')
				probasTensors.append( tf.reduce_sum(probas, axis=0, name='sumProbas'))

				guesses             = tf.argmax(spikeEncoder,1, name='guesses')
				good_guesses        = tf.equal(ySparse, guesses)
				# good_guesses        = tf.equal(tf.argmax(y,1), guesses)
				accuracy            = tf.reduce_mean(tf.cast(good_guesses, tf.float32), name='accuracy')
				confusion_matrix    = tf.confusion_matrix(ySparse, guesses, name='confusion')
				# confusion_matrix    = tf.confusion_matrix(tf.argmax(y,1), guesses, name='confusion')
				
				# cross_entropy       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=spikeEncoder))
				cross_entropy       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ySparse, logits=spikeEncoder))
				crossTrain          = tf.train.AdamOptimizer(0.00004).minimize(cross_entropy, name='trainer')

		

		with tf.variable_scope("bayesianDecoder"):

			binTime                     = tf.placeholder(tf.float32, shape=[1], name='binTime')
			allProbas                   = tf.reshape(tf.concat(probasTensors, 0), [1, Data['nClusters']], name='allProbas');
			# allProbas = tf.placeholder(tf.float32, shape=[1, Data['nClusters']], name='allProbas')

			# Place map stats
			occMask                     = tf.constant(Data['Mask'],              dtype=tf.float64, shape=[45,45])
			constantTerm                = tf.constant(sumConstantTerms,          dtype=tf.float32, shape=[45,45])
			occMask_flat                = tf.reshape(occMask, [45*45])
			constantTerm_flat           = tf.reshape(constantTerm, [45*45])

			rateMaps                    = tf.constant(allRateMaps,               dtype=tf.float32, shape=[Data['nClusters'], 45,45], name='rateMaps')
			rateMaps_flat               = tf.reshape(rateMaps, [Data['nClusters'], 45*45])
			spikesWeight                = tf.matmul(allProbas, rateMaps_flat)

			allWeights                  = tf.cast( spikesWeight - binTime * constantTerm_flat, tf.float64 )
			allWeights_reduced          = allWeights - tf.reduce_mean(allWeights)

			positionProba_flat          = tf.multiply( tf.exp(allWeights_reduced), occMask_flat )
			positionProba               = tf.reshape(positionProba_flat / tf.reduce_sum(positionProba_flat), [45,45], name='positionProba')

			xProba                      = tf.reduce_sum(positionProba, axis=1, name='xProba')
			yProba                      = tf.reduce_sum(positionProba, axis=0, name='yProba')
			xGuessed                    = tf.reduce_sum(tf.multiply(xProba, xBins)) / tf.reduce_sum(xProba)
			yGuessed                    = tf.reduce_sum(tf.multiply(yProba, yBins)) / tf.reduce_sum(yProba)
			xStd                        = tf.sqrt(tf.reduce_sum(xProba*tf.square(xBins-xGuessed)))
			yStd                        = tf.sqrt(tf.reduce_sum(yProba*tf.square(yBins-yGuessed)))

			positionGuessed             = tf.stack([xGuessed, yGuessed], name='positionGuessed')
			standardDeviation           = tf.stack([xStd, yStd], name='standardDeviation')



		print('Tensorflow graph has been built and is ready to train.')



		### Train
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

			for tetrode in range(n_tetrodes):
				print('Learning clusters of group '+str(tetrode+1))

				# start convolutions from weights learned in previous layer
				if tetrode > 0:
					for op in range(len(convolutions[tetrode])):
						convolutions[tetrode][op].set_weights(convolutions[tetrode-1][op].get_weights())

				i=0
				for i in range(nSteps+1):
					batch = next_batch(80, Data['spikes_train'][tetrode], somLabels[tetrode])
					if i%50 == 0:
						curr_eval = sess.run([MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/accuracy:0')], 
												{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): batch[0], 
												MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/ySparse:0'): batch[1]})
												# MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): batch[1]})
						sys.stdout.write('[%-30s] step : %d/%d, efficiency : %g' % ('='*(i*30//nSteps),i,nSteps,curr_eval[0]))
						sys.stdout.write('\r')
						sys.stdout.flush()

					# training step
					MOBSgraph.get_operation_by_name('group'+str(tetrode)+'-evaluator/trainer').run(
									{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): batch[0], 
									MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/ySparse:0'): batch[1]}) 
									# MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): batch[1]}) 
									# MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/ySparse:0'): np.argmax(batch[1],axis=1)}) 

				# final_eval, confusion = sess.run([MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/accuracy:0'), 
				# 								  MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-evaluator/confusion/SparseTensorDenseAdd:0')], 
				# 								{MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/x:0'): Data['spikes_test'][tetrode], 
				# 								MOBSgraph.get_tensor_by_name('group'+str(tetrode)+'-encoder/y:0'): Data['labels_test'][tetrode]}) 
				# efficiencies.append(final_eval)
				# print('\nglobal efficiency : ', efficiencies[-1])
				# print('confusion : ')
				# print(confusion)
				print()
			

			saver.save(sess, results_dir + 'mobsGraph')
		

	return efficiencies

















def decode_position(modules, Data, results_dir, start_time, stop_time, bin_time):

	print('\nDECODING\n')

	n_tetrodes = Data['nGroups']

	decodedPositions = []
	truePositions = [] ; truePositions.append([0.,0.])
	nSpikes = []

	feedDictData = []
	feedDictTensors = []


	### Load the required tensors
	print('Restoring tensorflow graph.')
	tf.reset_default_graph()
	saver =                           tf.train.import_meta_graph(results_dir + 'mobsGraph.meta')

	feedDictTensors.append(           tf.get_default_graph().get_tensor_by_name("bayesianDecoder/binTime:0") )

	for tetrode in range(n_tetrodes):
		feedDictTensors.append(       tf.get_default_graph().get_tensor_by_name("group"+str(tetrode)+"-encoder/x:0") )

	positionProba =                   tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0")
	outputShape = positionProba.get_shape().as_list()
	neutralOutput = np.ones(outputShape) / np.product(outputShape)
	


	### Cut the data up
	nBins = math.floor((stop_time - start_time)/bin_time)
	print('Preparing data.')
	for bin in range(nBins):
		bin_start_time = start_time + bin*bin_time
		bin_stop_time = bin_start_time + bin_time

		feedDictDataBin = []
		feedDictDataBin.append([bin_time])

		for tetrode in range(n_tetrodes):
			spikes = Data['spikes_all'][tetrode][np.where(np.logical_and(
								Data['spikes_time'][tetrode][:,0] >= bin_start_time,
								Data['spikes_time'][tetrode][:,0] < bin_stop_time))]

			nSpikes.append(len(spikes))
			feedDictDataBin.append(spikes)

		feedDictData.append(feedDictDataBin)

		position_idx = np.argmin(np.abs(bin_stop_time-Data['position_time']))
		position_bin = Data['positions'][position_idx,:]
		truePositions.append( truePositions[-1] if np.isnan(position_bin).any() else position_bin )

		if bin%10==0:
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()

	truePositions.pop(0)
	print("Data is prepared. We're sending it through the tensorflow graph.")


	# Send the spiking data through the tensorflow graph
	emptyBins = 0
	with tf.Session() as sess:
		saver.restore(sess, results_dir + 'mobsGraph')

		for bin in range(nBins):
			decodedPositions.append(positionProba.eval({i:j for i,j in zip(feedDictTensors, feedDictData[bin])}))
			if np.isnan(decodedPositions[-1].sum()):
				decodedPositions.pop()
				nSpikes.pop(len(decodedPositions))
				truePositions.pop(len(decodedPositions))

			if bin%10==0:
				sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
				sys.stdout.write('\r')
				sys.stdout.flush()
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*((bin+1)*30//nBins),bin+1,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()

	print('\nfinished.')
	return decodedPositions, truePositions, nSpikes









def auto_decode_position(modules, Data, results_dir, start_time, stop_time, bin_time):

	n_tetrodes = Data['nGroups']
	decodedPositions = []
	rateValues = []
	truePositions = [] ; truePositions.append([0.,0.])
	nSpikes = []

	### Load the required tensors
	print('Restoring tensorflow graph.')
	tf.reset_default_graph()
	saver =                           tf.train.import_meta_graph(results_dir + 'mobsGraph.meta')

	allProbas = tf.get_default_graph().get_tensor_by_name("bayesianDecoder/allProbas:0")
	binTime = tf.get_default_graph().get_tensor_by_name("bayesianDecoder/binTime:0")


	positionProba =                   tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionProba:0")
	outputShape = positionProba.get_shape().as_list()
	neutralOutput = np.ones(outputShape) / np.product(outputShape)


	### Cut the data up
	nBins = math.floor((stop_time - start_time)/bin_time)
	print('Preparing data.')
	for bin in range(nBins):
		bin_start_time = start_time + bin*bin_time
		bin_stop_time = bin_start_time + bin_time

		position_idx = np.argmin(np.abs(bin_stop_time-Data['position_time']))
		position_bin = Data['positions'][position_idx,:]
		truePositions.append( truePositions[-1] if np.isnan(position_bin).any() else position_bin )
		x = np.argmin(np.abs(truePositions[-1][0] - Data['Bins'][0]))
		y = np.argmin(np.abs(truePositions[-1][1] - Data['Bins'][1]))

		tetSpikes = []
		binCluValues = []
		for tetrode in range(n_tetrodes):
			spikes = Data['spikes_all'][tetrode][np.where(np.logical_and(
								Data['spikes_time'][tetrode][:,0] >= bin_start_time,
								Data['spikes_time'][tetrode][:,0] < bin_stop_time))]
			tetSpikes.append(len(spikes))
			
			tetCluValues = len(spikes) * softmax(np.array(    [Data['Rate_functions'][tetrode][clu][x,y] 
				for clu in range(Data['clustersPerGroup'][tetrode])]    ))
			binCluValues += list(tetCluValues)
		rateValues.append([binCluValues])
		nSpikes.append(tetSpikes)


		if bin%10==0:
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()

	truePositions.pop(0)
	print("Data is prepared. We're sending it through the tensorflow graph.")


	errors = []
	with tf.Session() as sess:
		saver.restore(sess, results_dir + 'mobsGraph')

		for bin in range(nBins):
			
			decodedPositions.append(positionProba.eval({allProbas : rateValues[bin], binTime : [bin_time]}))
			if np.isnan(decodedPositions[-1].sum()):
				decodedPositions.pop()
				truePositions.pop(len(decodedPositions))
				nSpikes.pop(len(decodedPositions))
				errors.append(bin)
			if bin%10==0:
				sys.stdout.write('[%-30s] step : %d/%d' % ('='*(bin*30//nBins),bin,nBins))
				sys.stdout.write('\r')
				sys.stdout.flush()
			sys.stdout.write('[%-30s] step : %d/%d' % ('='*((bin+1)*30//nBins),bin+1,nBins))
			sys.stdout.write('\r')
			sys.stdout.flush()

	print('\nfinished.')
	print(str(len(errors))+' errors.')
	return decodedPositions, truePositions, nSpikes, errors




if __name__ == '__main__':
	print(0)
	sys.exit(0)


    