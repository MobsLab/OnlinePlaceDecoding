import os
import sys
import datetime
import tables
import math
import random
import struct
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from functools import reduce



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

def normalize_spikes(spikes):
	shape = np.shape(spikes)
	normalized_spikes = spikes
	for x in range(shape[0]):
		norm_fact = np.abs(np.sum(spikes[x,:,:]))
		normalized_spikes[x,:,:] = spikes[x,:,:]/norm_fact
	return normalized_spikes

def shuffle_labels(labels, perc=0.5):
	n = int(len(labels)*perc)
	data = list(labels)
	idx = list(range(len(data)))
	random.shuffle(idx)
	idx = idx[:n]
	mapping = dict((idx[i], idx[i-1]) for i in range(n))
	return np.array([data[mapping.get(x,x)] for x in range(len(data))])

def randomize_labels(labels, perc=0.5):
	labels2 = labels
	n_labels = len(labels)
	n_clu = np.shape(labels)[1]
	n_rand = int(len(labels)*perc)

	idx = list(range(n_labels))
	random.shuffle(idx)
	idx = idx[:n_rand]
	for i in idx:
		true_label = np.argmax(labels2[i,:])
		fake_label = true_label
		while true_label == fake_label:
			random.shuffle(labels2[i,:])
			fake_label = np.argmax(labels2[i,:])

	return labels2
	

def modify_labels(labels,clu_modifier=1):

	if clu_modifier==1 :
		return labels

	elif clu_modifier==2 :
		n = len(labels)
		idx = np.concatenate(([0]*(n//2), [1]*(n-n//2)))
		random.shuffle(idx)
		cluA = np.array([labels[label,:]*   idx[label]  for label in range(len(idx))])
		cluB = np.array([labels[label,:]*(1-idx[label]) for label in range(len(idx))])
		return np.concatenate((cluA,cluB),1)

	elif clu_modifier==0.5 :
		n = len(labels)
		n_clu = np.shape(labels)[1]
		idx = list(range(n_clu))
		# random.shuffle(idx)
		labels2 = np.ndarray([n, (n_clu+1)//2])
		labels2 = [[(labels[x,idx[2*n]] + labels[x,idx[2*n+1]] 
			if (2*n+1<=len(idx)-1) 
			else labels[x,idx[2*n]])  
			for n in range(np.shape(labels2)[1])] 
			for x in range(n)]
		return np.array(labels2)








def neural_encode(modules, project_path, results_dir, list_channels, samplingRate, **keyword_parameters):
	"""Trains one artificial neural network per electrode group to classify spikes in clusters (with solution).
		retiring december 2018."""


	print('\nLEARNING CLUSTERS\n')
	start_time           = keyword_parameters.get('start_time',   0)
	stop_time            = keyword_parameters.get('stop_time',    1e10)
	random_param         = keyword_parameters.get('rand_param',   0)
	cluster_modifier     = keyword_parameters.get('cluster_modifier', 1)
	nSteps               = keyword_parameters.get('nSteps', 30000)

	clu_path = project_path[:len(project_path)-3]
	
	efficiencies = []
	n_tetrodes = len(list_channels)
	for tetrode in range(n_tetrodes):
		
		### Extract
		if os.path.isfile(clu_path + 'clu.' + str(tetrode+1)):
			with open(
						clu_path + 'clu.' + str(tetrode+1), 'r') as fClu, open(
						clu_path + 'res.' + str(tetrode+1), 'r') as fRes, open(
						clu_path + 'spk.' + str(tetrode+1), 'rb') as fSpk:
				clu_str = fClu.readlines()
				res_str = fRes.readlines()
				n_clu = int(clu_str[0])-1
				n_channels = len(list_channels[tetrode])
				spikeReader = struct.iter_unpack(str(32*n_channels)+'h', fSpk.read())

				# labels = np.array([[1. if int(clu_str[n+1])-1==l else 0. for l in range(n_clu)] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				# spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
				labels = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)])
				spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)])
				n=0
				spikes = []
				for it in spikeReader:
					# if int(clu_str[n+1])!=0:
					# 	spike = np.reshape(np.array(it), [32,n_channels])
					# 	spikes.append(np.transpose(spike))
					spike = np.reshape(np.array(it), [32,n_channels])
					spikes.append(np.transpose(spike))
					n = n+1
				spikes = np.array(spikes, dtype=float) * 0.195 # microvolts
		else:
			continue
		print('File from group ' + str(tetrode+1) + ' has been succesfully opened.')


		### Format

		spikes = spikes[np.intersect1d(
			np.where(spike_time[:,0] > start_time), 
			np.where(spike_time[:,0] < stop_time))]
		labels = labels[np.intersect1d(
			np.where(spike_time[:,0] > start_time), 
			np.where(spike_time[:,0] < stop_time))]

		spikes_train = spikes[0:len(spikes)*9//10,:,:]
		spikes_test = spikes[len(spikes)*9//10:len(spikes),:,:]
		labels_train = labels[0:len(labels)*9//10,:]
		labels_test = labels[len(labels)*9//10:len(labels),:]

		labels_train = shuffle_labels(labels_train, random_param)





		##### BUILDING THE MODEL
		tf.Graph()
		x            = tf.placeholder(tf.float32, shape=[None, np.shape(spikes)[1], 32],          name='x'+str(tetrode+1))
		y            = tf.placeholder(tf.float32, shape=[None, np.shape(labels)[1]],              name='y'+str(tetrode+1))
		keep_proba   = tf.placeholder(tf.float32,                                                 name='keep_proba'+str(tetrode+1))

		Graph        = modules['mobs_NB'].encoder(x,y,keep_proba, size=64)




		### Evaluation metrics
		probas =             tf.nn.softmax(Graph['Output'], name='probas'+str(tetrode+1))
		sumOutput    =       tf.reduce_sum(probas, axis=0, name='sumProbas'+str(tetrode+1))

		guesses =            tf.argmax(Graph['Output'],1, name='guesses'+str(tetrode+1))
		good_guesses =       tf.equal(tf.argmax(y,1), guesses)
		accuracy =           tf.reduce_mean(tf.cast(good_guesses, tf.float32), name='accuracy'+str(tetrode+1))
		confusion_matrix =   tf.confusion_matrix(tf.argmax(y,1), guesses)

		def covariance(x, y):
			return tf.reduce_mean((x - tf.reduce_mean(x)) * (y - tf.reduce_mean(y)))
		co1 = covariance(Graph['Output'],y)
		co2 = covariance(Graph['Output'],Graph['Output'])
		co3 = covariance(y,y)
		loss = -1 * co1 / tf.sqrt(co2 * co3)
		cross_entropy =      tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Graph['Output']))
		train =              tf.train.AdamOptimizer(0.00002).minimize(loss)
		crossTrain =         tf.train.AdamOptimizer(0.00004).minimize(cross_entropy)

		
		






		### Train
		i=0
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

			
			for i in range(nSteps+1):
				batch = next_batch(80, spikes_train, labels_train)
				if i%50 == 0:
					curr_eval = sess.run([accuracy], {x: batch[0], y: batch[1], keep_proba: 1.})
					sys.stdout.write('[%-30s] step : %d/%d, efficiency : %g' % ('='*(i*30//nSteps),i,nSteps,curr_eval[0]))
					sys.stdout.write('\r')
					sys.stdout.flush()
				crossTrain.run({x: batch[0], y: batch[1], keep_proba: 0.5})


			final_eval, confusion = sess.run([accuracy, confusion_matrix], {x: spikes_test, y: labels_test, keep_proba: 1.})
			efficiencies.append(final_eval)
			print('\nglobal efficiency : ', efficiencies[-1])
			print('confusion : ')
			print(confusion)
			print()
			
			saver.save(sess, results_dir + 'network_t' + str(tetrode+1))
		

	return efficiencies


















def neural_decode(spikes_info, xml_path, project_path, results_dir, list_channels, samplingRate, **keyword_parameters):

	print('\n\nINFERRING CLUSTERS')
	speed_cut            = keyword_parameters.get('speed_cut',    0)
	start_time           = keyword_parameters.get('start_time',   0)
	stop_time            = keyword_parameters.get('stop_time',    1e10)
	cluster_modifier     = keyword_parameters.get('cluster_modifier', 1)

	clu_path = xml_path[:len(xml_path)-3]

	allSpikes = spikes_info['spikes']
	allSpikeTime = spikes_info['spike_time']
	guessed_labels = []
	guessed_labels_time = []

	if speed_cut!=0:
		f = tables.open_file(project_path + 'nnBehavior.mat')
		positions = f.root.behavior.positions
		speed = f.root.behavior.speed
		position_time = f.root.behavior.position_time
		positions = np.swapaxes(positions[:,:],1,0)
		speed = np.swapaxes(speed[:,:],1,0)
		position_time = np.swapaxes(position_time[:,:],1,0)

	n_tetrodes = len(list_channels)
	for tetrode in range(n_tetrodes):

		### Extract
		i = 0
		if allSpikes==[]:
			if os.path.isfile(clu_path + 'clu.' + str(tetrode+1)):
				with open(
							clu_path + 'clu.' + str(tetrode+1), 'r') as fClu, open(
							clu_path + 'res.' + str(tetrode+1), 'r') as fRes, open(
							clu_path + 'spk.' + str(tetrode+1), 'rb') as fSpk:
					clu_str = fClu.readlines()
					res_str = fRes.readlines()
					n_clu = int(clu_str[0])-1
					n_channels = len(list_channels[tetrode])
					spikeReader = struct.iter_unpack(str(32*n_channels)+'h', fSpk.read())

					# spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
					labels = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)])
					spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)])
					n=0
					spikes = []
					for it in spikeReader:
						# if int(clu_str[n+1])!=0:
						# 	spike = np.reshape(np.array(it), [32,n_channels])
						# 	spikes.append(np.transpose(spike))
						spike = np.reshape(np.array(it), [32,n_channels])
						spikes.append(np.transpose(spike))
						n = n+1
					spikes = np.array(spikes, dtype=float) * 0.195 # microvolts
				
				spike_speed = np.array([speed[np.min((np.argmin(np.abs(spike_time[n]-position_time)), len(speed)-1)),:] for n in range(len(spike_time))])
			else:
				i = i+1
				continue
		else:
			spikes = allSpikes[tetrode-i]
			spike_time = allSpikeTime[tetrode-i]
		sys.stdout.write('Inferring clusters for group ' + str(tetrode+1) + ' (' + str(n_tetrodes) + ' total).')
		sys.stdout.write('\r')
		sys.stdout.flush()



		### Format
		if allSpikes!=[]:
			spike_speed = np.ones((len(spike_time),1))
			guessed_labels_time.append(spike_time)
		if speed_cut!=0:
			spikes = spikes[reduce(np.intersect1d, 
				(np.where(spike_time[:,0] > start_time), 
				np.where(spike_time[:,0] < stop_time), 
				np.where(spike_speed[:,0] > speed_cut)))]
			# labels = labels[reduce(np.intersect1d, 
			# 	(np.where(spike_time[:,0] > start_time), 
			# 	np.where(spike_time[:,0] < stop_time), 
			# 	np.where(spike_speed[:,0] > speed_cut)))]
			guessed_labels_time.append(spike_time[reduce(np.intersect1d, 
				(np.where(spike_time[:,0] > start_time), 
				np.where(spike_time[:,0] < stop_time), 
				np.where(spike_speed[:,0] > speed_cut)))])


		### Load the graph and Evaluate
		saver =           tf.train.import_meta_graph(results_dir + 'network_t' + str(tetrode+1) + '.meta')
		x =               tf.get_default_graph().get_tensor_by_name("x"+str(tetrode+1)+":0")
		y =               tf.get_default_graph().get_tensor_by_name("y"+str(tetrode+1)+":0")
		guesses =         tf.get_default_graph().get_tensor_by_name("guesses"+str(tetrode+1)+":0")
		# efficiency =      tf.get_default_graph().get_tensor_by_name("accuracy"+str(tetrode+1)+":0")
		keep_proba =      tf.get_default_graph().get_tensor_by_name("keep_proba"+str(tetrode+1)+":0")
		probas =          tf.get_default_graph().get_tensor_by_name("probas"+str(tetrode+1)+":0")
		
		with tf.Session() as sess:
			saver.restore(sess, results_dir + 'network_t' + str(tetrode+1))
			guessed_labels.append(probas.eval({x: spikes, keep_proba: 1.}))
		# guessed_labels.append(labels)
	print('Inferring clusters for ' + str(n_tetrodes) + ' groups finished.                      ')
	return {'clusters':guessed_labels, 'time':guessed_labels_time}













if __name__ == '__main__':

	dat_path ='/home/mobshamilton/Documents/dataset/Mouse-797/intanDecodingSession_181109_190106/'
	thresholds = [[273.5948839191798, 270.2454235467944, 212.94003900884405, 251.1280718964079], [251.750264536205, 248.98169063186415, 240.58748499183275, 261.44573846062906], [275.7657479892048, 263.34626585608737, 268.35621217049925, 263.7546160344242], [269.46935679346393, 265.52360601147603, 264.2993641746581, 267.81103231273744], [402.21431812314756, 408.0079565910697, 406.2138273107764, 272.1462047086307], [513.2257303279351, 520.1618066317986, 519.5801606309417, 515.3290398495378]];
	nChannels = 36
	list_channels = [[20, 21, 22, 23], [12, 13, 14, 15], [0, 1, 2, 3], [28, 29, 30, 31], [24, 25, 26, 28], [16, 17, 18, 19]]
	samplingRate = 20000


    