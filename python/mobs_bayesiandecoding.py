import os
import sys
import datetime
import math
import tables
import struct
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
# import mobs_nndecoding as mobsNN
from scipy import signal
from functools import reduce
from sklearn.neighbors import KernelDensity


def exp256(x):

	"""This function is supposedly 330 times faster than a classic exponential.
	This is an approximation, theoretically very good if x<5.
	It was also tested to be working normally here."""

	temp = 1 + x/1024
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	temp = temp * temp
	return temp
	# return np.exp(x)



def epanech_kernel_1d(size_kernel):
	values = np.ones(2*size_kernel)
	for n in range(-size_kernel, size_kernel):
		values[n+size_kernel] = 3/4/size_kernel*(1-(pow(n+1,3) - pow(n,3))/3/size_kernel**2)
	return values


def epanech_kernel_2d(size_kernel):
	kernel_1d = epanech_kernel_1d(size_kernel)
	return np.outer(kernel_1d, kernel_1d)


def kde2D(x, y, bandwidth, xbins=45j, ybins=45j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'epanechnikov')
	if ('edges' in kwargs):
		xx = kwargs['edges'][0]
		yy = kwargs['edges'][1]
	else:
		# create grid of sample locations (default: 150x150)
		xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]


	xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
	xy_train  = np.vstack([y, x]).T

	kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth)
	kde_skl.fit(xy_train)

	# score_samples() returns the log-likelihood of the samples
	z = np.exp(kde_skl.score_samples(xy_sample))
	zz = np.reshape(z, xx.shape)
	return xx, yy, zz/np.sum(zz)


def be_sure_about(bin_probas):
	"""Turns an array of probabilities into an array of zeros and ones"""

	truth = np.zeros(np.shape(bin_probas))
	for event in range(np.shape(bin_probas)[0]):
		truth[event, np.argmax(bin_probas[event,:])] = 1
	return truth












def build_kernel_densities(modules, project_path, xml_path, list_channels, samplingRate, **keyword_parameters):

	print('\nBUILDING RATE FUNCTIONS')
	speed_cut            = keyword_parameters.get('speed_cut',         0)
	start_time           = keyword_parameters.get('start_time',        0)
	stop_time            = keyword_parameters.get('stop_time',         None)
	end_time             = keyword_parameters.get('end_time',          None)
	bandwidth            = keyword_parameters.get('bandwidth',         None)
	kernel               = keyword_parameters.get('kernel',            'epanechnikov')
	masking_factor       = keyword_parameters.get('masking_factor',    20)
	cluster_modifier     = keyword_parameters.get('cluster_modifier',  1)
	rand_param           = keyword_parameters.get('rand_param',        0)

	clu_path = xml_path[:len(xml_path)-3]

	Rate_functions = []
	Marginal_rate_functions = []

	fake_labels = []
	fake_labels_time = []

	n_tetrodes = len(list_channels)
	# allSpikes = np.load(clu_path + 'trueSpikes.npy')
	for tetrode in range(n_tetrodes+1):

		if tetrode == 0:

			### EXTRACT GLOBAL DATA

			f = tables.open_file(project_path + 'nnBehavior.mat')
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
			learning_time = stop_time - start_time

			### GLOBAL OCCUPATION

			selected_positions = positions[reduce(np.intersect1d, 
				(np.where(speed[:,0] > speed_cut), 
				np.where(position_time[:,0] > start_time), 
				np.where(position_time[:,0] < stop_time)))]
			xEdges, yEdges, Occupation = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, kernel=kernel)
			Occupation[Occupation==0] = np.min(Occupation[Occupation!=0])  # We want to avoid having zeros

			mask = Occupation > (np.max(Occupation)/masking_factor)
			Occupation_inverse = 1/Occupation
			Occupation_inverse[Occupation_inverse==np.inf] = 0
			Occupation_inverse = np.multiply(Occupation_inverse, mask)


		else:

			### EXTRACT TETRODE DATA

			if os.path.isfile(clu_path + 'clu.' + str(tetrode)):
				with open(
							clu_path + 'clu.' + str(tetrode), 'r') as fClu, open(
							clu_path + 'res.' + str(tetrode), 'r') as fRes, open(
							clu_path + 'spk.' + str(tetrode), 'rb') as fSpk:
					clu_str = fClu.readlines()
					res_str = fRes.readlines()
					n_clu = int(clu_str[0])-1
					n_channels = len(list_channels[tetrode-1])
					spikeReader = struct.iter_unpack(str(32*n_channels)+'h', fSpk.read())

					# labels = np.array([[1. if int(clu_str[n+1])-1==l else 0. for l in range(n_clu)] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
					# spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1) if (int(clu_str[n+1])!=0)])
					labels          = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)])
					spike_time      = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)])
					spike_positions = np.array([positions[np.argmin(np.abs(spike_time[n]-position_time)),:] for n in range(len(spike_time))])
					spike_speed     = np.array([speed[np.min((np.argmin(np.abs(spike_time[n]-position_time)), len(speed)-1)),:] for n in range(len(spike_time))])
			else:
				print("File "+ clu_path + 'clu.' + str(tetrode) +" not found.")
				continue
			sys.stdout.write('File from tetrode ' + str(tetrode) + ' has been succesfully opened. ')
			sys.stdout.write('Processing ...')
			sys.stdout.write('\r')
			sys.stdout.flush()
			labels = modules['mobsNN'].modify_labels(labels,cluster_modifier)


			fake_labels_time.append(spike_time[reduce(np.intersect1d, 
				(np.where(spike_time[:,0] > stop_time), 
				np.where(spike_time[:,0] < end_time), 
				np.where(spike_speed[:,0] > speed_cut)))])
			fake_labels.append(modules['mobsNN'].shuffle_labels(labels[reduce(np.intersect1d,
				(np.where(spike_time[:,0] > stop_time),
				np.where(spike_time[:,0] < end_time),
				np.where(spike_speed[:,0] > speed_cut)))], rand_param))

			### MARGINAL RATE FUNCTION

			selected_positions = spike_positions[reduce(np.intersect1d, 
				(np.where(spike_speed[:,0] > speed_cut), 
				np.where(spike_time[:,0] > start_time), 
				np.where(spike_time[:,0] < stop_time)))]
			xEdges, yEdges, MRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=[xEdges,yEdges], kernel=kernel)
			MRF[MRF==0] = np.min(MRF[MRF!=0])
			MRF         = MRF/np.sum(MRF)
			MRF         = np.shape(selected_positions)[0]*np.multiply(MRF, Occupation_inverse)/learning_time
			Marginal_rate_functions.append(MRF)



			### LOCAL RATE FUNCTION FOR EACH CLUSTER

			Local_rate_functions = []

			for label in range(np.shape(labels)[1]):

				selected_positions = spike_positions[reduce(np.intersect1d, 
					(np.where(spike_speed[:,0] > speed_cut), 
					np.where(labels[:,label] == 1), 
					np.where(spike_time[:,0] > start_time), 
					np.where(spike_time[:,0] < stop_time)))]
				if np.shape(selected_positions)[0]!=0:
					xEdges, yEdges, LRF = kde2D(selected_positions[:,0], selected_positions[:,1], bandwidth, edges=[xEdges,yEdges], kernel=kernel)
					LRF[LRF==0] = np.min(LRF[LRF!=0])
					LRF         = LRF/np.sum(LRF)
					LRF         = np.shape(selected_positions)[0]*np.multiply(LRF, Occupation_inverse)/learning_time
					Local_rate_functions.append(LRF)
				else:
					Local_rate_functions.append(np.ones(np.shape(Occupation)))

	
			Rate_functions.append(Local_rate_functions)

		sys.stdout.write('We have finished building rates for group ' + str(tetrode) + ', loading next                           ')
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('We have finished building rates.                                                           ')
	sys.stdout.write('\r')
	sys.stdout.flush()
	return {'Occupation':Occupation, 'Marginal rate functions':Marginal_rate_functions, 'Rate functions': Rate_functions, 'Bins':[xEdges[:,0],yEdges[0,:]], 
				'fake_labels_info': {'clusters':fake_labels, 'time':fake_labels_time}}











def bayesian_decoding(project_path, bin_time, guessed_clusters_info, bayes_matrices, **keyword_parameters):
	
	print('\nBUILDING POSITION PROBAS')
	speed_cut            = keyword_parameters.get('speed_cut',         0)
	start_time           = keyword_parameters.get('start_time',        0)
	stop_time            = keyword_parameters.get('stop_time',         1e10)
	masking_factor       = keyword_parameters.get('masking_factor',    20)


	### Ground truth about position
	f = tables.open_file(project_path + 'nnBehavior.mat')
	positions = f.root.behavior.positions
	speed = f.root.behavior.speed
	position_time = f.root.behavior.position_time
	positions = np.swapaxes(positions[:,:],1,0)
	speed = np.swapaxes(speed[:,:],1,0)
	position_time = np.swapaxes(position_time[:,:],1,0)



	### Format matrices
	space_bins = [np.array((bayes_matrices['Bins'][0][b],bayes_matrices['Bins'][1][b])) for b in range(np.shape(bayes_matrices['Bins'][0])[0])]
	Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in ['Occupation','Marginal rate functions','Rate functions']]
	guessed_clusters, guessed_clusters_time = [guessed_clusters_info[key] for key in ['clusters','time']]
	mask = Occupation > (np.max(Occupation)/masking_factor)

	# for tetrode in range(np.shape(guessed_clusters)[0]):
	# 	for spike in range(np.shape(guessed_clusters[tetrode])[0]):
	# 		if guessed_clusters[tetrode][spike, 0] > 1/np.shape(guessed_clusters[tetrode])[1]:
	# 			guessed_clusters[tetrode][spike, :] = np.zeros([1,np.shape(guessed_clusters[tetrode])[1]])
	# 		else:
	# 			guessed_clusters[tetrode][spike, 0] = 0
	# 			guessed_clusters[tetrode][spike, :] = guessed_clusters[tetrode][spike, :]/np.sum(guessed_clusters[tetrode][spike, :], 0)


	n_bins = math.floor((stop_time - start_time)/bin_time)
	All_Poisson_term = [np.exp( (-bin_time)*Marginal_rate_functions[tetrode] ) for tetrode in range(len(guessed_clusters))]
	All_Poisson_term = reduce(np.multiply, All_Poisson_term)

	log_RF = []
	for tetrode in range(np.shape(Rate_functions)[0]):
		temp = []
		for cluster in range(np.shape(Rate_functions[tetrode])[0]):
			temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster]!=0])))
		log_RF.append(temp)




	### Decoding loop
	position_proba = [np.ones(np.shape(Occupation))] * n_bins
	position_true = [np.ones(2)] * n_bins
	nSpikes = []
	for bin in range(n_bins):

		bin_start_time = start_time + bin*bin_time
		bin_stop_time = bin_start_time + bin_time

		binSpikes = 0
		tetrodes_contributions = []
		tetrodes_contributions.append(All_Poisson_term)

		for tetrode in range(len(guessed_clusters)):
			# print(len(guessed_clusters[tetrode]))
			# print(len(guessed_clusters_time[tetrode]))
			# Clusters inside our window
			bin_probas = guessed_clusters[tetrode][np.intersect1d(
				np.where(guessed_clusters_time[tetrode][:,0] > bin_start_time), 
				np.where(guessed_clusters_time[tetrode][:,0] < bin_stop_time))]
			bin_clusters = np.sum(bin_probas, 0)
			binSpikes = binSpikes + np.sum(bin_clusters)


			# Terms that come from spike information (with normalization)
			if np.sum(bin_clusters) > 0.5:
				place_maps = reduce(np.multiply, 
					# [np.power( Rate_functions[tetrode][cluster] , bin_clusters[cluster] ) 
					# for cluster in range(np.shape(bin_clusters)[0])])
					[exp256( (log_RF[tetrode][cluster] * bin_clusters[cluster]) )
					for cluster in range(np.shape(bin_clusters)[0])])


				spike_pattern = math.pow(bin_time, np.sum(bin_clusters)) * place_maps
			else:
				spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

			tetrodes_contributions.append( spike_pattern )#/np.sum(spike_pattern) )

		nSpikes.append(binSpikes)

		position_proba[bin] = reduce( np.multiply, tetrodes_contributions )
		position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])

		position_true_mean = np.nanmean( positions[reduce(np.intersect1d, 
			(np.where(position_time[:,0] > bin_start_time), 
			np.where(position_time[:,0] < bin_stop_time), 
			np.where(speed[:,0] > speed_cut)))], axis=0 )
		position_true[bin] = position_true[bin-1] if np.isnan(position_true_mean).any() else position_true_mean

		if bin % 50 == 0:
			sys.stdout.write('[%-30s] : %.3f %%' % ('='*(bin*30//n_bins),bin*100/n_bins))
			sys.stdout.write('\r')
			sys.stdout.flush()
	sys.stdout.write('[%-30s] : %.3f %%' % ('='*((bin+1)*30//n_bins),(bin+1)*100/n_bins))
	sys.stdout.write('\r')
	sys.stdout.flush()

	position_true[0] = position_true[1]
	print('\nDecoding finished')
	return position_proba, position_true, nSpikes
















def simpleDecode(bin_time, guessed_clusters_info, bayes_matrices, **keyword_parameters):
	masking_factor       = keyword_parameters.get('masking_factor',    20)
	



	### Format matrices
	space_bins = [np.array((bayes_matrices['Bins'][0][b],bayes_matrices['Bins'][1][b])) for b in range(np.shape(bayes_matrices['Bins'][0])[0])]
	Occupation, Marginal_rate_functions, Rate_functions = [bayes_matrices[key] for key in ['Occupation','Marginal rate functions','Rate functions']]
	guessed_clusters, guessed_clusters_time = [guessed_clusters_info[key] for key in ['clusters','time']]
	mask = Occupation > (np.max(Occupation)/masking_factor)



	n_bins = math.floor((stop_time - start_time)/bin_time)
	All_Poisson_term = [np.exp( (-bin_time)*Marginal_rate_functions[tetrode] ) for tetrode in range(np.shape(guessed_clusters)[0])]
	All_Poisson_term = reduce(np.multiply, All_Poisson_term)

	log_RF = []
	for tetrode in range(np.shape(Rate_functions)[0]):
		temp = []
		for cluster in range(np.shape(Rate_functions[tetrode])[0]):
			temp.append(np.log(Rate_functions[tetrode][cluster] + np.min(Rate_functions[tetrode][cluster][Rate_functions[tetrode][cluster]!=0])))
		log_RF.append(temp)




	### Decoding loop
	position_proba = [np.ones(np.shape(Occupation))] * n_bins
	position_true = [np.ones(2)] * n_bins
	nSpikes = []
	for bin in range(n_bins):

		bin_start_time = start_time + bin*bin_time
		bin_stop_time = bin_start_time + bin_time

		binSpikes = 0
		tetrodes_contributions = []
		tetrodes_contributions.append(All_Poisson_term)

		for tetrode in range(np.shape(guessed_clusters)[0]):

			# Clusters inside our window
			bin_probas = guessed_clusters[tetrode][np.intersect1d(
				np.where(guessed_clusters_time[tetrode][:,0] > bin_start_time), 
				np.where(guessed_clusters_time[tetrode][:,0] < bin_stop_time))]
			bin_clusters = np.sum(bin_probas, 0)
			binSpikes = binSpikes + np.sum(bin_clusters)


			# Terms that come from spike information (with normalization)
			if np.sum(bin_clusters) > 0.5:
				place_maps = reduce(np.multiply, 
					# [np.power( Rate_functions[tetrode][cluster] , bin_clusters[cluster] ) 
					# for cluster in range(np.shape(bin_clusters)[0])])
					[exp256( (log_RF[tetrode][cluster] * bin_clusters[cluster]) )
					for cluster in range(np.shape(bin_clusters)[0])])


				spike_pattern = math.pow(bin_time, np.sum(bin_clusters)) * place_maps
			else:
				spike_pattern = np.multiply(np.ones(np.shape(Occupation)), mask)

			tetrodes_contributions.append( spike_pattern )#/np.sum(spike_pattern) )

		nSpikes.append(binSpikes)

		position_proba[bin] = reduce( np.multiply, tetrodes_contributions )
		position_proba[bin] = position_proba[bin] / np.sum(position_proba[bin])

		if bin % (n_bins//20) == 0:
			print('Looking good. Already finished '+str(bin)+' out of '+str(n_bins)+' : %.3f %%' % (bin*100/n_bins))

	return position_proba












if __name__ == '__main__':

	plt.imshow(epanech_kernel_2d(6))
	print(np.sum(epanech_kernel_2d(6)))
	plt.show()