
import numpy as np
import tensorflow as tf
import os
import sys
import csv
import json
import math
import struct
import matplotlib
import matplotlib.pyplot as plt

def plotOnlineDecoding(path, ax1, ax2):
	idx = []
	x = []
	y = []
	with open(path + '/decoder.txt') as csvFile:
		csvReader = csv.reader(csvFile, delimiter=';')
		for row in csvReader:
			x .append(float(row[4]) if row[4] != '-1' else None)
			y .append(float(row[5]) if row[5] != '-1' else None)

	ax1.plot(x, label='x, online decoding')
	ax2.plot(y, label='y, online decoding')





class HighPassFilter:
	def __init__(self, processor, blockSize, sampleFreq, cutOffFreq = 350.0):
		self.sampleFreq = sampleFreq
		self.cutOffFreq = cutOffFreq
		self.a = math.exp(-1.0 * 2 * math.pi * cutOffFreq / sampleFreq)
		self.b = 1.0 - self.a
		self.blockSize = blockSize
		self.processor = processor
		self.filterState = np.zeros(self.processor.nChannels)

	def __iter__(self):
		self.spl = 0
		return self

	def __next__(self):
		if self.spl < self.blockSize:
			self.filteredSample = self.processor.block[self.spl, :] - self.filterState
			self.filterState = self.a * self.filterState + self.b * self.processor.block[self.spl, :]
			self.spl += 1
			return self.filteredSample
		else:
			raise StopIteration









class SpikeDetector:
	def __init__(self, processor, path):
		self.processor = processor
		self.path = path
		self.groups = []
		self.thresholds = []
		with open(path + '/thresholds_portA.txt') as csvFile:
			csvReader = csv.reader(csvFile, delimiter=';')
			group = -1
			channels = []
			thresh = []
			for row in csvReader:
				if int(row[0]) == group:
					channels.append(int(row[1]))
					thresh.append(float(row[2]))
				else:
					if group != -1:
						self.groups.append(channels)
						self.thresholds.append(thresh)
					group = int(row[0])
					channels = [int(row[1])]
					thresh = [float(row[2])]
			self.groups.append(channels)
			self.thresholds.append(thresh)

	def __str__(self):
		temp = 'Thresholds loaded from ' + self.path + ' : \n'
		for group in range(len(self.groups)):
			for channel in range(len(self.groups[group])):
				temp += 'group ' + str(group)
				temp += ', channel ' + str(self.groups[group][channel])
				temp += ', threshold : ' + str(self.thresholds[group][channel]) + '\n'
		return temp


	def triggered(self, group, index, channel=None):
		if channel == None:
			# If no channel is specified, we run through all of the current group
			for channel in range(len(self.groups[group])):
				self.currentChannel = channel
				if self.triggered(group, index, channel):
					return True
			return False
		elif channel >= len(self.groups[group]):
			raise IndexError('Demanding channel ' + str(channel) + ' of a group of size ' + str(len(self.groups[group])))
		elif index >= self.processor.blockSize:
			raise IndexError('Fetching index ' + str(index) + ' in block of size ' + str(self.processor.blockSize))


		if self.thresholds[group][channel] > 0:
			if self.processor.filteredBlock[index - 1, self.groups[group][channel]] < self.thresholds[group][channel] and \
				self.processor.filteredBlock[index, self.groups[group][channel]] >= self.thresholds[group][channel]:
				return True
		else:
			if self.processor.filteredBlock[index - 1, self.groups[group][channel]] > self.thresholds[group][channel] and \
				self.processor.filteredBlock[index, self.groups[group][channel]] <= self.thresholds[group][channel]:
				return True
		return False




	def __call__(self, group):
		self.currentGroup = group
		return iter(self)
	def __iter__(self):
		self.currentIndex = 0
		return self
	def __next__(self):

		while not self.triggered(self.currentGroup, self.currentIndex):
			if self.currentIndex >= self.processor.blockSize - 17:
				raise StopIteration
			self.currentIndex += 1

		# find peak or trough
		if self.thresholds[self.currentGroup][self.currentChannel] > 0:
			self.currentIndex += np.argmax(self.processor.filteredBlock[self.currentIndex:self.currentIndex + 16, \
																		self.groups[self.currentGroup][self.currentChannel]])
		else:
			self.currentIndex += np.argmin(self.processor.filteredBlock[self.currentIndex:self.currentIndex + 16, \
																		self.groups[self.currentGroup][self.currentChannel]])
		
		self.currentIndex += 17

		# grab spike waveform
		if self.currentIndex >= 32:
			spike = self.processor.filteredBlock[self.currentIndex-32:self.currentIndex, self.groups[self.currentGroup]]
		else:
			spike = np.concatenate((
					self.processor.prevFilteredBlock[self.processor.blockSize-(32-self.currentIndex):, self.groups[self.currentGroup]],
					self.processor.filteredBlock[:self.currentIndex, self.groups[self.currentGroup]]), axis=0)
		return spike













class NeuralDecoder:
	def __init__(self, processor, paths, binTime):
		self.datFolder       = paths[0]
		self.jsonPath        = paths[1]
		self.binTime         = binTime
		self.processor       = processor
		with open(self.jsonPath) as f:
			self.graphMetaData = json.load(f)
		
		self.encoderPath     = self.graphMetaData["encodingPrefix"]
		self.session         = None
		self.saver           = tf.train.import_meta_graph(self.encoderPath + '.meta')
		self.feedDictData    = []
		self.feedDictTensors = []

		for grp in range(self.graphMetaData["nGroups"]):
			self.feedDictData.append( np.empty([0, self.graphMetaData["group"+str(grp)]["nChannels"], 32]) )
			self.feedDictData.append( 1.0 )
			self.feedDictTensors.append( tf.get_default_graph().get_tensor_by_name("group"+str(grp)+"/spikeEncoder/x:0") )
			self.feedDictTensors.append( tf.get_default_graph().get_tensor_by_name("group"+str(grp)+"/spikeEncoder/keep_proba:0") )
		self.feedDictData   .append( [binTime] )
		self.feedDictTensors.append( tf.get_default_graph().get_tensor_by_name("bayesianDecoder/binTime:0") )
		self.positionGuessed = tf.get_default_graph().get_tensor_by_name("bayesianDecoder/positionGuessed:0")

		self.decodedGroups = []
		for grp in self.processor.spikeDetector.groups:
			self.decodedGroups.append(NeuralDecoder.findGroup(grp, self.graphMetaData))

	def __enter__(self):
		self.session = tf.Session()
		self.saver.restore(self.session, self.encoderPath)
		print('Tensorflow graph is opened and loaded.')

	def __exit__(self, exc_type, exc_value, traceback):
		self.session.close()
		self.session = None

	def findGroup(grp, data):
		for nnGroup in range(data["nGroups"]):
			if NeuralDecoder.isSameGroup(grp, data["group"+str(nnGroup)]):
				return nnGroup
		raise IndexError('This group doesn\'t fit any in the tensorflow graph : '+str(grp))
	def isSameGroup(grp, data):
		if len(grp) != data["nChannels"]:
			return False
		for chnl in range(len(grp)):
			if grp[chnl] != data["channel"+str(chnl)]:
				return False
		return True

	def addSpike(self, spike, group):
		self.feedDictData[2*self.decodedGroups[group]] = np.concatenate((self.feedDictData[2*self.decodedGroups[group]], [np.transpose(spike)]))

	def decodedPosition(self):
		if self.session == None or self.session._closed:
			raise AttributeError('Asking for decoding when session has not been opened')

		result = self.positionGuessed.eval({i:j for i,j in zip(self.feedDictTensors, self.feedDictData)},
											session = self.session)

		for group in range(len(self.processor.spikeDetector.groups)):
			self.feedDictData[2*self.decodedGroups[group]] = np.delete(self.feedDictData[2*self.decodedGroups[group]],
																	range(len(self.feedDictData[2*self.decodedGroups[group]])), 0)

		return result











class SignalProcessor:
	def __init__(self, paths, nChannels, blockSize, sampleFreq, cutOffFreq = 350.0):
		self.path = paths[0]
		self.blockSize = blockSize
		self.nChannels = nChannels
		self.filteredBlock = np.zeros([blockSize, nChannels])
		self.prevFilteredBlock = np.zeros([blockSize, nChannels])
		self.filter = HighPassFilter(self, blockSize, sampleFreq, cutOffFreq)
		self.spikeDetector = SpikeDetector(self, paths[0])
		self.neuralDecoder = NeuralDecoder(self, paths, blockSize / sampleFreq)
		self.decodedPositions = []

	def __repr__(self):
		return 'SignalProcessor({!r}, {!r}, {!r}, {!r}, {!r})'.format(
			self.path, self.nChannels, self.blockSize, self.filter.sampleFreq, self.filter.cutOffFreq)

	def __enter__(self):
		self.nWindows = os.path.getsize(self.path + '/amplifier.dat')/2/self.nChannels/self.blockSize
		if self.nWindows != int(self.nWindows):
			raise ValueError('File length does not match parameters.')
		else:
			self.nWindows = int(self.nWindows)
		self.datFile = open(self.path + '/amplifier.dat', 'rb')
		self.blockReader = struct.iter_unpack(str(self.nChannels*self.blockSize)+'h', self.datFile.read())
		print('Amplifier file is opened successfully. Expected number of windows : ', self.nWindows)
		self.neuralDecoder.__enter__()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.neuralDecoder.__exit__(exc_type, exc_value, traceback)
		self.datFile.close()
		del self.blockReader

	def nextBlock(self):
		try:
			self.block = np.resize(next(self.blockReader), [self.blockSize, self.nChannels]) * 0.195
			return True
		except StopIteration:
			return False

	def filterBlock(self):
		spl = 0
		self.prevFilteredBlock = self.filteredBlock.copy()
		for filteredSample in self.filter:
			self.filteredBlock[spl, :] = filteredSample
			spl += 1

	def findSpikes(self):
		for group in range(len(self.spikeDetector.groups)):
			for spike in self.spikeDetector(group):
				self.neuralDecoder.addSpike(spike, group)

	def decodePosition(self):
		self.decodedPositions.append( self.neuralDecoder.decodedPosition() )

	def plotPosition(self, ax1, ax2):
		positions = np.array(self.decodedPositions)
		ax1.plot(positions[:,0], label='x, offline decoding')
		ax2.plot(positions[:,1], label='y, offline decoding')
		self.decodedPositions = []





samplingRate = 20000
blockSize = 720
nChannels = 32

metaDataPath = '/home/mobshamilton/Documents/dataset/Mouse-797/ERC-Mouse-797-09112018-Hab_SpikeRef.json'
recordingFolder = '/home/mobshamilton/Documents/dataset/testingOutput/REAL_TEST_190121_115311'

fig, ax = plt.subplots(figsize=(20,20))
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0))


window = 0
decodedPositions = []

with SignalProcessor([recordingFolder, metaDataPath], nChannels, blockSize, samplingRate) as signalProcessor:

	while signalProcessor.nextBlock():
		signalProcessor.filterBlock()
		signalProcessor.findSpikes()
		signalProcessor.decodePosition()


		window += 1
		sys.stdout.write('[%-30s] step : %d/%d' % ('='*((window*30)//signalProcessor.nWindows),window,signalProcessor.nWindows))
		sys.stdout.write('\r')
		sys.stdout.flush()

	signalProcessor.plotPosition(ax1, ax2)

print()

plotOnlineDecoding(recordingFolder, ax1, ax2)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()