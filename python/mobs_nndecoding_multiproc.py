import numpy as np
import tensorflow as tf
import multiprocessing
import sys
import struct
import tables
from sklearn.neighbors import KernelDensity

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

def kde2D(x, y, bandwidth, xbins=45j, ybins=45j, **kwargs):
	"""Build 2D kernel density estimate (KDE)."""

	kernel       = kwargs.get('kernel',       'gaussian')
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








class MultiGroupReader:
	class InvalidSpeed(Exception):
		pass

	def __init__(self, path, samplingRate, speedCut, bandwidth, maskingFactor):
		getDir             = lambda fullPath : fullPath[:-1] if fullPath[-2]=='/' else getDir(fullPath[:-1])
		self.dir           = getDir(path)
		self.path          = path
		self.samplingRate  = samplingRate
		self.speedCut      = speedCut
		self.bandwidth     = bandwidth
		self.maskingFactor = maskingFactor

		with tables.open_file(self.dir + 'nnBehavior.mat') as f:
			self.positions = f.root.behavior.positions
			self.speed     = f.root.behavior.speed
			self.posTime   = f.root.behavior.position_time
			self.positions = np.swapaxes(self.positions[:,:], 1,0)
			self.speed     = np.swapaxes(self.speed[:,:], 1,0)
			self.posTime   = np.swapaxes(self.posTime[:,:], 1,0)

		if np.shape(self.speed)[0] != np.shape(self.positions)[0]:
			if np.shape(self.speed)[0] == np.shape(self.positions)[0] - 1:
				speed.append(self.speed[-1])
			elif np.shape(self.speed)[0] == np.shape(self.positions)[0] + 1:
				self.speed = self.speed[:-1]
			else:
				raise self.Break

	def setGroups(self, listChannels):
		self.listChannels = listChannels
	def getGroups(self):
		return self.listChannels

	def __enter__(self):
		self.dataReaders = []
		for group in range(len(self.listChannels)):
			self.dataReaders.append( DataReader(self)(group, len(self.listChannels[group])) )
			try:
				self.dataReaders[-1].__enter__()
			except FileNotFoundError:
				self.dataReaders.pop()

		for group in range(len(self.dataReaders)):
			self.dataReaders[group].setGroup(group)

	def __exit__(self, exc_type, exc_value, traceback):
		exitTrace = True
		for group in range(len(self.dataReaders)):
			exitTrace *= self.dataReaders[group].__exit__(exc_type, exc_value, traceback)
		return exitTrace

	def moveTo(self, timeStamp):

		self.index = 0
		while self.posTime[self.index] < timeStamp:
			self.index += 1

		for group in range(len(self.dataReaders)):
			try:
				self.dataReaders[group].moveTo(timeStamp)
			except:
				if self.dataReaders[group].__exit__(*sys.exc_info()):
					self.dataReaders.pop(group)
				else:
					raise

	def getLearningData(self, startTime, endTime, neuralNet):

		sys.stdout.write('Processing position data.\n')
		sys.stdout.flush()

		self.moveTo(startTime)

		allPositionSelection = []
		self.learningTime = endTime - self.posTime[self.index]
		index = self.index
		while self.posTime[index] < endTime:
			if self.speed[index] > self.speedCut:
				allPositionSelection.append(index)
			index += 1

		xEdges, yEdges, self.Occupation = kde2D(self.positions[allPositionSelection,0], 
												self.positions[allPositionSelection,1], self.bandwidth)
		self.Occupation[self.Occupation==0] = np.min(self.Occupation[self.Occupation!=0])  # We want to avoid having zeros
		self.xBins = xEdges[:,0]
		self.yBins = yEdges[0,:]

		self.mask = self.Occupation > (np.max(self.Occupation)/self.maskingFactor)
		self.Occupation_inverse = 1/self.Occupation
		self.Occupation_inverse[self.Occupation_inverse==np.inf] = 0
		self.Occupation_inverse = np.multiply(self.Occupation_inverse, self.mask)


		for group in range(len(self.dataReaders)):
			try:
				sys.stdout.write('Processing data from group '+str(group+1)+'.\n')
				sys.stdout.flush()
				self.dataReaders[group].getLearningData(endTime, neuralNet, [xEdges, yEdges])
				sys.stdout.write('Processing data from group '+str(group+1)+' is done.\n')
				sys.stdout.flush()
			except:
				print('a problem occured during learning data from group '+str(group+1)+'.\n')
				if self.dataReaders[group].__exit__(*sys.exc_info()):
					self.dataReaders.pop(group)
				else:
					raise

		self.index = index
		self.timeCursor = endTime
		return self.Occupation, [self.xBins, self.yBins]

	def getSpkPosIdx(self, index, spikeTimeStamp):
		if index == len(self.posTime):
			return index

		if abs(self.posTime[index] - spikeTimeStamp) < abs(self.posTime[index+1] - spikeTimeStamp):
			if self.speed[index] < self.speedCut:
				raise self.InvalidSpeed
			return index
		else:
			return self.getSpkPosIdx(index + 1, spikeTimeStamp)

	def setEndTime(self, endTime):
		for reader in self.dataReaders:
			reader.setEndTime(endTime)
	def feedBinData(self, binTime, neuralNet):
		self.timeCursor += binTime
		i = self.index
		while self.posTime[i] < self.timeCursor:
			i += 1

		self.currentXPos = np.mean(self.positions[self.index:i+1, 0])
		self.currentYPos = np.mean(self.positions[self.index:i+1, 1])
		meanSpeed        = np.mean(self.speed[self.index:i+1])
		self.index       = i

		state = False
		for reader in self.dataReaders:
			state |= reader.getBinData(binTime, neuralNet)

		# if meanSpeed < self.speedCut:
		# 	neuralNet.clear()
		# 	return self.feedBinData(binTime, neuralNet)

		return state






class DataReader:
	class Break(Exception):
		"""Break out in case of trouble"""


	def __init__(self, generalReader):
		self.opened         = False
		self.generalReader  = generalReader
		self.timeCursor     = 0

	def setGroup(self, group):
		self.group = group

	''' Three functions have the role to open files and close them if they exist, 
	and handle exceptions if they don't.'''
	def __call__(self, group, nChannels):
		self.group     = group
		self.nChannels = nChannels
		self.opened    = False
		self.cluPath   = self.generalReader.path[:-4] + '.clu.' + str(group+1)
		self.resPath   = self.generalReader.path[:-4] + '.res.' + str(group+1)
		self.spkPath   = self.generalReader.path[:-4] + '.spk.' + str(group+1)
		return self

	def __enter__(self):
		sys.stdout.write('Entering data of group '+str(self.group+1) +'\n')
		sys.stdout.flush()
		try:
			self.fClu = open(self.cluPath, 'r')
			self.fRes = open(self.resPath, 'r')
			self.fSpk = open(self.spkPath, 'rb')
			self.opened = True
		except:
			self.__exit__(*sys.exc_info())
			raise
		
		self.nClusters      = int(next(self.fClu))
		self.spikeReader    = struct.iter_unpack(str(32*self.nChannels)+'h', self.fSpk.read())
		self.nextSpike()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		if exc_type==None and self.opened:
			self.fClu.close()
			self.fRes.close()
			self.fSpk.close()
			print('leaving reader of group ' + str(self.group + 1))
			return True
		elif isinstance(exc_value, FileNotFoundError):
			print('Impossible to open file '+ exc_value.filename)
			if exc_value.filename == self.cluPath:
				pass
			elif exc_value.filename == self.resPath:
				self.fClu.close()
			elif exc_value.filename == self.spkPath:
				self.fClu.close()
				self.fRes.close()
			else: # unknown case
				return False
			return True
		elif exc_type == self.Break:
			print('leaving reader of group ' + str(self.group + 1))
			return True
		else:
			return False



	def moveTo(self, timeStamp):
		''' ignore data to get to start time of learning '''
		while self.spikeTimeStamp < timeStamp:
			if self.nextSpike() == False:
				print('No data fit parameters.')
				raise self.Break
		self.timeCursor = timeStamp


	def nextSpike(self):
		try:
			self.cluster        = int(next(self.fClu))
			self.spikeTimeStamp = float(next(self.fRes))/self.generalReader.samplingRate
			self.spike          = np.transpose(np.reshape(np.array(next(self.spikeReader)), [32, self.nChannels]))
			return True
		except StopIteration:
			self.spike          = np.empty([0, self.nChannels, 32])
			return False


	def getLearningData(self, timeStamp, neuralNet, edges):
		''' start saving training data and build stat maps '''
		self.timeCursor = timeStamp
		self.trainingSpikes = []
		self.trainingLabels = []
		spikePositionSelection = []
		clusterPositionSelection = [[] for clu in range(self.nClusters)]
		
		index = self.generalReader.index
		while self.spikeTimeStamp < timeStamp:
			self.trainingSpikes.append(self.spike)
			self.trainingLabels.append(self.cluster)
			try:
				index = self.generalReader.getSpkPosIdx(index, self.spikeTimeStamp)
				spikePositionSelection.append(index)
				clusterPositionSelection[self.cluster].append(index)
			except self.generalReader.InvalidSpeed:
				pass

			if self.nextSpike() == False:
				break

		self.testingLabels = self.trainingLabels[9*len(self.trainingLabels)//10:len(self.trainingLabels)]
		self.testingSpikes = self.trainingSpikes[9*len(self.trainingSpikes)//10:len(self.trainingSpikes)]
		self.trainingLabels = self.trainingLabels[0:9*len(self.trainingLabels)//10]
		self.trainingSpikes = self.trainingSpikes[0:9*len(self.trainingSpikes)//10]

		xEdges = edges[0]
		yEdges = edges[1]

		xEdges, yEdges, MRF   = kde2D(self.generalReader.positions[spikePositionSelection,0], 
										self.generalReader.positions[spikePositionSelection,1], 
										self.generalReader.bandwidth, edges=[xEdges,yEdges])
		MRF[MRF==0]           = np.min(MRF[MRF!=0])
		MRF                   = MRF/np.sum(MRF)
		MRF                   = len(spikePositionSelection)*np.multiply(MRF, self.generalReader.Occupation_inverse)/self.generalReader.learningTime

		CRF = []
		for cluster in range(self.nClusters):
			xEdges, yEdges, LRF   = kde2D(self.generalReader.positions[clusterPositionSelection[cluster],0], 
											self.generalReader.positions[clusterPositionSelection[cluster],1], 
											self.generalReader.bandwidth, edges=[xEdges,yEdges])
			LRF[LRF==0]           = np.min(LRF[LRF!=0])
			LRF                   = LRF/np.sum(LRF)
			CRF.append(           len(clusterPositionSelection[cluster])*np.multiply(LRF, self.generalReader.Occupation_inverse)/self.generalReader.learningTime )
		neuralNet.addGraph(self, MRF, CRF)
		self.decodingState  = True

	def setEndTime(self, endTime):
		self.endTime = endTime
	def getBinData(self, binTime, neuralNet):
		self.timeCursor += binTime
		if self.timeCursor >= self.endTime or not self.decodingState:
			return False

		while self.spikeTimeStamp < self.timeCursor:
			neuralNet.addSpike( self.spike, self.group )
			if self.nextSpike() == False:
				self.decodingState = False

		return self.decodingState






class NeuralNet:
	def __init__(self, modules, path, dataReader):
		self.modules = modules
		self.path = path
		self.dataReader = dataReader
		self.nGroups = 0
		self.nClusters = []
		self.sumConstantTerms = []
		self.allRateMaps = []
		self.probasTensors = []

		self.trainSpikes = []
		self.trainLabels = []
		self.testSpikes = []
		self.testLabels = []
		self.accuracy = []
		self.x = []
		self.y = []
		self.keep_proba = []
		self.confusion_matrix = []
		self.crossTrain = []

		self.feedDictData    = []
		self.feedDictTensors = []

	def addGraph(self, groupReader, MRF, CRF):
		nChannels = groupReader.nChannels
		self.nGroups += 1
		if self.sumConstantTerms == []:
			self.currentGroup = 0
			self.sumConstantTerms = MRF
		else:
			self.currentGroup += 1
			self.sumConstantTerms = np.sum([self.sumConstantTerms, MRF], axis = 0)
		self.allRateMaps += [np.log(CRF[clu] + np.min(CRF[clu][CRF[clu]!=0])) for clu in range(len(CRF))]

		self.trainSpikes.append(groupReader.trainingSpikes)
		self.trainLabels.append(groupReader.trainingLabels)
		self.testSpikes.append(groupReader.testingSpikes)
		self.testLabels.append(groupReader.testingLabels)

		with tf.name_scope("group"+str(self.currentGroup)):
		
			with tf.name_scope("spikeEncoder"):
				self.nClusters.append(len(CRF))
				self.x.append(          tf.placeholder(tf.float32, shape=[None, nChannels, 32],      name='x') )
				yTemp                 = tf.placeholder(tf.float32, shape=[None, len(CRF)],           name='yTemp')
				self.y.append(          tf.placeholder(tf.int64,   shape=[None],                     name='y') )
				self.keep_proba.append( tf.placeholder(tf.float32,                                   name='keep_proba') )

				spikeEncoder        = self.modules['mobs_NB'].encoder(self.x[-1],yTemp,self.keep_proba[-1], size=64)



			with tf.name_scope("spikeEvaluator"):

				probas              = tf.nn.softmax(spikeEncoder['Output'], name='probas')
				self.probasTensors.append( tf.reduce_sum(probas, axis=0, name='sumProbas'))

				guesses             = tf.argmax(spikeEncoder['Output'],1, name='guesses')
				good_guesses        = tf.equal(self.y[-1], guesses)
				self.accuracy.append( tf.reduce_mean(tf.cast(good_guesses, tf.float32), name='accuracy') )
				self.confusion_matrix.append( tf.confusion_matrix(self.y[-1], guesses, name='confusion') )
				
				cross_entropy       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[-1], logits=spikeEncoder['Output']))
				self.crossTrain.append( tf.train.AdamOptimizer(0.00004).minimize(cross_entropy, name='trainer') )
		
		self.feedDictData.append( np.empty([0, nChannels, 32]) )
		self.feedDictData.append( 1.0 )
		self.feedDictTensors.append( self.x[-1] )
		self.feedDictTensors.append( self.keep_proba[-1] )


	def buildDecoder(self, binTime):
		nClusters = len(self.allRateMaps)
		with tf.name_scope("bayesianDecoder"):

			binTimeTensor               = tf.placeholder(tf.float32, shape=[1], name='binTime')
			allProbas                   = tf.reshape(tf.concat(self.probasTensors, 0), [1, nClusters], name='allProbas');

			# Place map stats
			occMask                     = tf.constant(self.dataReader.mask,              dtype=tf.float64, shape=[45,45])
			constantTerm                = tf.constant(self.sumConstantTerms,          dtype=tf.float32, shape=[45,45])
			occMask_flat                = tf.reshape(occMask, [45*45])
			constantTerm_flat           = tf.reshape(constantTerm, [45*45])

			self.allRateMaps            = np.array(self.allRateMaps)
			rateMaps                    = tf.constant(self.allRateMaps,               dtype=tf.float32, shape=[nClusters, 45,45], name='rateMaps')
			rateMaps_flat               = tf.reshape(rateMaps, [nClusters, 45*45])
			spikesWeight                = tf.matmul(allProbas, rateMaps_flat)

			allWeights                  = tf.cast( spikesWeight - binTime * constantTerm_flat, tf.float64 )
			allWeights_reduced          = allWeights - tf.reduce_mean(allWeights)

			positionProba_flat          = tf.multiply( tf.exp(allWeights_reduced), occMask_flat )
			self.positionProba          = tf.reshape(positionProba_flat / tf.reduce_sum(positionProba_flat), [45,45], name='positionProba')

			xBins                       = tf.constant(np.array(self.dataReader.xBins), shape=[45], name='xBins')
			yBins                       = tf.constant(np.array(self.dataReader.yBins), shape=[45], name='yBins')
			xProba                      = tf.reduce_sum(self.positionProba, axis=1, name='xProba')
			yProba                      = tf.reduce_sum(self.positionProba, axis=0, name='yProba')
			xGuessed                    = tf.reduce_sum(tf.multiply(xProba, xBins)) / tf.reduce_sum(xProba)
			yGuessed                    = tf.reduce_sum(tf.multiply(yProba, yBins)) / tf.reduce_sum(yProba)
			xStd                        = tf.sqrt(tf.reduce_sum(xProba*tf.square(xBins-xGuessed)))
			yStd                        = tf.sqrt(tf.reduce_sum(yProba*tf.square(yBins-yGuessed)))

			self.positionGuessed        = tf.stack([xGuessed, yGuessed], name='positionGuessed')
			self.standardDeviation      = tf.stack([xStd, yStd], name='standardDeviation')

		self.feedDictData   .append( [binTime] )
		self.feedDictTensors.append( binTimeTensor )


	def train(self, nSteps):
		self.saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

			for group in range(self.nGroups):
				print('train group '+str(group))
				for i in range(nSteps):
					batch = next_batch(80, self.trainSpikes[group], self.trainLabels[group])

					if i%50 == 0:
						curr_eval = sess.run([self.accuracy[group]], 
												{self.x[group]: batch[0], 
												self.y[group]: batch[1], 
												self.keep_proba[group]: 1.})
						sys.stdout.write('[%-30s] step : %d/%d, efficiency : %g' % ('='*(i*30//nSteps),i,nSteps,curr_eval[0]))
						sys.stdout.write('\r')
						sys.stdout.flush()

					# training step
					self.crossTrain[group].run({self.x[group]: batch[0], 
												self.y[group]: batch[1], 
												self.keep_proba[group]: 0.5})
				sys.stdout.write('[%-30s] step : %d/%d, efficiency : %g' % ('='*((i+1)*30//nSteps),i+1,nSteps,curr_eval[0]))
				sys.stdout.write('\r')
				sys.stdout.flush()
				final_eval, confusion = sess.run([self.accuracy[group], 
												  self.confusion_matrix[group]], 
												{self.x[group]: self.testSpikes[group], 
												self.y[group]: self.testLabels[group], 
												self.keep_proba[group]: 1.})
				efficiency = final_eval
				print('\nglobal efficiency : ', efficiency)
				print('confusion : ')
				print(confusion)
				sys.stdout.flush()

			self.saver.save(sess, self.path + '/mobsGraph')
			self.session = None

		






	def __enter__(self):
		self.session = tf.Session()
		self.saver.restore(self.session, self.path+'/'+'mobsGraph')
		print('Tensorflow graph is opened and loaded.')
	def __exit__(self, exc_type, exc_value, traceback):
		self.session.close()
		self.session = None

	def addSpike(self, spike, group):
		self.feedDictData[2*group] = np.concatenate((self.feedDictData[2*group], [spike]))

	def decodedPosition(self):
		if self.session == None or self.session._closed:
			raise AttributeError('Asking for decoding when session has not been opened')

		# result = self.positionGuessed.eval({i:j for i,j in zip(self.feedDictTensors, self.feedDictData)},
		# 									session = self.session)
		position, standDev, positionProba = self.session.run(
			[self.positionGuessed, self.standardDeviation, self.positionProba],
			{i:j for i,j in zip(self.feedDictTensors, self.feedDictData)})

		self.clear()

		return position, standDev, positionProba

	def clear(self):
		for group in range(self.nGroups):
			self.feedDictData[2*group] = np.delete(self.feedDictData[2*group],
													range(len(self.feedDictData[2*group])), 0)







