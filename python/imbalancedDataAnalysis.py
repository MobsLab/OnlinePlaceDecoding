import os
os.system('clisp ' + 
	os.path.expanduser('~/Dropbox/Kteam/PrgMatlab/Thibault/lisp/extractConfusionMatrices.lsp') + ' ' +
	os.path.expanduser('~/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding/python/crossEntropy-confusions.txt'))
	# os.path.expanduser('~/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding/python/MSFE-confusions.txt'))
	# os.path.expanduser('~/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding/python/weightedMSE-confusions.txt'))
exec(open(os.path.expanduser('~/Dropbox/Kteam/PrgMatlab/OnlinePlaceDecoding/python/confusionsForPython.py')).read())

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
	print(Data.keys())
except:
	# Data = np.load(os.path.expanduser('~/Documents/dataset/RatCatanese/mobsEncoding_2019-05-06_15:10/_data.npy')).item()
	Data = np.load(os.path.expanduser('~/Documents/dataset/RatCatanese/mobsEncoding_2019-05-22_13:07/_data.npy')).item()
truLabels = [Data['labels_test'][group].argmax(axis=1) for group in range(Data['nGroups'])]

numLabels = [[np.sum(truLabels[group][:]==l) for l in range(Data['labels_test'][group].shape[1])] for group in range(Data['nGroups'])]

overallMeanAccuracy = 0
for group in range(Data['nGroups']):
	for label in range(Data['clustersPerGroup'][group]):
		overallMeanAccuracy += allConfusions[group][label][label]
overallMeanAccuracy /= np.sum(np.sum(numLabels))
print('overallMeanAccuracy :', overallMeanAccuracy)

groupMeanAccuracy = 0
for group in range(Data['nGroups']):
	nSpk = np.sum(numLabels[group])
	temp=0
	for label in range(Data['clustersPerGroup'][group]):
		temp += allConfusions[group][label][label]
	temp /= nSpk
	groupMeanAccuracy += temp / Data['nGroups']
print('groupMeanAccuracy :', groupMeanAccuracy)

ignoredClusters = 0
allClusterMeanAccuracy = 0
clusterAccuracies = []
for group in range(Data['nGroups']):
	temp =[]
	for label in range(Data['clustersPerGroup'][group]):
		if numLabels[group][label]==0:
			temp.append([])
			ignoredClusters += 1
			continue
		temp.append(allConfusions[group][label][label] / numLabels[group][label])
		allClusterMeanAccuracy += allConfusions[group][label][label] / numLabels[group][label]
	clusterAccuracies.append(temp.copy())
allClusterMeanAccuracy /= (Data['nClusters'] - ignoredClusters)
print('allClusterMeanAccuracy :', allClusterMeanAccuracy)

groupedClusterMeanAccuracy = 0
for group in range(Data['nGroups']):
	temp = 0
	ignoredClusters = 0
	for label in range(Data['clustersPerGroup'][group]):
		if numLabels[group][label]==0:
			ignoredClusters += 1
			continue
		temp += allConfusions[group][label][label] / numLabels[group][label]
	temp /= (Data['clustersPerGroup'][group] - ignoredClusters)
	groupedClusterMeanAccuracy += temp / Data['nGroups']
print('groupedClusterMeanAccuracy :', groupedClusterMeanAccuracy)
print()




### figure chance of being overclassified or underclassified depending on clu size
spikesOK = 0
spikesInBiggerClu = 0
spikesInSmallerClu = 0
sizeClu = []
overclassificationIndex = []

for group in range(Data['nGroups']):
	for label in range(Data['clustersPerGroup'][group]):
		sizeClu.append(numLabels[group][label])
		tempover = 0
		tempunder = 0
		for classifiedLabel in range(Data['clustersPerGroup'][group]):
			if label==classifiedLabel:
				spikesOK += allConfusions[group][label][classifiedLabel]
			elif numLabels[group][classifiedLabel]>=numLabels[group][label]:
				spikesInBiggerClu += allConfusions[group][label][classifiedLabel]
				tempover += allConfusions[group][label][classifiedLabel]
			else:
				spikesInSmallerClu += allConfusions[group][label][classifiedLabel]
				tempunder += allConfusions[group][label][classifiedLabel]
		if (tempunder+tempover)==0:
			sizeClu.pop()
		else:
			rate = np.sum(numLabels[group][:] > numLabels[group][label]) / (Data['clustersPerGroup'][group]-1)
			overclassificationIndex.append((tempover/(tempover+tempunder))/rate)

print('number of spikes correctly classified :', spikesOK)
print('number of spikes classified in bigger cluster :', spikesInBiggerClu)
print('number of spikes classified in smaller cluster :', spikesInSmallerClu)
fig = plt.figure()
plt.semilogx(sizeClu, overclassificationIndex, 'b.')
plt.axhline(1)
plt.ylabel('tendency to classify in bigger clusters')
plt.xlabel('size of cluster')
fig.suptitle('tendency to classify in bigger clusters against size of cluster (nClu = '+str(len(sizeClu))+')\nindex bigger than 1 means spikes are classified in bigger cluster more than chance')
fig.savefig(os.path.expanduser('~/Documents/dataset/RatCatanese/overClassifyingTendency.png'))
print('figure saved')
# plt.show()

# ### place maps analysis
stdv = []
# allStdv = []
# allPlaceMaps = []
for group in range(Data['nGroups']):
	grpStd = []
	for label in range(Data['clustersPerGroup'][group]):
		temp = Data['Rate_functions'][group][label] / Data['Rate_functions'][group][label].sum()
# 		allPlaceMaps.append( Data['Rate_functions'][group][label] / Data['Rate_functions'][group][label].sum() )
		xStdv = temp.sum(axis=1).std()
		yStdv = temp.sum(axis=0).std()
		grpStd.append(np.sqrt(np.power(xStdv,2) + np.power(yStdv,2)))
# 		allStdv.append(np.sqrt(np.power(xStdv,2) + np.power(yStdv,2)))
	stdv.append(grpStd)
# plt.figure(figsize=(15,9))
# for i in range(24):
# 	ax = plt.subplot2grid((4,6),(i//6,i%6))
# 	temp = np.argsort(allStdv)
# 	ax.imshow(allPlaceMaps[temp[i*len(temp)//24]])

### figure numLabel against clusterAccuracies
x = []
y = []
z = []
for group in range(Data['nGroups']):
	for label in range(Data['clustersPerGroup'][group]):
		if numLabels[group][label]!=0:
			x.append(clusterAccuracies[group][label])
			y.append(numLabels[group][label])
			z.append(stdv[group][label])
z = np.array(z)
x = np.array(x)
y = np.array(y)
fig = plt.figure(figsize=(15,9))

def cluAcc(ax, x,y,z):
	# ax = plt.gca()
	plt.scatter(x, y ,c=z, s=20, cmap=mpl.cm.jet)
	ax.set_yscale('log')
	plt.ylabel('number of spikes in cluster')
	plt.xlabel('accuracy of clusters')
	fig.suptitle('size of a cluster against accuracy of decoding (nClu = '+str(len(x))+')\ncolor coding is stand dev of place map')

	ax = ax.twinx()
	nBins = 8
	_, edges = np.histogram(x, nBins)
	histIdx = []
	for bin in range(nBins):
	    temp=[]
	    for n in range(len(x)):
	        if x[n]<edges[bin+1] and x[n]>=edges[bin]:
	            temp.append(n)
	    histIdx.append(temp)
	ax.errorbar(
	    [(edges[n+1]+edges[n])/2 for n in range(nBins)],
	    [np.mean(z[histIdx[n]]) for n in range(nBins)],
	    yerr=np.array([np.std(z[histIdx[n]]) for n in range(nBins)])/np.array([np.sqrt(len(histIdx[n])) for n in range(nBins)]),
	    label="mean of stand dev " +r'$\pm \sigma$')
	ax.plot([(edges[n+1]+edges[n])/2 for n in range(nBins)], [np.median(z[histIdx[n]]) for n in range(nBins)], label=r'$median \pm 20 percentile$')
	ax.legend()

	cbar = plt.colorbar()
	ticks = [str(t) for t in cbar.get_ticks()]
	ticks[0] = 'strong place field'
	ticks[-1] = 'not a place cell'
	cbar.ax.set_yticklabels(ticks)
	cbar.ax.set_ylabel('stand dev of place map')
	print('figure saved')

ax = plt.subplot2grid((2,2),(0,0),colspan=2)
cluAcc(ax, x,y,z)

ax = plt.subplot2grid((2,2),(1,0))
selection = np.where(z < 0.05)
cluAcc(ax, x[selection], y[selection], z[selection])

ax = plt.subplot2grid((2,2),(1,1))
selection = np.where(z > 0.05)
cluAcc(ax, x[selection], y[selection], z[selection])
fig.savefig(os.path.expanduser('~/Documents/dataset/RatCatanese/imbalancedClustering.png'))
plt.show()

