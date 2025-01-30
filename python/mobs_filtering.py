import scipy
from scipy import signal
from scipy import io
import numpy as np
import os
import math
import struct
import bisect
import datetime

def medianFilter(signal, N):

    signalLength = len(signal)
    filteredSignal = np.zeros(signalLength)

    for spl in range(signalLength):

        if spl < N:
            window = [signal[0]] * (N-spl-1)
            window = window + list(signal[0:spl+2+N])
            window.sort()

        elif spl > signalLength - N - 2:
            window = [signal[signalLength-1]] * (N+spl-signalLength+2)
            window = list(signal[spl-N+1:signalLength]) + window
            window.sort()

        else:
            window.pop( bisect.bisect_left(window, signal[spl-N]) )
            bisect.insort(window, signal[spl+1+N])
        

        filteredSignal[spl] = window[N]

        if spl % 50000 == 0:
            print('filtering, sample '+str(spl))

    return signal - filteredSignal

def INTANfilter(signal, cutoff, samplingRate):
    signalLength = len(signal)
    filteredSignal = np.zeros(signalLength)
    a = math.exp(-2*math.pi*cutoff/samplingRate)
    b = 1-a

    filterState = 0

    for spl in range(signalLength):
        filteredSignal[spl] = signal[spl] - filterState
        filterState = a*filterState + b*signal[spl]

    return filteredSignal















def find_spikes(xml_path, windowHalfSize, nChannels, list_channels, samplingRate):

    clu_path = xml_path[:len(xml_path)-3]

    with open(clu_path + 'dat', 'rb') as fDat:
        datReader = struct.iter_unpack(str(nChannels)+'h', fDat.read())

        
        n=0
        allChannels = []
        for it in datReader:
            data = np.array(it)
            allChannels.append(np.transpose(data))
            n = n + 1
            if n % 50000 == 0:
                print('Extracting raw data, sample '+str(n))
        allChannels = np.array(allChannels, dtype=float) * 0.195 # microvolts
	




    MOBSspikes = []
    MOBSspike_time = []
    MOBSthresholds = []

    for tetrode in range(len(list_channels)):

        groupSize = len(list_channels[tetrode])
        channelsDatum = allChannels[:, list_channels[tetrode]]

        if os.path.isfile(clu_path + 'clu.' + str(tetrode+1)):
            with open(
                        clu_path + 'clu.' + str(tetrode+1), 'r') as fClu, open(
                        clu_path + 'res.' + str(tetrode+1), 'r') as fRes, open(
                        clu_path + 'spk.' + str(tetrode+1), 'rb') as fSpk:
                clu_str = fClu.readlines()
                res_str = fRes.readlines()
                n_clu = int(clu_str[0])-1
                spikeReader = struct.iter_unpack(str(32*groupSize)+'h', fSpk.read())

                labels = np.array([[1. if int(clu_str[n+1])==l else 0. for l in range(n_clu+1)] for n in range(len(clu_str)-1)])
                spike_time = np.array([[float(res_str[n])/samplingRate] for n in range(len(clu_str)-1)])
                n=0
                spikes = []
                for it in spikeReader:
                    spike = np.reshape(np.array(it), [32,groupSize])
                    spikes.append(np.transpose(spike))
                    n = n+1
                spikes = np.array(spikes, dtype=float) * 0.195 # microvolts
        else:
            continue









        thresholds = []
        oldThresholds = []
        channelsFilteredDatum = []
        for chnl in range(groupSize):
            # channelsFilteredDatum.append( medianFilter(channelsDatum[:,chnl], windowHalfSize) )
            channelsFilteredDatum.append( INTANfilter(channelsDatum[:,chnl], 350., samplingRate) )
            thresholds.append(            np.sqrt(np.mean(channelsFilteredDatum[chnl]**2)) )
            oldThresholds.append(0.)
        channelsFilteredDatum = np.transpose( np.array(channelsFilteredDatum) )











        print('There was '+str(len(spike_time))+' found previously')
        # The loop is going on while we search for a good threshold level
        while True:

            # First, we search for spikes with the current threshold level
            nFoundSpikes = 0
            spikeTest = []
            spikeTest.append(0)
            spl = 0
            while spl < (len(channelsFilteredDatum)):


                for chnl in range(groupSize):
                    if channelsFilteredDatum[spl, chnl]>thresholds[chnl]:
                        temp = np.argmax(channelsFilteredDatum[spl:spl+20, chnl])
                        if temp != 0 and temp != 19:
                            spl = temp + spl
                            if (spl-spikeTest[-1])>17:
                                spikeTest.append(spl)
                                spl = spl + 16
                                break

                    elif channelsFilteredDatum[spl, chnl]<(-thresholds[chnl]):
                        temp = np.argmin(channelsFilteredDatum[spl:spl+20, chnl])
                        if temp != 0 and temp != 19:
                            spl = temp + spl
                            if (spl-spikeTest[-1])>17:
                                spikeTest.append(spl)
                                spl = spl + 16
                                break

                spl = spl + 1

            nFoundSpikes = nFoundSpikes + len(spikeTest) - 1

            print('We found '+str(nFoundSpikes)+' spikes with INTAN')

            # We break out if we have as much spike as before, within .5%
            if np.abs(nFoundSpikes-len(spike_time))/len(spike_time) < 0.5:#0.005:
                break
            # Else we update the thresholds, with a kind of binary scheme
            for chnl in range(groupSize):
                temp = thresholds[chnl]
                if nFoundSpikes > len(spike_time):
                    if oldThresholds[chnl] < thresholds[chnl]:
                        thresholds[chnl] = 2*thresholds[chnl] - oldThresholds[chnl]
                    else:
                        thresholds[chnl] = thresholds[chnl] + (oldThresholds[chnl] - thresholds[chnl])/2
                else:
                    if oldThresholds[chnl] < thresholds[chnl]:
                        thresholds[chnl] = oldThresholds[chnl] + (thresholds[chnl] - oldThresholds[chnl])/2
                    else:
                        thresholds[chnl] = 2*thresholds[chnl] - oldThresholds[chnl]
                oldThresholds[chnl] = temp



        # And now for spike extraction, fiou !
        spl = 0
        tempSpikes = []
        tempSpikeTime = []
        tempSpikeTime.append([0])
        while spl < (len(channelsFilteredDatum)):


            for chnl in range(groupSize):
                if channelsFilteredDatum[spl, chnl]>thresholds[chnl]:
                    temp = np.argmax(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]) )
                            spl = spl + 16
                            break

                elif channelsFilteredDatum[spl, chnl]<(-thresholds[chnl]):
                    temp = np.argmin(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]))
                            spl = spl + 16
                            break

            spl = spl + 1

        tempSpikes = np.array(tempSpikes)
        tempSpikeTime.pop(0)
        tempSpikeTime = np.array(tempSpikeTime)
        print('We found '+str(len(tempSpikeTime))+' spikes with INTAN')

        MOBSspikes.append(tempSpikes)
        MOBSspike_time.append(tempSpikeTime)
        MOBSthresholds.append(thresholds)


    return {'spikes' : MOBSspikes, 'spike_time' : MOBSspike_time, 'thresholds' : MOBSthresholds}





def extract_spikes(dat_path, nChannels, list_channels, samplingRate, thresholds):


    with open(dat_path+'amplifier.dat', 'rb') as fDat:
        datReader = struct.iter_unpack(str(nChannels)+'h', fDat.read())

        
        n=0
        allChannels = []
        for it in datReader:
            data = np.array(it)
            allChannels.append(np.transpose(data))
            n = n + 1
            if n % 50000 == 0:
                print('Extracting raw data, sample '+str(n))
        allChannels = np.array(allChannels, dtype=float) * 0.195


    MOBSspikes = []
    MOBSspike_time = []

    for tetrode in range(len(list_channels)):

        groupSize = len(list_channels[tetrode])
        channelsDatum = allChannels[:, list_channels[tetrode]]

        channelsFilteredDatum = []
        for chnl in range(groupSize):
            # channelsFilteredDatum.append( medianFilter(channelsDatum[:,chnl], windowHalfSize) )
            channelsFilteredDatum.append( INTANfilter(channelsDatum[:,chnl], 350., samplingRate) )
        channelsFilteredDatum = np.transpose( np.array(channelsFilteredDatum) )


        spl = 0
        tempSpikes = []
        tempSpikeTime = []
        tempSpikeTime.append([0])
        while spl < (len(channelsFilteredDatum)):


            for chnl in range(groupSize):
                if channelsFilteredDatum[spl, chnl]>thresholds[tetrode][chnl]:
                    temp = np.argmax(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]) )
                            spl = spl + 16
                            break

                elif channelsFilteredDatum[spl, chnl]<(-thresholds[tetrode][chnl]):
                    temp = np.argmin(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]))
                            spl = spl + 16
                            break

            spl = spl + 1


        if (np.shape(tempSpikes[-1])!=(groupSize,32)):
            tempSpikes.pop()
            tempSpikeTime.pop()
        tempSpikes = np.array(tempSpikes)
        tempSpikeTime.pop(0)
        tempSpikeTime = np.array(tempSpikeTime)
        print('We found '+str(len(tempSpikeTime))+' spikes with INTAN')

        MOBSspikes.append(tempSpikes)
        MOBSspike_time.append(tempSpikeTime)

    return {'spikes' : MOBSspikes, 'spike_time' : MOBSspike_time, 'thresholds' : thresholds}















if __name__=='__main__':

    dat_path ='/home/mobshamilton/Documents/dataset/Mouse-797/intanDecodingSession_181109_190106/'
    thresholds = [[273.5948839191798, 270.2454235467944, 212.94003900884405, 251.1280718964079], [251.750264536205, 248.98169063186415, 240.58748499183275, 261.44573846062906], [275.7657479892048, 263.34626585608737, 268.35621217049925, 263.7546160344242], [269.46935679346393, 265.52360601147603, 264.2993641746581, 267.81103231273744], [402.21431812314756, 408.0079565910697, 406.2138273107764, 272.1462047086307], [513.2257303279351, 520.1618066317986, 519.5801606309417, 515.3290398495378]];
    nChannels = 36
    list_channels = [[20, 21, 22, 23], [12, 13, 14, 15], [0, 1, 2, 3], [28, 29, 30, 31], [24, 25, 26, 28], [16, 17, 18, 19]]
    samplingRate = 20000


    with open(dat_path+'amplifier.dat', 'rb') as fDat:
        datReader = struct.iter_unpack(str(nChannels)+'h', fDat.read())

        
        n=0
        allChannels = []
        for it in datReader:
            data = np.array(it)
            allChannels.append(np.transpose(data))
            n = n + 1
            if n % 50000 == 0:
                print('Extracting raw data, sample '+str(n))
        allChannels = np.array(allChannels, dtype=float) * 0.195


    MOBSspikes = []
    MOBSspike_time = []

    for tetrode in range(len(list_channels)):

        groupSize = len(list_channels[tetrode])
        channelsDatum = allChannels[:, list_channels[tetrode]]

        channelsFilteredDatum = []
        for chnl in range(groupSize):
            # channelsFilteredDatum.append( medianFilter(channelsDatum[:,chnl], windowHalfSize) )
            channelsFilteredDatum.append( INTANfilter(channelsDatum[:,chnl], 350., samplingRate) )
        channelsFilteredDatum = np.transpose( np.array(channelsFilteredDatum) )


        spl = 0
        tempSpikes = []
        tempSpikeTime = []
        tempSpikeTime.append([0])
        while spl < (len(channelsFilteredDatum)):


            for chnl in range(groupSize):
                if channelsFilteredDatum[spl, chnl]>thresholds[tetrode][chnl]:
                    temp = np.argmax(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]) )
                            spl = spl + 16
                            break

                elif channelsFilteredDatum[spl, chnl]<(-thresholds[tetrode][chnl]):
                    temp = np.argmin(channelsFilteredDatum[spl:spl+20, chnl])
                    if temp != 0 and temp != 19:
                        spl = temp + spl
                        if (spl-tempSpikeTime[-1][0]*samplingRate)>17:
                            tempSpikeTime.append([spl/samplingRate])
                            tempSpikes.append( np.transpose(channelsFilteredDatum[spl-15:spl+17, :]))
                            spl = spl + 16
                            break

            spl = spl + 1
        if np.shape(tempSpikes[-1])!=(4,32):
            tempSpikes.pop()
            tempSpikeTime.pop()
        tempSpikes = np.array(tempSpikes)
        tempSpikeTime.pop(0)
        tempSpikeTime = np.array(tempSpikeTime)
        print('We found '+str(len(tempSpikeTime))+' spikes with INTAN')

        MOBSspikes.append(tempSpikes)
        MOBSspike_time.append(tempSpikeTime)
