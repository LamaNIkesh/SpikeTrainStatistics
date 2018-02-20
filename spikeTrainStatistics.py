'''
@author: Nikesh Lama

Some Spike train statistics.

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MaxNLocator
import io
 
def readSpikeTrains(filename):
	""" Returns array of spike times read from a txt file

	Keyword arguements:

	filename -- location of the spike time file
	"""


	spikefile = open(filename,'r')
	#spiketrain = spikefile.read().replace('\n','').split(' ')
	spiketrain = np.loadtxt(filename, delimiter = " ", unpack = False)
	#spiketrain = np.asarray(spiketrain)

	print ("spiketrain: {}".format(spiketrain))

	return spiketrain


def createBinnedSpikeTrain(spiketrainArray, startWindow, endWindow, BinSize):
	"""	Returns binned spike train with desired bin size.

	Keyword arguments:

	spiketrainArray -- spike train with spike times
	startWindow -- start of time window
	endWindow -- end of time window 
	BinSize -- desired size of the bin
	"""
	window = [startWindow, endWindow]

	spikes = spiketrainArray[(spiketrainArray >= window[0]) & (spiketrainArray <= window[1])]
	#simple normalisation, basically subtracting 
	spikes -= window[0]

	#print (spikes)
	#no of bins
	nBins = (window[1] - window[0])/BinSize #

	#print ("Number of bins: {}".format(nBins))
	
	#print(np.shape(spikes))
	#print(np.shape(np.transpose(spikes)))

	BinnedSignal = np.zeros(int(nBins))
	BinnedSignal[np.floor(spikes/BinSize).astype('int')] = 1. # Index Signal with Stamps while calculating the bin number from the signal value
	
	#print("Empty signal : {}".format(BinnedSignal))

	return BinnedSignal


def PSTH(spiketrain, binSize):
	"""Returns PSTH for a spiketrain with desired bin size

	Keyword arguments:
	spiketrain --spike times
	binSize -- size of the bin 
	"""

	binarySpikeTrain = createBinnedSpikeTrain(spiketrain,500,3500,1)
	totalTimestamps = np.size(binarySpikeTrain)
	#print (binarySpikeTrain)
	spikeTrainSize = np.size(binarySpikeTrain)
	print("no of data points: {}".format(spikeTrainSize))
	noOfBins = int(spikeTrainSize/binSize) 
	#Here each row will be one bin
	binarySpikeTrain_reshaped = binarySpikeTrain.reshape(noOfBins, int(binSize))
	np.shape(binarySpikeTrain)
	print ("array shape: {}".format(np.shape(binarySpikeTrain)))
	#array to store no of spikes in a bin
	PSTH_array = np.zeros((noOfBins,binSize))

	#This works but PSTH plot is all fucked up 
	'''
	for i in range(noOfBins):
		PSTH_array[i] =  np.count_nonzero(binarySpikeTrain_reshaped[i,:] == 1)

	print("PSTH :{} PSTH size{}".format(PSTH_array, np.shape(PSTH_array)))

	return binarySpikeTrain, PSTH_array, noOfBins
	'''
	for i in range(noOfBins):
		for j in range(binSize):
			if j == 0: 
				PSTH_array[i][j] = np.count_nonzero(binarySpikeTrain_reshaped[i,:] == 1)
			else:
				PSTH_array[i][j] = 0

	print(PSTH_array)
	return binarySpikeTrain, PSTH_array.flatten(),binSize


def ISI():
	"""Calcutates ISI distances

	"""
	pass

def plot_ISIH():
	"""Plots ISI histogram
	"""



if __name__ == "__main__":
	
	#spike train file name
	filename1 = 'spiketrainch47.txt'
	filename2 = 'spiketrainch46.txt'
	#reading spike times from the file
	spiketrain1 = readSpikeTrains(filename1)
	spiketrain2 = readSpikeTrains(filename2)

	#creating binned signal with a binary values, for this purpose we are taking a snippet of spike train 
	#from 1000 ms to 2500 ms
	startWindow = 500 # in ms
	endWindow = 3500 #in ms
	BinSize = 1 # in ms
	#Binned array with 
	BinnedSpikeTrain1 = createBinnedSpikeTrain(spiketrain1, startWindow, endWindow, BinSize)
	BinnedSpikeTrain2 = createBinnedSpikeTrain(spiketrain2, startWindow, endWindow, BinSize)
	'''
	# Now that we have created a binned spike train, lets try autocorrelation i.e. correlation among time lagged version of the same signal
	AutoCor = np.correlate(BinnedSpikeTrain1, BinnedSpikeTrain2, mode='full') # np.correlate is much faster than scipy.signal.correlate (that's open-source for ya)
	print ("auto corr: {}".format(AutoCor))
	plt.plot(AutoCor)
	plt.show()
	'''
	print ("Binned Spike Train1: {}".format(BinnedSpikeTrain1))#
	print ("No of spikes in spike train 1: {}".format(np.count_nonzero(BinnedSpikeTrain1 == 1)))
	print ("Binned Spike Train2: {}".format(BinnedSpikeTrain2))
	print ("No of spikes in spike train 2: {}".format(np.count_nonzero(BinnedSpikeTrain2 == 1)))
	# fig, (ax1,ax2) = plt.subplots(nrows = 2, figsize = (20,5))
	# #plt.figure(figsize=(5,15))
	# x1 = np.arange(np.size(BinnedSpikeTrain1))
	# print (x1)
	# x2 = np.arange(np.size(BinnedSpikeTrain2))
	# np.savetxt('binnedspiketrain.txt', BinnedSpikeTrain1)
	# ax1.bar(x1,BinnedSpikeTrain1)
	# ax1.yaxis.set_major_locator(MaxNLocator(integer = True))
	# ax2.bar(x2,BinnedSpikeTrain2)
	# ax2.yaxis.set_major_locator(MaxNLocator(integer = True))
	# plt.show()

	#Some PSTH stuff, trying
	binSize = 40
	binarySpikeTrain1, PSTH1, noOfBins = PSTH(spiketrain1,binSize)
	binarySpikeTrain2, PSTH2, noOfBIns = PSTH(spiketrain2,binSize)
	
	print ("PSTH array {} PSTH size{}".format(PSTH1, np.size(PSTH1)))

	#print (PSTH1 + PSTH2)
	PSTH = PSTH1 + PSTH2
	
	x = np.arange(np.size(PSTH1))
	print ("x axis: {} psth size{}".format(x, PSTH1))
	plt.figure(figsize = (20,10))
	plt.bar(x,PSTH,width = binSize, color = 'k')
	plt.xlabel("Bin Counts")
	plt.ylabel("Firing rate (#Spikes/s)")
	plt.show()
	