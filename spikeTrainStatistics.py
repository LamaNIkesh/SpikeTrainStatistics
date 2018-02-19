'''
@author: Nikesh Lama

Some Spike train statistics.

'''

import numpy as np
import matplotlib.pyplot as plt
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
	BinnedSignal[np.floor(spikes/BinSize).astype('int')] = 1. # Index Signal with Stamps while calculating the bin number from the Stamp value
	
	#print("Empty signal : {}".format(BinnedSignal))

	return BinnedSignal



if __name__ == "__main__":

	#spike train file name
	filename = 'spiketrainch47.txt'

	#reading spike times from the file
	spiketrain = readSpikeTrains(filename)

	#creating binned signal with a binary values, for this purpose we are taking a snippet of spike train 
	#from 1000 ms to 2500 ms
	startWindow = 1000 # in ms
	endWindow = 2500 #in ms
	BinSize = 20 # in ms
	#Binned array with 
	BinnedSpikeTrain = createBinnedSpikeTrain(spiketrain, startWindow, endWindow, BinSize)
	
	# Now that we have created a binned spike train, lets try autocorrelation i.e. correlation among time lagged version of the same signal
	AutoCor = np.correlate(BinnedSpikeTrain, BinnedSpikeTrain, mode='full') # np.correlate is much faster than scipy.signal.correlate (that's open-source for ya)
	print ("auto corr: {}".format(AutoCor))
	plt.plot(AutoCor)
	plt.show()
