'''
@author: Nikesh Lama

Some Spike train statistics.

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import MaxNLocator
import io
import scipy.stats as stats
 
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


def raster(event_times_list, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    #plt.xlim([3900,4100])   #here we can limit which part of the raster plot we want to look
    #plt.ylim([45,54])
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 2.8, **kwargs)
        #print("ith: {}, trial: {}".format(ith,trial))
        #print ("trial ",trial)
    #plt.ylim(.5, len(event_times_list) + .5)
    #plt.ylim([45,54])
    return ax

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

def PSTH_multi(spiketrain, binSize, totalTime):
	""" Returns spike count in each bin for multiple spike trains

	"""
	#lets try to bin at a divisble values 
	no_of_bins = int(totalTime/binSize)
	#creating empty 1D list with the size same as bin count to store number of spikes in that bin
	PSTH_count = [0]*no_of_bins
	print("PSTH: {}, size :{}".format(PSTH_count,len(PSTH_count)))

	#lets loop through each spike train and count number of spikes
	print("numnber of spike trains: {}".format(len(spiketrain)))
	print("first element: {}".format(spiketrain[0][0]))
	for i in range(len(spiketrain)):
		#loop through each spike train for each bin and check if there is values in the range. 
		#if there is bincount is incremented by 1
		#spiketrains are in a 2D list
		startpoint = 0 # this is a temporary holder for range used in the next loop
		endpoint = binSize 
		
		for j in range(no_of_bins):
			#look into range which is the bin widht
			#for eg for 20ms bin, range is 0-20, 20-40, and so on...
			if(spiketrain[i][0] == -1.0):
				#nothing here empty spike train so we just skip this all together
				break
				#print("empty spike train")
				#PSTH_count[j]
			else:

				
				#if(np.count_nonzero((startpoint < spiketrain[i]) & (spiketrain[i] < endpoint))):
					#print("spike train #: {}".format(i))
					#print("range is : {} to {}".format(startpoint,endpoint))
					#print ("check True/Flase: {}".format(np.count_nonzero((startpoint < spiketrain[i]) & (spiketrain[i] < endpoint))))			
				#add if a spike is discovered in the range
				PSTH_count[j] = PSTH_count[j] + np.count_nonzero((startpoint < spiketrain[i][:]) & (spiketrain[i][:] < endpoint))
				#for next iteration endpoint becomes a new start point and new end point becomes endpoint + binSize
				startpoint = endpoint
				endpoint = endpoint + binSize
		#print("End of loop....")
		
	print("PSTH count: {}".format(PSTH_count))		
	return PSTH_count, no_of_bins,binSize

def PSTH_realTimeAxis(PSTH_binCount, binSize):
	"""Plots the PSTH. original PSTH list contains spike count for each time bin.
		To plot in a real time axis, the list is extended with empty elements after each intial element.this results in the original time range_index
		for eg. for binsize = 20 for 1000ms , original PSTH list has 1000/20=50 bins hence 50 elements.

		so to extend this , after each element, 19 empty elements are added. so that you get[12,0,0,....0,34,0,0.......]	
		the bar graph will have same width as binsize so that bar width cover the other binsize-1 empty elements.

		This is not the best way, will come back to this later. 


	"""

	PSTH_realTimeAxis = []
	n = 1 
	for start_index in range(0, len(PSTH_binCount), n):
		if start_index>0:
			PSTH_realTimeAxis.extend(PSTH_binCount[start_index:start_index+n])
			for i in range(binSize):
				PSTH_realTimeAxis.append(0)

	print ("New PSTH: {}".format(PSTH_realTimeAxis))
	return PSTH_realTimeAxis

	'''
	plotting bits, conducted outside the function.
	#get current axis
	ax = plt.gca()
	x = np.arange(np.size(PSTH_realTimeaxis))
	
	#plt.figure(figsize = (20,10))
	plt.bar(x,PSTH_realTimeaxis, width = binSize, color = 'k')
	return ax 
	
	plt.xlabel("Bin Counts")
	plt.ylabel("Firing rate (#Spikes/s)")
	#plt.show()
	'''


def PSTH(spiketrain, binSize):
	"""Returns PSTH for a spiketrain with desired bin size

	Keyword arguments:
	spiketrain --spike times
	binSize -- size of the bin 
	"""

	binarySpikeTrain = createBinnedSpikeTrain(spiketrain,0,5000,1)
	totalTimestamps = np.size(binarySpikeTrain)
	#print (binarySpikeTrain)
	spikeTrainSize = np.size(binarySpikeTrain)
	#print("no of data points: {}".format(spikeTrainSize))
	noOfBins = int(spikeTrainSize/binSize) 
	#Here each row will be one bin
	binarySpikeTrain_reshaped = binarySpikeTrain.reshape(noOfBins, int(binSize))
	np.shape(binarySpikeTrain)
	#print ("array shape: {}".format(np.shape(binarySpikeTrain)))
	#array to store no of spikes in a bin
	PSTH_array = np.zeros((noOfBins,binSize))
	#Not sure this is the best way but wanted to get the real time axis and not binned axis
	# so for each bin count is stored in a first element of row
	# the psth bar plot width is the binsize so the rest of the elements are not needed
	# hence, all the other elements are just stored as zero. 
	# not optimised at all. .....
	# will come back to this later. 

	###TO DO -- make it automatic to read any number of spike trains and plot corresponding PSTH. 


	for i in range(noOfBins):
		for j in range(binSize):
			if j == 0: 
				PSTH_array[i][j] = np.count_nonzero(binarySpikeTrain_reshaped[i,:] == 1)
			else:
				PSTH_array[i][j] = 0

	print(PSTH_array)
	return binarySpikeTrain, PSTH_array.flatten(),binSize


def ISI(spiketrain1 = [], spiketrain2 = [], *args):
	"""Calcutates ISI distances
		Interspike interval is a temporal distance between two consecutive spikes
		Basically, we calculate distance between n and n+1 spike. Then plot the density histogram

		Try with one spike train first.
	"""
	ISI_distances = list()
	range_index = np.size(spiketrain1) - 1
	#print("range :{}".format(np.size(spiketrain)))
	for i in range(range_index):
		#print ("index: {}, spiketrain[48] = {}".format(i,spiketrain[48]))
		ISI_distances.append(spiketrain1[i+1] - spiketrain1[i])
		#print("ISI Distance at index {} is {}".format(i,ISI_distances[i]))
	for i in range(np.size(spiketrain2) -1):
		 ISI_distances.append(spiketrain2[i+1] - spiketrain2[i])
	return ISI_distances

def ISI_multi(spiketrain = [], *args):
	"""Calcutates ISI distances for a list of spike trains
		Interspike interval is a temporal distance between two consecutive spikes
		Basically, we calculate distance between n and n+1 spike. Then plot the density histogram

		Try with one spike train first.
	"""
	ISI_distances = list()
	range_index = len(spiketrain) # no of spike trains
	#print("range :{}".format(np.size(spiketrain)))
	for i in range(range_index):
		#print ("index: {}, spiketrain[48] = {}".format(i,spiketrain[48]))
		for j in range(len(spiketrain[i]) - 1):
			ISI_distances.append(spiketrain[i][j+1] - spiketrain[i][j])
			#print("ISI Distance at index {} is {}".format(i,ISI_distances[i]))
	return ISI_distances



def plot_ISIH(ISI_distances = [], noOfbins = int):
	"""Plots ISI histogram
	"""
	ax = plt.gca()
	#calculating a density estimation line as well
	density = stats.gaussian_kde(ISI_distances)
	#print ("density: {}".format(density))
	
	#plt.figure(figsize = (10,10))
	n,bins,patches = plt.hist(ISI_distances,bins = noOfbins, histtype = 'stepfilled', normed = True,color = 'blue',linewidth = 1.0)
	plt.plot(bins,density(bins),'r--', label = 'Density estimation')
	'''
	plt.xlabel("ISI-distances(ms)")
	plt.ylabel("# of intervals per bin")
	plt.legend(loc = 'upper right')
	plt.title("ISI distances histogram")
	'''
	return ax
	#plt.show()


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
	#print ("Binned Spike Train1: {}".format(BinnedSpikeTrain1))#
	#print ("No of spikes in spike train 1: {}".format(np.count_nonzero(BinnedSpikeTrain1 == 1)))
	#print ("Binned Spike Train2: {}".format(BinnedSpikeTrain2))
	#print ("No of spikes in spike train 2: {}".format(np.count_nonzero(BinnedSpikeTrain2 == 1)))
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
	
	#print ("PSTH array {} PSTH size{}".format(PSTH1, np.size(PSTH1)))

	#print (PSTH1 + PSTH2)
	PSTH = PSTH1 + PSTH2
	'''
	x = np.arange(np.size(PSTH1))
	print ("x axis: {} psth size{}".format(x, PSTH1))
	plt.figure(figsize = (20,10))
	plt.bar(x,PSTH,width = binSize, color = 'k')
	plt.xlabel("Bin Counts")
	plt.ylabel("Firing rate (#Spikes/s)")
	plt.show()
	'''

	#ISI distances matrix
	ISI_distances = []
	'''
	ISI_distances = ISI(spiketrain1,spiketrain2)
	print("ISI distances: {}".format(ISI_distances))
	#plotting a line as well
	density = stats.gaussian_kde(ISI_distances)
	print ("density: {}".format(density))
	n,x,_ = plt.hist(ISI_distances,bins = 10, histtype = u'step',normed = False)
	#plt.plot(x,density(x))
	plt.xlabel("ISI-distances(ms)")
	plt.ylabel("# of intervals per bin")
	plt.title("ISI distances histogram")
	plt.show()
	'''
	spiketrain = []
	SpikeFile_retina = 'SpikeTrains/final_interpolated_retina.txt'
	SpikeFile_hippo = 'SpikeTrains/final_interpolated.txt'
	SpikeFile_long = 'SpikeTrains/final_interpolated_long.txt'
	try:
		with open(SpikeFile_long) as f:
			lines=f.readlines()
			for line in lines:
				spiketrain.append(np.fromstring(line, dtype=float, sep=' '))
	            #print(spiketrain)
		#print("spike train list is : {}, The shape is {}, elements in each row is {}".format(spiketrain,len(spiketrain), len(spiketrain[2])))
		
	except:
		print("File not found!!!")
	'''	
	ax = raster(spiketrain)
	plt.title('Raster Plot')
	plt.xlabel('Time(ms)')
	plt.ylabel('Neuron #')
	plt.show()

	ISI_distances = ISI_multi(spiketrain)

	plot_ISIH(ISI_distances, noOfbins = 500)
	
	PSTH_binCount,noOfbins,binSize = PSTH_multi(spiketrain, binSize = 50 , totalTime = 5000)
	x = np.arange(np.size(PSTH_binCount))
	plt.figure(figsize = (20,10))
	plt.bar(x,PSTH_binCount, color = 'k')
	plt.xlabel("Bin Counts")
	plt.ylabel("Firing rate (#Spikes/s)")
	plt.show()
	'''
	#plotting raster plot, PSTH and ISI on top of each other.....

	fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex = False ,figsize=(8, 12))
	fig.tight_layout()

	#ax1 = raster(spiketrain)
	#####raster plot
	for ith, trial in enumerate(spiketrain):
		ax1.vlines(trial, ith + .5, ith + 2.8)
	ax1.set_title('Raster Plot')
	#ax1.set_xlabel('Time(ms)')
	ax1.set_ylabel('Neuron #')

	#PSTH plot
	PSTH_binCount,noOfbins,binSize = PSTH_multi(spiketrain, binSize = 10 , totalTime = 40000)
	PSTH_realTimeAxis = PSTH_realTimeAxis(PSTH_binCount, binSize)	
	x = np.arange(np.size(PSTH_realTimeAxis))
	#plt.figure(figsize = (20,10))
	ax2.bar(x,PSTH_realTimeAxis, width = binSize, color = 'green')
	#ax2.set_xlabel("Bin Counts")
	ax2.set_ylabel("Firing rate (#Spikes/s)")
	ax2.set_title("PSTH")
	

	#ISI histogram-----------------------

	ISI_distances = ISI_multi(spiketrain)
	#calculating a density estimation line as well
	density = stats.gaussian_kde(ISI_distances)
	#print ("density: {}".format(density))
	
	#plt.figure(figsize = (10,10))
	n,bins,patches = plt.hist(ISI_distances,bins = noOfbins, histtype = 'step', normed = True,color = 'blue',linewidth = 1.0)
	ax3.plot(bins,density(bins),'r--', label = 'Density estimation')
	
	ax3.set_xlabel("Time(ms)")
	ax3.set_ylabel("# of intervals per bin")
	#ax3.legend(loc = 'upper right')
	ax3.set_title("ISI distances histogram")

	plt.show()

