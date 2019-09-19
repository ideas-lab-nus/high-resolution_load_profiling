import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score

class Bldg2profile:
	def __init__(self, dir = '', Pre = True):
		# the instance expect the directory of a csv file containing time series energy consumption data of a building
		# with the first column as the timestamp and the second column as the data
		self.data = pd.read_csv(dir)
		if Pre:
			self.preprocess()
		self.dates = self.data.index
		print('the building is metered from {} to {}.'.format(self.dates[0],self.dates[-1]))
		# initialize so that the plot function can be called anytime
		self.labelOri = np.asarray([0]*len(self.dates))
		self.label1 = None
		self.label2 = None
		self.labels = None
		self.labelExt = None

	def preprocess(self):
		# for preprocessing, the 1-dimension time series data is 1. max-normalized and 2. reshaped to form the daily profile
		c = self.data.columns
		# normalize the data so that the hyperparameters are generalizable 
		self.data[c[1]] = self.data[c[1]]/np.max(self.data[c[1]] )
		# aggregate the data in the same day
		self.data[c[0]] = pd.to_datetime(self.data[c[0]])
		self.data['date'] = self.data[c[0]].apply(lambda x:x.date())
		self.data = self.data.groupby('date')[c[1]].apply(list)
		self.data = self.data.apply(lambda x:pd.Series(x))

	def PreKmeans(self):
		# for preliminary K-means, iteratively calculate the CH index and pick K with the highest score
		kscores = []
		for k in range(2,10):
		    k1 = KMeans(n_clusters=k).fit(self.data)
		    kscores.append(calinski_harabasz_score(self.data,k1.labels_))
		k1 = KMeans(n_clusters=kscores.index(np.max(kscores))+2).fit(self.data)
		self.label1 = k1.labels_

	@staticmethod
	def CHforDBSCAN(data,label):
		# static method to calculate the CH index for the DBSCAN results within preliminary clusters
	    if sum(label==-1)/len(label)>.3:
	        raise ValueError
	    # transform all outliers into single clusters before calculation
	    for idx in range(len(label)):
	        if label[idx]==-1:
	            label[idx] = max(label)+1
	    return calinski_harabasz_score(data,label)
		
	def DBSCAN(self,subdata):
    	# find the optimal DBSCAN clustering result in the pre-defined parameter range
		scores = []
	    # the range of Eps and MinPt is fixed since the samples have been normalized
		for Eps in range(2,60,2):
			for MinPt in range(2,15):
				c1 = DBSCAN(eps=Eps/100, min_samples=MinPt).fit(subdata)
				try:
					scores.append(tuple([self.CHforDBSCAN(np.asarray(subdata),c1.labels_),Eps/100,MinPt]))
				except ValueError:
					continue
		scores.sort(reverse = True)
		c1 = DBSCAN(eps=scores[0][1], min_samples=scores[0][2]).fit(subdata)
		return c1.labels_

	def FinerDBSCAN(self):
		# for DBSCAN within preliminary clusters
		label2 = pd.DataFrame(columns = ['label_2','label'])
		# store the number of subclusters in self.summary2
		self.summary2 = {}

		for l in np.unique(self.label1):
		    subdata = self.data[self.label1 == l]
		    # use the DBSCAN method defined in this class to cluster the preliminary clusters with optimal parameters
		    sublabel = pd.DataFrame(self.DBSCAN(subdata),index = subdata.index,columns = ['label_2'])
		    # combine the clustering results of the preliminary clusters
		    sublabel['label'] = sublabel['label_2'].apply(lambda x: '{}_{}'.format(l,x) if x != -1 else '-1')
		    label2 = pd.concat([label2, sublabel])
		    self.summary2[l] = sublabel['label_2'].max()+1
	    
	    # change the label back to int for the ease of plot method
		l = list(np.unique(label2['label']))[1:]
		label2['label'] = label2['label'].apply(lambda x: l.index(x) if x != '-1' else -1)
		self.label2 = np.asarray(label2.sort_index()['label'])


	def plot(self, step, method = ''):
		ls = [self.labelOri,self.label1,self.label2,self.labels,self.labelExt]
		steps = ['Pre-processing','Pre-Kmeans','Finer DBSCAN','Final',method]
		labels = list(np.unique(ls[step]))
		fig = plt.figure(figsize=(10,5))
		lines = ['-', '--', '-.', ':']
		# deal with the outliers for the result of the last two steps
		if step > 1:
			try:
				plt.plot(self.data.loc[ls[step]==-1].T,color = (.2,.2,.2),alpha = .05)
				labels = labels[1:]
			except:
				pass
		color_palette = sns.color_palette('hls', len(labels))
		for i in labels:
		    plt.plot(self.data.loc[ls[step]==i].T,color = color_palette[labels.index(i)],alpha = .05)
		for i in labels:
		    plt.plot(np.mean(self.data.loc[ls[step]==i]),color = color_palette[labels.index(i)],
		             linestyle=lines[i%4],label='C'+str(i),linewidth = 3)
		plt.title('{} result'.format(steps[step]),size=30)
		plt.legend(handlelength = 1.5,fontsize=24,loc='upper left')
		plt.yticks(size=12)
		plt.xticks(size=12)
		plt.xlim(0,len(self.data.columns)-1)
		plt.ylim(0,1)
		plt.show()

	def postprocess(self):
		# calculate the Pearson Correlation Coefficients between the centroids
	    centroids = self.data[self.label2 != -1].groupby(self.label2[self.label2 != -1]).mean()
	    corrMatrix = centroids.T.corr(method='pearson')
	    labels = corrMatrix.index
	    
	    # rank the cluster pairs according to PCC, then merge the pairs with higher scores first
	    pairs = []    
	    for i in range(0,len(corrMatrix)):
	        for j in range(i+1,len(corrMatrix)):
	            pairs.append([corrMatrix.iloc[i,j],labels[i],labels[j]])
	    pairs.sort(reverse=True)

	    # merge the clusters with PCC higher than the threshold
	    # test threshold from 0.75 to 1 and decide based on CH index
	    maxScore=0
	    for th in range(75,100):
	        left = set(labels)
	        combine = {}
	        count = 0
	        for i in pairs:
	            if i[0]>th/100 and len(set(i[1:])&left)>0:
	                for key in combine.keys():
	                    if (i[1] in combine[key]) or (i[2] in combine[key]):
	                        combine[key] = combine[key]|set(i[1:])
	                        left -= set(i[1:])
	                        break
	                else:
	                    combine[count] = set(i[1:])
	                    left -= set(i[1:])
	                    count += 1
	        for i in left:
	            combine[count] = [i]
	            count += 1
	        
	        # if the result is acceptable, generate the labels after merging similar clusters
	        if len(combine)>1:
	            lookup = {-1:-1}
	            for key in combine.keys():
	                for l in combine[key]:
	                    lookup[l] = key

	            tag = np.asarray([lookup[x] for x in self.label2])
	            subdata = self.data[tag!=-1]
	            s = calinski_harabasz_score(subdata,tag[tag!=-1])
	            if s>=maxScore:
	                maxScore = s
	                self.labels = tag

	def profile(self, showKmeans = False, showDBSCAN = False, showFinal = False):
		# the profiling algorithm consists of three main steps: PreKmeans, FinerDBSCAN and post-processing
		# three argument indicating whether or not to show the intermediate results

		self.PreKmeans()

		if showKmeans:
			print('The preliminary K-means clustering resulted in {} clusters.'.format(len(np.unique(self.label1))))
			self.plot(1)

		self.FinerDBSCAN()

		if showDBSCAN:
			print('Within cluster DBSCAN gerated {} for the preliminary clusters.'.format(self.summary2))
			self.plot(2)

		self.postprocess()

		if showFinal:
			print('The clustering resulted in {} typical profiles.'.format(len(np.unique(self.labels))-1))
			self.plot(3)