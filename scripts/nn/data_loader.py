import pandas as pd 
import numpy as np 
import keras 

import os 
random_state = 666

class DataGenerator(object):

	def __init__(self , path , path_labels , batch_size):

		self.path = path 
		self.data = pd.read_csv(path ).values
		self.Y = pd.read_csv(path_labels ).values

		self.batch_size = batch_size 

		self.nsamples = self.data.shape[ 0 ]
		self.nfeatures = self.data.shape[ 1 ]


	def generate( self ):

		while 1:
			imax = int( self.nsamples / self.batch_size )

			for i in range(imax):

				x , y = self.sample_data( )
				yield x , y 

	def getSteps(self):

		return self.nsamples / self.batch_size 

	def getNFeatures(self):

		return self.nfeatures 

	def getData(self):

		return self.data 

	def sample_data( self  ):

		indx = np.random.choice(  self.nsamples , self.batch_size )

		x = self.data[ indx , : ]
		y = self.Y[indx , : ]
		#print(  y.mean() )
		#print( x.shape )
		#print( y.shape )

		return x ,y 





