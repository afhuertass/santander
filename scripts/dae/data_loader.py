import pandas as pd 
import numpy as np 
import keras 

import os 
random_state = 666

class DataGenerator(object):

	def __init__(self , path , batch_size):

		self.path = path 
		self.data = pd.read_csv(path)
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

	def sample_data( self  ):

		d = self.data.sample( self.batch_size , random_state = random_state )

		x = d.copy().values
		y = d.copy().values 

		# [ batch_size , nfeatures ]
		#print("shappeee ")
	
		nfeatures2permute = int( 0.50* self.nfeatures )


				# 15 
		col2replace = np.random.choice( self.nfeatures , nfeatures2permute )
		noise = self.data.sample( self.batch_size , random_state = random_state ).values
		#print( noise.shape )
		#print( x.shape )
		x[: , col2replace  ] = noise[ : , col2replace ]
		# lista de 
		#rows2select = np.random.choice( self.nfeatures , nfeatures2permute ) 


		#x[ : ,rows2replace ] = x[ : , rows2select]
		return x , y 





