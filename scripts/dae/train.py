#keras 
import pandas as pd
import numpy as np 
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import DataGenerator 
from keras.models import load_model
from keras import backend as K
import model

from scipy.special import erfinv

NEPOCHS = 10000
batch_size = 128
learning_rate = 1e-4

features_input = 100 
nhidden = 1000

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def train():

	# create dataset
	dataGenerator = DataGenerator( "../../data/train_norm.csv" , batch_size )
	features_input = dataGenerator.getNFeatures()
	steps_per_epoch  = dataGenerator.getSteps()
	#generator = dataGenerator.generate()

	m = model.get_model(features_input , nhidden)
	decay_rate =  learning_rate / NEPOCHS 
	optimizer = optimizers.Adam(lr = learning_rate , decay = decay_rate  )

	callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath= "./best_dae", monitor='val_loss', save_best_only=True)]

	m.compile( loss = "mean_squared_error"  , optimizer = optimizer , metrics = ["mse"] )


	m.fit_generator( generator = dataGenerator.generate(), steps_per_epoch = steps_per_epoch , epochs = NEPOCHS , callbacks = callbacks , validation_data = dataGenerator.generate()
			 , validation_steps = steps_per_epoch )





def predict( file ="train"):


	model = load_model("./best_dae")

	#dataGenerator = DataGenerator("../../data/sparse/train.csv" )
	df = pd.read_csv("../../data/{}_norm.csv".format( file )  )
	layers_names = [u"l1" , u"l2" , u"l3" , u"l4"]
	inp = model.input                                           # input placeholder
	print( [x.name for x in model.layers])
	outputs = [layer.output for layer in model.layers  if layer.name in  layers_names ]          # all layer outputs
	functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

	# Testing
	#test = np.random.random(input_shape)[np.newaxis,...]


	X =  df.values[:100]
	outputs_all = []
	i = 1 
	for g, df_ in df.groupby(np.arange(len( df )) // 128 ):
		
		layer_outs = functor([  df_.values , 1.])
		shape = df_.shape[0]
		outputs = np.array ( layer_outs   )

		#outputs = outputs.reshape( (  shape , -1 ))
		elems = []
		for elem in range(0,outputs.shape[1] ):
			feats = []
			for j in range(0, len( layers_names )  ) :
				d = outputs[ j , elem , :  ] 
				feats.append( d )
			feats = np.hstack( feats )
			elems.append( feats )

		elems = np.array( elems )
		print( elems.shape )
		print( i*128 )
		i = i + 1 
		outputs_all.append( elems )


	new_trainData = np.vstack( outputs_all )
	print( new_trainData.shape )

	np.save("../../data/{}_new.csv".format( file ) , new_trainData )
	df_ = pd.DataFrame( new_trainData ) #.to_csv( "../data/")
	df_.columns = np.arange(  new_trainData.shape[1] )
	print( "Gauss ranking ")
	df_ = df_.apply( rank_gauss )

	df_.to_csv("../../data/{}_new.csv".format(file) , index = False  )
	print( df_.head() )
	return "" 

if __name__ =="__main__":

	#train()
	#fakedata()
	predict("test")