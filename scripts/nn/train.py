#keras 
import pandas as pd
import numpy as np 
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint


from data_loader import DataGenerator 
from keras.models import load_model
from keras import backend as K

import model
import keras.losses
import keras.metrics 



NEPOCHS = 10000
batch_size = 64
learning_rate = 1e-6

features_input = 100 
nhidden = 1000

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(   K.square(  y_pred -  y_true  ), axis=-1)) 


keras.losses.root_mean_squared_error = root_mean_squared_error
keras.metrics.root_mean_squared_error = root_mean_squared_error

def train():

	# create dataset
	dataGenerator = DataGenerator( "../../data/train_new.csv" , "../../data/labels_train.csv" , batch_size )
	features_input = dataGenerator.getNFeatures()
	steps_per_epoch  = dataGenerator.getSteps()
	#generator = dataGenerator.generate()

	m = model.get_model(features_input , nhidden)
	decay_rate =  learning_rate / NEPOCHS 
	optimizer = optimizers.Adam(lr = learning_rate , decay = decay_rate  )

	callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath= "./best_m", monitor='val_loss', save_best_only=True)]

	m.compile( loss = root_mean_squared_error  , optimizer = optimizer , metrics = [ root_mean_squared_error ] ) 


	m.fit_generator( generator = dataGenerator.generate(), steps_per_epoch = steps_per_epoch , epochs = NEPOCHS , callbacks = callbacks , validation_data = dataGenerator.generate()
			 , validation_steps = steps_per_epoch )


	#y_train = m.predict( dataGenerator.getData()  )
	#print( y_train )


def predict2():
	model = load_model("./best_m")

	df = pd.read_csv( "../../data/test_new.csv" )
	y_preds = []

	for g, df_ in df.groupby(np.arange(len( df )) // 128 ):

		y_s = model.predict( df_.values  )
		y_s = y_s.reshape( (1 , -1 ) ).flatten()
		#print( y_s.shape )
		y_preds.append(   y_s )

	preds = []
	for b in y_preds:
		for x in b:
			preds.append( x )



	y_preds = np.array( preds )
	#y_preds = y_preds.flatten()

	pd.DataFrame( { "index": np.arange( y_preds.shape[0] ) , 'preds': y_preds }).to_csv("../../data/preds1.csv")


if __name__ =="__main__":

	#train()
	#fakedata()
	predict2()