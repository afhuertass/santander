
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout

def get_model( input_features , nhidden ):

	model = Sequential()


	model.add( Dense( nhidden , input_dim = input_features )  )
	model.add( Activation("elu"  , name = "l1")  )

	model.add( BatchNormalization() )
	model.add(Dense(nhidden))
	model.add( Activation("elu" , name = "l2")  )
	model.add ( Dropout( 0.2 ) )

	model.add( BatchNormalization() )
	model.add( Dense(nhidden ) )
	model.add(Activation("elu" , name = "l3") )
	model.add ( Dropout( 0.2 ) )

	model.add( BatchNormalization() )
	model.add( Dense( nhidden ) )
	model.add(Activation("elu" , name = "l4") )
	model.add ( Dropout( 0.2 ) )

	#model.add( BatchNormalization() )
	model.add( Dense( 1  , activation="linear")  )

	#model.add( Activation("activation='linear'"))
	#model.add( Activation("relu" ,  name = "output")  )
	

	return model 



