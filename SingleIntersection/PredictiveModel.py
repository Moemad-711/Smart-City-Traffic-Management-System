import os

from numpy.core.fromnumeric import shape
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import sys 

import tensorflow as tf
import numpy as np
import pandas as pd
from spektral.layers.convolutional import ChebConv
from tensorflow import keras
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from utils import data_split



class PredictiveModel:
    def __init__(self,input_shape, adjacency_matrix, batch_size, prediction_steps):
        self.adjacency_matrix = adjacency_matrix
        self.adjacency_matrix_tensor = tf.convert_to_tensor(adjacency_matrix)
        self.batch_size = batch_size
        self.prediction_steps = prediction_steps
        self.kernel_size =  int((batch_size-prediction_steps)/4)+1
        self.chebychev_order = 1
        self.model = self.build_model(input_shape,self.kernel_size, self.chebychev_order)

    def build_model(self, input_shape, kernel_size, chebychev_order):
        """
        Build and Compile the Predictive Model
        """
        visible_x = Input(shape=input_shape)
        visible_a = Input(shape=self.adjacency_matrix_tensor.shape)
        visible_x_reshaped = Reshape(target_shape=(input_shape[1], input_shape[0], input_shape[2]))(visible_x)
        # Convolution Block Start
        # Y1 Calculator
        y1 = Conv1D(8, kernel_size)(visible_x_reshaped)
        print("Y1: ", y1.shape)
        # Sigmoid(Y2)
        y2 = Conv1D(8, kernel_size, activation="sigmoid")(visible_x_reshaped)
        print("Y2: ", y2.shape)
        # Bit-wise Multiply(Y1,Y2)
        glu1 = y1 * y2
        glu1_shape = glu1.shape
        print("Glu1: ", glu1_shape)

        # GCN_ChebyChev Convolution
        glu1_reshaped = Reshape(target_shape=(glu1_shape[2], glu1_shape[1], glu1_shape[3]))(glu1)

        chebConv1 = ChebConv(8, chebychev_order, activation='relu')([glu1_reshaped, visible_a])

        chebConv1_reshaped = Reshape((chebConv1.shape[2], chebConv1.shape[1], chebConv1.shape[3]))(chebConv1)
        print("chebConv1: ", chebConv1.shape)

        # Y1 Calculator
        y3 = Conv1D(8, kernel_size)(chebConv1_reshaped)
        print("Y3: ", y3.shape)
        # Sigmoid(Y2)
        y4 = Conv1D(8, kernel_size, activation="sigmoid")(chebConv1_reshaped)
        print("Y4: ", y4.shape)
        # Bit-wise Multiply(Y1,Y2)
        glu2 = y3 * y4
        print("Glu2: ", glu2.shape)
        # Convolution Block End

        # Convolution Block Start
        # Y1 Calculator
        y5 = Conv1D(8, kernel_size)(glu2)
        print("Y5: ", y5.shape)
        # Sigmoid(Y2)
        y6 = Conv1D(8, kernel_size, activation="sigmoid")(glu2)
        print("Y6: ", y6.shape)
        # Bit-wise Multiply(Y1,Y2)
        glu3 = y5 * y6
        print("Glu3: ", glu3.shape)

        # GCN_ChebyChev Convolution
        glu3_reshaped = Reshape(target_shape=(glu3.shape[2], glu3.shape[1], glu3.shape[3]))(glu3)

        chebConv2 = ChebConv(8, chebychev_order, activation='relu')([glu3_reshaped, visible_a])

        chebConv2_reshaped = Reshape(target_shape=(chebConv2.shape[2], chebConv2.shape[1], chebConv2.shape[3]))(chebConv2)
        print("chebConv2: ", chebConv2.shape)

        # Y1 Calculator
        y7 = Conv1D(8, kernel_size)(chebConv2_reshaped)
        print("Y3: ", y7.shape)
        # Sigmoid(Y2)
        y8 = Conv1D(8, kernel_size, activation="sigmoid")(chebConv2_reshaped)
        print("Y4: ", y8.shape)
        # Bit-wise Multiply(Y1,Y2)
        glu4 = y7 * y8
        print("Glu4: ", glu4.shape)
        # Convolution Block End

        # Fully-Connected layer
        glu4_reshaped = Reshape(target_shape=(glu4.shape[2], glu4.shape[1], glu4.shape[3]))(glu4)

        fully_connected = Dense(8)(glu4_reshaped)
        print("output: ", fully_connected.shape)

        model = keras.Model(inputs=[visible_x,visible_a], outputs= fully_connected, name='predictive_model')
        rmse = tf.keras.metrics.RootMeanSquaredError()
        model.compile(optimizer="adam", loss="mse", metrics=["mae", rmse])
        print(model.summary())

        return model

    def train_model(self, x_train, y_train, x_val, y_val, epochs):
        """
        Train the Predictive Model
        """
        a_train_repeated = np.tile(self.adjacency_matrix, (x_train.shape[0],1,1))
        print(a_train_repeated.shape)

        a_test_repeated = np.tile(self.adjacency_matrix, (x_val.shape[0],1,1))

        history = self.model.fit(x=[x_train, a_train_repeated],y=y_train, batch_size=1,epochs=epochs,validation_data=([x_val,a_test_repeated],y_val))
        print(history)
     
    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self.model.save(os.path.join(path, 'trained_predictive_model.h5'))
        plot_model(self.model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)

    
    def load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_predictive_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            self.model = loaded_model
        else:
            sys.exit("Model number not found")

class TestPredictiveModel:
    def __init__(self, adjacency_matrix, model_path):
        self.adjacency_matrix = adjacency_matrix
        self.model = self.load_my_model(model_path)


    def load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_predictive_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")
    
    def predict_one(self,input_x):

        a_repeated = np.tile(self.adjacency_matrix, (input_x.shape[0],len(self.adjacency_matrix),len(self.adjacency_matrix[0])))
        return self.model.predict(input_x,a_repeated)


    
    