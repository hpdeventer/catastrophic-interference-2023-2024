import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Embedding, Reshape, RepeatVector, Multiply, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
import matplotlib.pyplot as plt
import numpy as np

# Define the target function
def f(x1, x2):
    return np.sin(4 * np.pi * x1) * np.sin(4 * np.pi * x2)

import random

def create_partitions(n_partitions, seed=1):
    # Set seed
    random.seed(seed)

    # Create a list to store the partitions' limits
    partitions = []

    # Create equally sized intervals in [0, 1]^2
    for i in range(n_partitions):
        for j in range(n_partitions):
            partitions.append([(i/n_partitions, (i+1)/n_partitions), (j/n_partitions, (j+1)/n_partitions)])

    # Randomly shuffle the order of the partitions
    random.shuffle(partitions)

    return partitions

def generate_training_data(partitions, n_samples):
    # Create a list to store all training data and labels
    all_X_train = []
    all_y_train = []

    # For each partition generate training points and train all models on them
    for idx, ((x1_min, x1_max), (x2_min, x2_max)) in enumerate(partitions):
        # Generate training data within the current partition and evaluate the function at these points
        X_train_partition = np.random.uniform([x1_min, x2_min], [x1_max, x2_max], (n_samples, 2))
        y_train_partition = f(X_train_partition[:, 0], X_train_partition[:, 1])

        # Append the current partition's data to all_X_train and all_y_train 
        all_X_train.append(X_train_partition)
        all_y_train.append(y_train_partition)

    # Concatenate all training data and labels into numpy arrays
    all_X_train = np.concatenate(all_X_train, axis=0)
    all_y_train = np.concatenate(all_y_train, axis=0)

    return all_X_train, all_y_train

def plot_training_data(partitions,X,y,n_samples, plot_name='', save=False):
    
    n_partitions = 4

    values = np.linspace(0, 1, n_partitions**2)
    colormap = plt.cm.viridis
    colors = colormap(values)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for idx in range(len(partitions)):
        # Plot the sampled points for visual confirmation
        ax.scatter(X[idx*n_samples:(idx+1)*n_samples, 0], X[idx*n_samples:(idx+1)*n_samples, 1], color=colors[idx % len(colors)], alpha=0.82, s=10., label='Partition '+str(idx+1))
        ax.text((partitions[idx][0][0]+partitions[idx][0][1])/2,(partitions[idx][1][0]+partitions[idx][1][1])/2,str(idx+1), color='black',ha='center',va='center',weight='bold', fontsize=12,bbox=dict(facecolor='white', edgecolor='none'))

    ax.set_xlabel('$x_1$') 
    ax.set_ylabel('$x_2$')

    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if save:
        plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def predict_models(model_list):
    
    # Create a grid of points in the domain [0, 1]^2
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    X_test = np.stack([X1.flatten(), X2.flatten()]).T
    
    # List to store all predictions
    all_pred = []

    for (model, name) in model_list:
        print(f"Predicting with {name}...")
        y_pred = model.predict(X_test,verbose=None)

        # Reshape the prediction array to have the same shape as X1 and X2
        Z_pred = y_pred.reshape(X1.shape)
        
        # Store Z_pred in list
        all_pred.append(Z_pred)

    return all_pred

def plot_predictions(model_list, predictions, plot_name='', save=False):
    fig, axs = plt.subplots(1, len(model_list), figsize=(len(model_list)*3, 5), sharex=True, sharey=True)
    
    vmin = -1.0 #min([pred.min() for pred in predictions])
    vmax = +1.0 #max([pred.max() for pred in predictions])

    for (model, name), ax, Z_pred in zip(model_list, axs.flatten(), predictions):
    
        # Plot the predicted function using imshow with shared color limits
        im = ax.imshow(Z_pred, extent=[0, 1, 0 ,1], origin='lower', cmap='viridis', vmin=vmin,vmax=vmax)
        
        ax.set_title(name)

    for ax in axs:
        ax.set_xlabel('$x_1$') 

    axs[0].set_ylabel('$x_2$')  # only set y-label for the first subplot

    # Create a divider for the existing axes instance.
    divider = make_axes_locatable(ax)

    # Add an axes to the right of the main axes.
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add a colorbar to the last subplot with adjusted height
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def pseudorehearsal(input_dim: int, num_samples: int, 
                    model: tf.keras.Model, 
                    train_x: np.ndarray, 
                    train_y: np.ndarray, 
                    seed_val: int) -> tuple:
    '''
    Generate pseudorehearsal samples using a given model, combine them with the original training data, 
    and return the shuffled combined dataset.

    Args:
    - input_dim (int): Dimension of the input space.
    - num_samples (int): Number of pseudorehearsal samples to generate.
    - model (tf.keras.Model): Model to generate rehearsal targets.
    - train_x (np.ndarray): Original training input data.
    - train_y (np.ndarray): Original training target data.
    - seed_val (int): Seed value for random number generation.

    Returns:
    - tuple: Shuffled combined input and target data.
    '''
    
    # Generate pseudorehearsal samples, assume uniform over [0,1]^n
    rng = np.random.default_rng(seed_val)
    rehearsal_samples = rng.uniform(0, 1, size=(num_samples, input_dim))
    #print("rehearsal starts")
    rehearsal_targets = model(rehearsal_samples)
    #print("ends")
    
    # Combine training data with pseudorehearsal data and shuffle them
    combined_x = np.concatenate([train_x, rehearsal_samples], axis=0)
    combined_y = np.concatenate([train_y, rehearsal_targets], axis=0)
    
    indices = np.arange(combined_x.shape[0])
    rng.shuffle(indices)

    shuffled_x = combined_x[indices]
    shuffled_y = combined_y[indices]

    return shuffled_x, shuffled_y

def initialize_all_models(input_dimension: int, 
                          seed_val: int, 
                          output_dim: int = 1,
                          hidden_units_wide: int = 1000,
                          hidden_units_deep: int = 16,
                          hidden_layers: int = 8,
                          num_exps: int = 6) -> list:
    """Initialize models with given configurations."""
    common_args = {
        'input_dim': input_dimension, 
        'output_dim': output_dim, 
        #'seed': seed_val
    }

    models = [
        (create_wide_relu_ann(hidden_units=hidden_units_wide, **common_args), "Wide ReLU ANN"),
        (create_deep_relu_ann(hidden_units=hidden_units_deep, hidden_layers=hidden_layers, **common_args), "Deep ReLU ANN"),
    ]

    for partition_num in [20]:
        models.append((SplineANN(subintervals=4*partition_num, **common_args), 
                       f"Spline ANN (z={partition_num})"))
        # input_dim, partition_number, output_dim, num_exps, periodic=False
        models.append((ABELSpline(partition_num=partition_num, num_exps=num_exps, **common_args), 
                       f"ABEL-Spline (z={partition_num})"))
        models.append((LookupTableModel(partition_num=partition_num, default_val=-1., **common_args),
                       f"Lookup Table (z={partition_num})"))

    return models

def compile_models(models, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_absolute_error'):
    """Compile TensorFlow/Keras models."""
    for model, name in models:
        model.compile(optimizer=optimizer, loss=loss)


def create_linear_model(input_dim: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a linear model with rescaling and a dense layer.
    
    Args:
        input_dim: The input dimension.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the linear layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model

def create_wide_relu_ann(input_dim: int, hidden_units: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a wide ReLU activated artificial neural network.
    
    Args:
        input_dim: The input dimension.
        hidden_units: The number of hidden units.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the ANN layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    model.add(Dense(hidden_units, activation='relu', kernel_initializer=initializer))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model

def create_deep_relu_ann(input_dim: int, hidden_units: int, hidden_layers: int, output_dim: int = 1, seed: int = 42) -> Sequential:
    """Create a deep ReLU activated artificial neural network.
    
    Args:
        input_dim: The input dimension.
        hidden_units: The number of hidden units.
        hidden_layers: The number of hidden layers.
        output_dim: The output dimension. Defaults to 1.
        seed: The seed for deterministic weight initialization. Defaults to 42.
        
    Returns:
        A Sequential model consisting of the ANN layers.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)

    model = Sequential()
    model.add(Rescaling(scale=2., offset=-1., input_shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_units, activation='relu', kernel_initializer=initializer))
    model.add(Dense(output_dim, kernel_initializer=initializer))
    return model 

class LookupTableModel(tf.keras.Model):
    def __init__(self, input_dim: int, partition_num: int, output_dim: int = 1,
                 default_val: float = 0.0, seed: int = 55):
        super(LookupTableModel, self).__init__()
        self.input_dim = input_dim
        self.partition_num = partition_num
        initializer = tf.keras.initializers.RandomUniform(seed=seed)
        self.embedding = tf.keras.layers.Embedding(partition_num**input_dim + 1, output_dim,
                                                   embeddings_initializer=initializer)
        self.default_val = tf.constant(default_val, dtype=tf.float32)

        # Set last entry in embedding to be default value
        self.embedding.build((None,))
        self.embedding.set_weights([tf.concat([self.embedding.weights[0].numpy()[:-1],
                                               [[default_val]*output_dim]], axis=0)])
        
        # Changed to integer type
        self.partition_num_powers = tf.cast(tf.pow(partition_num, tf.range(input_dim)), dtype=tf.int32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # ReLU operation to drop negative inputs 
        inputs = tf.maximum(0., inputs)

        # Scale, floor and cast to integers
        scaled_input = tf.cast(tf.floor(inputs * self.partition_num), dtype=tf.int32)

        # Bounding the indices by partition number - 1
        bounded_inputs = tf.minimum(scaled_input, self.partition_num - 1)

        # Flatten each vector to get a single index for each sample.
        indices = tf.reduce_sum(bounded_inputs * self.partition_num_powers, axis=1)
        
        outputs = self.embedding(indices)
        return outputs

class SplineLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplineLayer, self).__init__()

    def call(self, inputs):
        x = tf.math.floormod(inputs[..., tf.newaxis], 1.)  # add an axis and retain only fractional part
        polynomial_1 = (tf.math.pow(x, 3) / 6)
        polynomial_2 = ((-3. * tf.math.pow(x, 3) + 3. * tf.math.pow(x, 2) + (3. * x) + 1.) / 6.)
        polynomial_3 = ((3 * tf.math.pow(x, 3) - 6 * tf.math.pow(x, 2) + 4.) / 6.)
        polynomial_4 = (tf.math.pow((1. - x), 3) / 6.)

        return tf.concat([polynomial_4, polynomial_3, polynomial_2, polynomial_1], axis=-1)

class CoeffLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, density, output_dim, seed=55, periodic=False):
        super(CoeffLayer, self).__init__()
        self.input_dim = input_dim
        self.density = density
        self.output_dim = output_dim
        self.periodic = periodic
        self.period = (self.density - 3) if self.periodic else (self.density )
        self.reshape = tf.keras.layers.Reshape((self.input_dim * 4,), name="Reshape_Ints")
        self.undo_reshape = tf.keras.layers.Reshape((self.input_dim, 4,), name="Undo_Reshape_Ints")
        self.input_dimension_shift = tf.repeat(tf.range(0, self.input_dim, dtype=tf.float32) * self.density, 4)
        self.embedding = tf.keras.layers.Embedding(
            input_dim= self.input_dim * self.density,
            output_dim=self.output_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(seed=seed),
            trainable=True)

    def call(self, inputs):
        x = tf.math.floor(inputs[..., tf.newaxis])         # add an axis and calculate floor
        x = tf.concat([x, x+1, x+2, x+3], axis=-1)  # Concatenate the tensors along the last axis.
        x = tf.math.floormod(x, self.period) # Compute modulus to wrap around values exceeding bounds
        x = self.reshape(x)
        x = tf.cast(tf.nn.bias_add(x, self.input_dimension_shift), tf.int32)
        x = self.undo_reshape(x)
        return  self.embedding(x)

class SplineANN(tf.keras.Model):
    """
    This is a Cubic Spline Model.
    It maps values from [0,1]^n to R^m using cardinal cubic spline functions.
    """
    def __init__(self, input_dim, subintervals, output_dim, periodic=False):
        super(SplineANN, self).__init__()
        self.input_dim = input_dim
        self.density = subintervals + 3
        self.output_dim = output_dim
        self.splines = SplineLayer()
        self.coefficients = CoeffLayer(self.input_dim, self.density, output_dim, periodic=periodic)

    def call(self, inputs):
        inputs_scaled_by_density = tf.multiply(inputs, (self.density - 3))
        splines = self.splines(inputs_scaled_by_density)
        repeated_splines = tf.repeat(splines[..., tf.newaxis], repeats=self.output_dim, axis=-1)
        coefficients = self.coefficients(inputs_scaled_by_density)        
        y = tf.reduce_sum((repeated_splines * coefficients), axis=-2, keepdims=False)

        return tf.reduce_sum(y, axis=-2, keepdims=False) # Sum over input dimension

class ABELSpline(tf.keras.Model):
    def __init__(self, input_dim, partition_num, output_dim, num_exps, periodic=False):
        super(ABELSpline, self).__init__()
        
        self.input_dim = input_dim
        self.subintervals = 4*partition_num
        self.density = self.subintervals + 3
        self.output_dim = output_dim
        self.num_exps = num_exps
        self.F_spline = SplineANN(self.input_dim, self.subintervals, self.output_dim, periodic)
        
        if num_exps > 0:
            self.attenuation = tf.constant(-2.*tf.math.log(tf.range(0.,num_exps)+1.), dtype=tf.float32)
            self.G_spline = SplineANN(self.input_dim, self.subintervals, self.output_dim*self.num_exps, periodic)
            self.H_spline = SplineANN(self.input_dim, self.subintervals, self.output_dim*self.num_exps, periodic)
    
    def call(self, inputs):
        output_acc = self.F_spline(inputs)
        if self.num_exps > 0:
            G = tf.reshape(self.G_spline(inputs), (-1, self.output_dim, self.num_exps))
            Exp_G = tf.reduce_sum(tf.exp(G + self.attenuation), axis=-1,keepdims=False)
            
            H = tf.reshape(self.H_spline(inputs), (-1, self.output_dim, self.num_exps))
            Exp_H = tf.reduce_sum(tf.exp(H + self.attenuation), axis=-1, keepdims=False)
            output_acc = tf.keras.layers.Add()([output_acc, tf.keras.layers.subtract([Exp_G, Exp_H])])
        
        return output_acc
