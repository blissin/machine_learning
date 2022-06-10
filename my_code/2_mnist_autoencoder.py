import sys, os
sys.path.append("./dataset")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import pickle
from dataset.mnist import load_mnist

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
'''
#----------------------------------------------------------------------------------------------------------------------------------------
(train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(flatten = True, normalize = False)
def mnist_show(n) :
    image = train_image_data[n]
    image_reshaped = image.reshape(28, 28)
    image_reshaped.shape
    label = train_label_data[n]
    plt.figure(figsize = (4, 4))
    plt.title("sample of " + str(label))
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()

# mnist_show(2745)

#https://dkfk2747.tistory.com/18
#----------------------------------------------------------------------------------------------------------------------------------------

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
(train_data, _), (test_data, _) = mnist.load_data()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
# display(train_data, noisy_train_data)

input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
# autoencoder.summary()

autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=5,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data),
)
# predictions = autoencoder.predict(test_data)
# display(test_data, predictions)
predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)

#https://keras.io/examples/vision/autoencoder/
#----------------------------------------------------------------------------------------------------------------------------------------
'''



# Load digits data 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print shapes
# print("Shape of X_train: ", X_train.shape)
# print("Shape of y_train: ", y_train.shape)
# print("Shape of X_test: ", X_test.shape)
# print("Shape of y_test: ", y_test.shape)

# Normalize (divide by 255) input data
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Display images of the first 10 digits in the training set and their true lables
# fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
# n=0
# for i in range(0,2):
#     for j in range(0,5):
#         axs[i,j].matshow(X_train[n])
#         axs[i,j].set(title=y_train[n])
#         n=n+1
# plt.show() 

# Specify how much noise to add
level_of_noise=0.5

# Add random noise based on sampling from Gaussian distribution
X_train_noisy = X_train + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

# Enforce min-max boundaries so it does not go beyond [0,1] range
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Display images of the first 10 digits in the noisy training data
# fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
# n=0
# for i in range(0,2):
#     for j in range(0,5):
#         axs[i,j].matshow(X_train_noisy[n])
#         axs[i,j].set(title=y_train[n])
#         n=n+1
# plt.show() 

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train_noisy = X_train_noisy.reshape(60000, 784)
X_test_noisy = X_test_noisy.reshape(10000, 784)

#--- Define Shapes
n_inputs=X_train.shape[1] # number of input neurons = the number of features X_train

from tensorflow import keras
from keras.models import Model # for creating a Neural Network Autoencoder model
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, LeakyReLU, BatchNormalization # for adding layers to DAE model
from tensorflow.keras.utils import plot_model # for plotting model diagram
main_dir=os.path.dirname(sys.path[0])

#--- Input Layer 
# visible = Input(shape=(n_inputs,), name='Input-Layer') # Specify input shape


# #--- Encoder Layer
# e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
# e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
# e = LeakyReLU(name='Encoder-Layer-Activation')(e)

# #--- Middle Layer
# middle = Dense(units=n_inputs, activation='linear', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer')(e)

# #--- Decoder Layer
# d = Dense(units=n_inputs, name='Decoder-Layer')(middle)
# d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
# d = LeakyReLU(name='Decoder-Layer-Activation')(d)

# #--- Output layer
# output = Dense(units=n_inputs, activation='sigmoid', name='Output-Layer')(d)

# # Define denoising autoencoder model
# model = Model(inputs=visible, outputs=output, name='Denoising-Autoencoder-Model')

# # Compile denoising autoencoder model
# model.compile(optimizer='adam', loss='mse')

# # Print model summary
# print(model.summary())

# # Plot the denoising autoencoder model diagram
# plot_model(model, to_file=main_dir+'/pics/Denoising_Autoencoder.png', show_shapes=True, dpi=300)