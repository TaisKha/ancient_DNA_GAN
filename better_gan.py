
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.stats import binom, uniform
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import math
import matplotlib.pyplot as plt

read_length = 60
a, b = (0.001, 0.05)
train_size = 1000
N_reads = 1000

def histogram(data, size, save=False, n=0):
  X = np.arange(read_length)
  Y = np.sum(data, axis=(0, 1))
  Y /= np.max(Y)
  fig = plt.figure()
  plt.plot(X, Y, "o-")
  plt.title(f'epoch = {n}')
  plt.ylim(0, 1)
  if save:
    plt.savefig(f'./images-e1000-t1000-n1000-dense-batchnorm-conv1d/{n}.png')
#  plt.show()  
  plt.close(fig)
def get_rand_ind(left: int, right: int, count: int) -> list:
    res_indexes = set()
    while len(res_indexes) < count:
        new_index = random.randrange(left, right)
        if new_index not in res_indexes:
            res_indexes.add(new_index)
    return list(res_indexes)
    
def dataset_generator(dataset_identifier):
  dataset = np.empty((dataset_identifier.shape[0], N_reads, read_length))
  sum_rl = 0
  s = 0
  for i in range(1, read_length+1):
    sum_rl += 1 / i
  for _i in range(dataset_identifier.shape[0]):
    data = []
    err = np.exp(random.uniform(np.log(a), np.log(b)))
    s += err
    corr = read_length * err / sum_rl
    if dataset_identifier[_i] == 0:
      for _j in range(N_reads):
        k = binom.rvs(read_length, err)
        err_lst = get_rand_ind(0, read_length, k)
        res = [1 if i in err_lst else 0 for i in range(read_length)]
        data.append(res)
    elif dataset_identifier[_i] == 1:
      for _j in range(N_reads):
        cube = [random.uniform(0, 1) for i in range(read_length)]
        p = [1/x for x in range(1, read_length+1)]
        res = [1 if cube[i] < p[i] * corr else 0 for i in range(read_length)]
        data.append(res)
    data = np.array(data)
    dataset[_i] = data
  print(s)  
  return dataset

def meanlayer(tensors):
  out = tf.reduce_mean(tensors, axis=2)
  return out

y_train = np.ones(train_size)
X_train = dataset_generator(y_train)

histogram(X_train, train_size)

BATCH_SIZE = 100
EPOCHES = 1000

def generator_model():
#images-e1000-t1000-n1000-dense-batchnorm-conv1d
    model = Sequential()
    model.add(Dense(N_reads, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(N_reads*read_length, activation='relu'))
    model.add(Reshape((N_reads, read_length)))
    model.add(Conv1D(read_length, 1, padding='same'))
    model.add(Activation('relu'))
    return model
#images_e1000_t500_N1000_dense_and_batchnorm
#mages_e10000_t1000_N1000_dense_and_batchnorm
    #model = Sequential()
    #model.add(Dense(N_reads, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dense(N_reads*read_length, activation='relu'))
    #model.add(Reshape((N_reads, read_length)))
#images_e500_t500_N1000_addrelu
   # model = Sequential()
   # model.add(Conv1D(read_length*2, 1, strides=2))
   # model.add(Activation('relu'))
   # model.add(Conv1D(read_length, 1, padding='same'))
   # model.add(Activation('relu'))


def discriminator_model():

    model = Sequential()
    model.add(Conv1D(read_length // 4, 1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(Lambda(meanlayer, (BATCH_SIZE, N_reads, 1)))
    model.add(Dense(read_length))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(Dense(read_length, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def train():
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)

    d_optim = tf.keras.optimizers.Adam(1e-4)
    g_optim = tf.keras.optimizers.Adam(1e-4)
    g_loss_func = tf.losses.BinaryCrossentropy()
    d_loss_func = tf.losses.BinaryCrossentropy()

    generator.compile(loss=g_loss_func)
    discriminator_on_generator.compile(
        loss=d_loss_func, optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss=d_loss_func, optimizer=d_optim)
    for epoch in range(EPOCHES):
        print("Epoch is", epoch)
        
        noise = np.array(np.random.uniform(size=(1, N_reads)))
        generated_data_to_show = generator.predict(noise, verbose=0)
        histogram(generated_data_to_show, 1, save=True, n=epoch)##############HISTOGRAM
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            # noise = np.zeros((BATCH_SIZE, N_reads, read_length))
            # noise = np.array(np.random.randint(2, size=(BATCH_SIZE, N_reads, read_length)), dtype='float32')
            noise = np.array(np.random.uniform(size=(BATCH_SIZE, N_reads)))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_data = generator.predict(noise, verbose=0)
            X = np.concatenate((image_batch, generated_data))
            y = np.array([0] * BATCH_SIZE + [1] * BATCH_SIZE).reshape(2*BATCH_SIZE, 1)
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))


            # noise = np.array(np.random.randint(2, size=(BATCH_SIZE, N_reads, read_length)), dtype='float32')
            noise = np.array(np.random.uniform(size=(BATCH_SIZE, N_reads)))     
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
#            if index % 10 == 9:
#                generator.save_weights('../generator', True)
#                discriminator.save_weights('../discriminator', True)

train()

