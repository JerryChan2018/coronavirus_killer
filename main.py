import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import itertools
from scipy.stats import pearsonr
from collections import Counter
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential

np.set_printoptions(linewidth=100)

#%matplotlib inline

data = pd.read_table('D:\Bandizip\Documentary\Bio_data\Table_S8_machine_learning_input.txt', index_col=0)

data.head()

# print some statistics from this dataset

print('total sgRNAs:', len(data))
print('total series:', len(data['perfect match sgRNA'].unique()))
print('total genes:', len(data['gene'].unique()))
print('phenotypes from K562 and Jurkat: %.2f%%'%(len(data[(data.K562==True) & (data.Jurkat==True)])/
                                                 float(len(data))*100))

# make a list of tuples pairing genome and input sequences

sequence_tuples = list(zip(data['genome input'], data['sgRNA input']))


def binarize_sequence(sequence):
    """
    converts a 26-base nucleotide string to a binarized array of shape (4,26)
    """
    arr = np.zeros((4, 26))
    for i in range(26):
        if sequence[i] == 'A':
            arr[0, i] = 1
        elif sequence[i] == 'C':
            arr[1, i] = 1
        elif sequence[i] == 'G':
            arr[2, i] = 1
        elif sequence[i] == 'T':
            arr[3, i] = 1
        else:
            raise Exception('sequence contains characters other than A,G,C,T \n%s' % sequence)

    return arr

# example of what the above function is doing
# using the genome input sequence from the first row of the data table

test_sequence = list(sequence_tuples)[0][0]
print(test_sequence, '\n')
print(binarize_sequence(test_sequence), '\n')
print(binarize_sequence(test_sequence).shape, '\n')

fig,ax=plt.subplots()
ax.imshow(binarize_sequence(test_sequence))
ax.set_yticks(range(4))
ax.set_yticklabels(list('ACGT'))
ax.set_xticks([])

# for each tuple, binarize the sequences and stack them into a 3D array of shape (4,26,2)
print(sequence_tuples)
stacked_arrs = [np.stack((binarize_sequence(genome_input), binarize_sequence(sgrna_input)), axis=2)
                for (genome_input, sgrna_input) in sequence_tuples]

# the feature input X is a 4D array containing all of the 3D arrays generated above
print(stacked_arrs)
X = np.concatenate([arr[np.newaxis] for arr in stacked_arrs])

# the target input y is a 1D array of relative activities

y = data['mean relative gamma'].values

# an array of series IDs will allow mapping of each element in X or y
# to be used in assigning each series to the training or validation set

series = data['perfect match sgRNA']

# check the shape of each array

print('X:', X.shape)
print('y:', y.shape)
print('series:', series.shape)

# randomly select 20% of sgRNA series to be set aside for validation

np.random.seed(99)
val_series = np.random.choice(np.unique(series), size=int(len(np.unique(series))*.20), replace=False)
val_indices = np.where(np.isin(series, val_series))
train_indices = np.where(~np.isin(series, val_series))

# define train and validation sets

X_train = X[train_indices]
X_val = X[val_indices]
y_train = y[train_indices]
y_val = y[val_indices]

# check the shape of each array

print('X train:', X_train.shape)
print('y train:', y_train.shape, '\n')
print('X validation:', X_val.shape)
print('y validation:', y_val.shape)

# assign training target values to 5 bins with evenly spaced edges
# relative activities below 0 or above 1 are included in the lowest or highest bin, respectively

nbins=5
y_train_clipped = y_train.clip(0,1)
y_train_binned, histbins = pd.cut(y_train_clipped, np.linspace(0,1,nbins+1), labels=range(nbins), include_lowest=True, retbins=True)

print('bin edges:', histbins)

# calculate a weight for each bin, inversely proportional to the population in that bin

class_weights = {k:1/float(v) for k,v in Counter(y_train_binned).items()}

# increase the class 0 weight by multiplying by 1.5
# this empirically improved model accuracy during parameter optimization on the training data

class_weights[0] = class_weights[0] * 1.5

# scale weights to sum to 1

class_weights = {k:v/sum(class_weights.values()) for k,v in class_weights.items()}
class_weights

# generate a list mapping each element in y_train to its class weight

sample_weights = [class_weights[Y] for Y in y_train_binned]

# build and compile the CNN model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', input_shape=(4,26,2), data_format='channels_last'))
model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last'))
model.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.compile(loss='logcosh', metrics=['mse'], optimizer='adam')

# train the model for 8 epochs

model_history = model.fit(X_train,
                          y_train.ravel(),
                          sample_weight=np.array(sample_weights),
                          batch_size=32,
                          epochs=8,
                          validation_data=(X_val, y_val.ravel()))

# plot measured vs. predicted relative activity

fig,ax = plt.subplots(figsize=(5,5))
ax.scatter(model.predict(X_val), y_val, marker='.', alpha=.2)
ax.set_xlabel('predicted activity')
ax.set_ylabel('measured activity')
ax.set_title('performance on validation set');

print('r squared = %.3f'%pearsonr(y_val, model.predict(X_val).ravel())[0]**2)

# starting from the sgRNA and corresponding genomic sequences

                             #
sgrna  = 'TAGGTACTGAGCGCGCGAGCTGAGGA'
genome = 'TAGGTACTGAGCGCGCGAGGTGAGGA'
                             # -3 rC:dC

def get_predicted_activity(genome_seq, sgrna_seq, cnn_model):
    """
    takes 26-nt sgRNA and genome sequences, plus a trained model
    outputs the predicted relative activity of the sgRNA
    """
    X = np.stack((binarize_sequence(genome_seq),
                  binarize_sequence(sgrna_seq)), axis=2)[np.newaxis]

    return cnn_model.predict(X)[0][0]

print(get_predicted_activity(genome, sgrna, model))
