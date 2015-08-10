import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

n_data_chn = 32
n_label_chn = 6

def load_data(seg_len, lite=False):
	# DATA
	# id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10
	# subj10_series1_0,
	# -304,-156,-411,-640,-505,-603,-451,104,-344,-784,
	# -387,-308,35,-730,-161,-345,-342,-928,-536,-371,
	# -660,-107,-197,-597,-242,-472,-56,-338,-335,-518,-371,-177
	# 12 chn

	# LABEL
	# id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased
	# subj1_series1_0,	0,0,1,0,0,0
	# subj1_series1_1,	0,0,2,0,9,0
	if not lite:
		data_list = os.listdir('../data/train/data')
		label_list = os.listdir('../data/train/label')
	else:
		data_list = os.listdir('../data_lite/train/data')
		label_list = os.listdir('../data_lite/train/label')

	data_segs = []
	for data_file in data_list:
		data = pd.read_csv(open(os.path.join('../data/train/data', data_file)))
		data = data.as_matrix()[:,1:]
		for i in range(0, data.shape[0] / seg_len):
			data_segs.append(data[i * seg_len:(i+1) * seg_len, :])
	label_segs = []
	for label_file in label_list:
		label = pd.read_csv(open(os.path.join('../data/train/label', label_file)))
		label = label.as_matrix()[:,1:]
		for i in range(0, label.shape[0] / seg_len):
			label_segs.append(label[i * seg_len:(i+1) * seg_len, :])

	n_segs = len(data_segs)
	raw_data = np.empty((n_segs, 1, seg_len, n_data_chn))
	raw_labels = np.empty((n_segs, 1, seg_len, n_label_chn))

	for i in range(len(data_segs)):
		raw_data[i,:,:,:] = np.array(data_segs[i])
		raw_labels[i,:,:,:] = np.array(label_segs[i])

	raw_labels = raw_labels.swapaxes(1, 3)

	return raw_data, raw_labels

def evaluate(pred, label):
	return np.sqrt(np.mean(np.sqr(pred - label)))

def train(train_data, train_label):
	model = Sequential()
	model.add(Convolution2D(
		20, 1, 50, 1, init='uniform', activation='tanh'))
	model.add(MaxPooling2D(poolsize=(2, 1)))
	model.add(Convolution2D(
		10, 20, 10, 1, init='uniform', activation='tanh'))
	model.add(MaxPooling2D(poolsize=(2, 1)))
	model.add(Convolution2D(
		n_label_chn, 10, 3, 32))

	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)

	train_label = train_label[:,:,49:-1:2,:]
	train_label = train_label[:,:,9:-1:2,:]
	train_label = train_label[:,:,2:,:]

	for i in range(30):
		model.fit(
			train_data, train_label, 
			batch_size=100, nb_epoch=1, 
			shuffle=False, verbose=1,
			validation_split=0.2)
		print "Run #%d" % i
		model.save_weights('./model/train_%d_epochs.model' % (i*20))


if __name__ == "__main__":
	seg_len = 2000
	data, label = load_data(seg_len, lite=True)
	train(data, label)