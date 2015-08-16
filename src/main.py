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

def load_data(subjs, series, seg_len=None, dataset="train"):
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
	if dataset == 'train':
		tests = range(1,9)
	else:
		tests = range(9,11)
	data_list = [
		'subj%d_series%d_data.csv'%(s, t)  
		for s in subjs for t in tests
		]
	label_list = [
		'subj%d_series%d_events.csv'%(s, t)
		for s in subjs for t in tests
		]
	
	data_segs = []
	for data_file in data_list:
		data = pd.read_csv(open(os.path.join('../data/%s/data' % dataset, data_file)))
		data = data.as_matrix()[:,1:]
		if seg_len is None:
			seg_len = data.shape[0]
		for i in range(0, data.shape[0] / seg_len):
			data_segs.append(data[i * seg_len:(i+1) * seg_len, :])

	n_segs = len(data_segs)
	raw_data = np.empty((n_segs, 1, seg_len, n_data_chn))
	for i in range(len(data_segs)):
		raw_data[i,:,:,:] = np.array(data_segs[i])

	label_segs = []
	raw_labels = None
	if dataset == "train":
		for label_file in label_list:
			label = pd.read_csv(open(os.path.join('../data/%s/label' % dataset, label_file)))
			label = label.as_matrix()[:,1:]
			for i in range(0, label.shape[0] / seg_len):
				label_segs.append(label[i * seg_len:(i+1) * seg_len, :])

		assert data.shape[0] == label.shape[0], 'Data and label length don\'t match'
		n_segs = len(data_segs)
		raw_labels = np.empty((n_segs, 1, seg_len, n_label_chn))

		for i in range(len(data_segs)):
			raw_labels[i,:,:,:] = np.array(label_segs[i])

		raw_labels = raw_labels.swapaxes(1, 3)

	return raw_data, raw_labels

def construct_model():
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

	return model

def train():
	
	def train_run(model, train_data, train_label):
		train_label = train_label[:,:,49:-1:2,:]
		train_label = train_label[:,:,9:-1:2,:]
		train_label = train_label[:,:,2:,:]

		model.fit(
			train_data, train_label, 
			batch_size=100, nb_epoch=20, 
			shuffle=False, verbose=1,
			validation_split=0.2)

	seg_len = 2000
	model = construct_model()

	for run in range(10):
		print "Run #%d" % run
		for i in range(1,13,2):
			data, label = load_data(
				subjs=range(i,i+2), series=range(1,9), seg_len=seg_len
				)
		model = train_run(model, data, label)
		model.save_weights('./model/train_run_%d.model' % run)

	return model

def predict(test_data):
	fn = './model/train_9_epochs.model'
	model = construct_model()

	model.load_weights(fn)
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)

	rst = np.empty((test_data.shape[2], n_label_chn))

	for s in range(22):
		for c in range(n_label_chn):
			pred = np.round(model.predict(test_data[s:s+1,:,:,:]))
			tmp = pred[0,c:c+1,:,0]
			tmp = np.round([0] * 9 + list(np.repeat(tmp, 2)) + [0])
			tmp = np.round([0] * 49 + list(np.repeat(tmp, 2)) + [0])
			rst[0:len(tmp),c] = tmp

	rst[np.isnan(rst)] = 0

	return rst


if __name__ == "__main__":
	model = train()
	seg_len = 2000
	test_data, _ = load_data(
		subjs=range(1, 13), series=range(9, 11), dataset="test"
		)
	predict()