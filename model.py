import tflearn
import numpy as np
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
#print('abuuuuuuuuuuuuuuuuuu')
import matplotlib.pyplot as plt
#print('babuuuuuuuuuuuuuuu')
import tensorflow as tf
tf.reset_default_graph()

train_data=np.load('train_data.npy', encoding='latin1')
train=train_data[:-500]
test=train_data[-500:]

x=np.array([i[0] for i in train]).reshape(-1,50,50,1)
y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,50,50,1)
test_y=[i[1] for i in test]






convnet = input_data(shape=[None, 50, 50, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')


if os.path.exists('catdog.meta'):
	model.load('catdog')
	print('loaded')

#model.fit({'input': x}, {'targets': y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
#    snapshot_step=500, show_metric=True, run_id='catdog')

model.save('catdog')






test_data=np.load('test_data.npy', encoding='latin1')


fig=plt.figure()

for num,data in enumerate(test_data[:12]):
	img_num=data[1]
	img_data=data[0]

	y=fig.add_subplot(3,4,num+1)
	orig=img_data
	data=img_data.reshape(50,50,1)

	model_out=model.predict([data])[0]

	if np.argmax(model_out)==1: label='Dog'
	else: label='cat'

	y.imshow(orig,cmap='gray')
	plt.title(label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
#plt.show()


with open('submission.csv','w') as f:
	f.write('id,label\n')

with open('submission.csv','a') as f:
	for data in test_data:
		img_num=data[1]
		img_data=data[0]
		data=img_data.reshape(50,50,1)
		model_out=model.predict([data])[0]
		f.write('{},{}\n'.format(img_num,model_out[1]))

