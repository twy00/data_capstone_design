from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
import numpy.random as npr
from tensorflow import nn
from trainOps import TrainOps
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import lrelu
import numpy as np
from tensorflow.contrib.layers import flatten
from densenet import DenseNet
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

def make_conv2d(inputs, filters, ksize=5, strides=2, padding='SAME', activation=lrelu):
	return tf.layers.conv2d(inputs, filters=filters, kernel_size=ksize, strides=strides, padding=padding,
							activation=activation)

def make_fc(inputs, output_size, stddev=0.1, activation=tf.nn.relu):
	layer = tf.layers.dense(inputs, output_size, activation=activation,
							kernel_initializer=tf.initializers.truncated_normal(stddev=stddev))

	return layer

class GAN(object):
	def __init__(self, mode,classifier,generator, gan_type, loss_function, num_labels, hidden_repr_size, noise_dim,
				extractor_learning_rate, generator_learning_rate, discriminator_learning_rate, classifier_learning_rate, dataset, batch_size):
		self.mode = mode
		# self.extractor_learning_rate = extractor_learning_rate
		self.generator_learning_rate = generator_learning_rate
		self.discriminator_learning_rate = discriminator_learning_rate
		self.classifier_learning_rate = classifier_learning_rate
		self.generator = generator
		self.hidden_repr_size = int(hidden_repr_size)
		self.classifier = classifier
		
		self.num_labels = num_labels
		self.gan_type = gan_type
		self.loss_function = loss_function
		self.noise_dim = noise_dim
		self.batch_size = batch_size

		self.dataset = dataset

		if self.dataset == 'cifar10':
			self.feature_height = 32
			self.feature_width = 32
			self.feature_depth = 64
		elif self.dataset == 'fashion_mnist':
			self.feature_height = 14
			self.feature_width = 14
			self.feature_depth = 6

	def set_image_dimension(self, image_width, image_height, image_depth):
		self.image_size_width = image_width
		self.image_size_height = image_height
		self.image_depth = image_depth

	def feature_extractor(self, image, reuse=False, return_output = False, current_op = '', keep_prob=0.5, batch_prob=True):

		with tf.variable_scope('feature_extractor', reuse=reuse) as vs:
				if reuse: 
					vs.reuse_variables()
				
				print("CLASSIFIER")
				print(image.shape)

				print("block 1")
				#block1
				net = tf.layers.conv2d(image, 64, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 64, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.nn.dropout(net, keep_prob)
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				print(net.shape)
				print("blok2")
				#block2
				net = tf.layers.conv2d(net, 128, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 128, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.nn.dropout(net, keep_prob)
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				print(net.shape)
				
				#block3
				print("block3")
				net = tf.layers.conv2d(net, 128, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 256, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 256, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.nn.dropout(net, keep_prob)
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				print(net.shape)
				
				#block4
				print("block4")
				net = tf.layers.conv2d(net, 256, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				print(net.shape)
				net = tf.nn.dropout(net, keep_prob)
				net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				print(net.shape)
				
				self.feature_height, self.feature_width, self.feature_depth = net.shape[1], net.shape[2], net.shape[3]
				
				if (self.mode == 'train_feature_extractor') or( self.mode == 'all' and current_op == 'train_feature_extractor') or return_output: 
					

					#block5
					print("block5")
					net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
					net = tf.layers.batch_normalization(net, training=batch_prob)
					net = tf.nn.leaky_relu(net)
					print(net.shape)
					net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
					net = tf.layers.batch_normalization(net, training=batch_prob)
					net = tf.nn.leaky_relu(net)
					print(net.shape)
					net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
					net = tf.layers.batch_normalization(net, training=batch_prob)
					net = tf.nn.leaky_relu(net)
					print(net.shape)
					net = tf.nn.dropout(net, keep_prob)
					net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
					print(net.shape)

					#fully connected layer
					flatten = tf.reshape(net, shape=[-1, net.shape[1]*net.shape[2]*net.shape[3]])

					net = tf.layers.dense(flatten, 4096, kernel_initializer=tf.contrib.layers.xavier_initializer())
					net = tf.layers.batch_normalization(net, training=batch_prob)
					net = tf.nn.leaky_relu(net)
					net = tf.nn.dropout(net, keep_prob)
					print(net.shape)

					net = tf.layers.dense(net, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
					net = tf.layers.batch_normalization(net, training=batch_prob)
					net = tf.nn.leaky_relu(net)
					net = tf.nn.dropout(net, keep_prob)
					print(net.shape)
					
					dense2 = tf.layers.dense(net, self.num_labels, kernel_initializer=tf.contrib.layers.xavier_initializer())
					dense2 = tf.layers.batch_normalization(dense2, training=batch_prob)
					dense2 = tf.nn.leaky_relu(dense2)
					print(dense2.shape)

					return dense2
				else:
					return net

	def feature_generator(self, noise, labels, reuse = False, keep_prob=0.5, batch_prob=True):
		
		print("GENERATOR")

		
		if self.gan_type == 'acgan':
			inputs = noise
			
		if self.generator == 'wgan':
			with tf.variable_scope('feature_generator', reuse=reuse):
				
				noise = tf.reshape(noise, [-1, 1, 1, self.noise_dim])
				labels = tf.reshape(labels, [-1, 1, 1, self.num_labels])

				try:
					inputs = tf.concat([noise, tf.cast(labels,tf.float32)], 3)
				except:
					inputs = tf.concat(3,[noise, tf.cast(labels,tf.float32)])

				dense = tf.layers.dense(inputs, 1*1*2048, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

				net = tf.reshape(dense, [-1, 1, 1, 2048])

				if not reuse:	print(net.shape)

				conv1 = tf.layers.conv2d_transpose(net, 2048, 5, 5, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				conv1 = tf.layers.batch_normalization(conv1, -1)
				conv1 = tf.nn.leaky_relu(conv1)
				if not reuse:	print(conv1.shape)

				conv2 = tf.layers.conv2d_transpose(conv1, 1024, 5, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				conv2 = tf.layers.batch_normalization(conv2, -1)
				conv2 = tf.nn.leaky_relu(conv2)
				if not reuse:	print(conv2.shape)

				conv3 = tf.layers.conv2d_transpose(conv2, 512, 5, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				conv3 = tf.layers.batch_normalization(conv3, -1)
				conv3 = tf.nn.leaky_relu(conv3)
				if not reuse:	print(conv3.shape)
				
				conv4 = tf.layers.conv2d_transpose(conv3, 512, 5, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				conv4 = tf.layers.batch_normalization(conv4, -1)
				conv4 = tf.nn.leaky_relu(conv4)

				conv5 = tf.layers.conv2d_transpose(conv4, 512, 5, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				conv5 = tf.layers.batch_normalization(conv5, -1)
				conv4 = tf.nn.tanh(conv5)

				if not reuse:	print(conv5.shape)
				
				return conv5
				
	def feature_discriminator(self, features, reuse=False):

		if self.generator == 'wgan':
			with tf.variable_scope('feature_discriminator') as vs:
				if reuse:
					vs.reuse_variables()
				if not reuse:	print("DISCRIMINATOR")
				if not reuse:	print(features.shape)
				
				net = tf.layers.conv2d(features, 512, 5, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				net = tf.layers.batch_normalization(net, -1)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)
				
				net = tf.layers.conv2d(net, 512, 5, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				net = tf.layers.batch_normalization(net, -1)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)
				
				net = tf.layers.conv2d(net, 1024, 5, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),padding='SAME')
				net = tf.layers.batch_normalization(net, -1)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)

				net = tf.reshape(net, shape=[-1, 2*2*1024])

				net = tf.layers.dense(net, 1024)
				net = tf.layers.batch_normalization(net, -1)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)

				net = tf.layers.dense(net, 64)
				net = tf.layers.batch_normalization(net, -1)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)

				net = tf.layers.dense(net, 1)
				net = tf.layers.batch_normalization(net, -1)
				net = tf.math.sigmoid(net)
				if not reuse:	print(net.shape)

				return net

	def feature_classifier(self, features, reuse = False, keep_prob=0.5, batch_prob=True):

		with tf.variable_scope('feature_classifier', reuse=reuse):

			if self.classifier == 'vggnet16':
				
				if not reuse:	print("CLASSIFIER")
				
				net = tf.layers.conv2d(features, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)
				
				net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)
				
				net = tf.layers.conv2d(net, 512, [3,3], 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				if not reuse:	print(net.shape)
				
				net = tf.nn.dropout(net, keep_prob)
				net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				if not reuse:	print(net.shape)

				#fully connected layer
				flatten = tf.reshape(net, shape=[-1, net.shape[1]*net.shape[2]*net.shape[3]])

				net = tf.layers.dense(flatten, 4096, kernel_initializer=tf.contrib.layers.xavier_initializer())
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				net = tf.nn.dropout(net, keep_prob)
				if not reuse:	print(net.shape)

				net = tf.layers.dense(net, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
				net = tf.layers.batch_normalization(net, training=batch_prob)
				net = tf.nn.leaky_relu(net)
				net = tf.nn.dropout(net, keep_prob)
				if not reuse:	print(net.shape)
				
				dense2 = tf.layers.dense(net, self.num_labels, kernel_initializer=tf.contrib.layers.xavier_initializer())
				dense2 = tf.layers.batch_normalization(dense2, training=batch_prob)
				dense2 = tf.nn.leaky_relu(dense2)
				if not reuse:	print(dense2.shape)

				return dense2

	def build_model(self, current_op = ''):
		
		print("build model..")

		if (self.mode == 'all' and current_op == 'train_feature_extractor')  or self.mode == 'train_feature_extractor' or current_op == "train_feature_extractor":
			self.images = tf.placeholder(tf.float32, (None, self.image_size_width, self.image_size_height, self.image_depth))
			self.extractor_learning_rate = tf.placeholder(tf.float32, (None))
			self.keep_prob = tf.placeholder(tf.float32, (None))
			self.batch_prob = tf.placeholder(tf.bool)
			self.labels = tf.placeholder(tf.int64, (None))
			one_hot_y = tf.one_hot(self.labels, self.num_labels)
			self.logits= self.feature_extractor(self.images, current_op = current_op, reuse=False, keep_prob=self.keep_prob, batch_prob=self.batch_prob)
			
			self.pred = tf.argmax(self.logits, 1)
			self.correct_pred = tf.equal(self.pred, self.labels)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

			self.loss = tf.reduce_mean(slim.losses.softmax_cross_entropy(self.logits, one_hot_y))

			t_vars = tf.trainable_variables()
			e_vars = [var for var in t_vars if 'feature_extractor' in var.name]

			self.train_op = tf.train.AdamOptimizer(learning_rate=self.extractor_learning_rate, beta1=0.5, beta2=0.999).minimize(self.loss,var_list=e_vars)

			# summary
			loss_summary = tf.summary.scalar('classification_loss', self.loss)
			src_accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
			self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary])

		elif (self.mode == 'all' and current_op=='train_feature_generator')  or self.mode =='train_feature_generator':
			scale = 10.0
		
			self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
			self.labels = tf.placeholder(tf.int64, (None))
			self.extractor_learning_rate = tf.placeholder(tf.float32, (None))
			self.keep_prob = tf.placeholder(tf.float32, (None))
			self.batch_prob = tf.placeholder(tf.bool)
			self.target_real = tf.placeholder(tf.float32, (None,1))
			self.target_fake = tf.placeholder(tf.float32, (None,1))

			
			self.real_features = self.images = tf.placeholder(tf.float32, [None, 10, 10, 512], 'features')
			
			self.gen_features = self.feature_generator(self.noise, self.labels, keep_prob=self.keep_prob, batch_prob=self.batch_prob)

			self.logits_real = tf.cast(self.feature_discriminator(self.real_features), tf.float32)
			self.logits_fake = tf.cast(self.feature_discriminator(self.gen_features, reuse=True), tf.float32)

			self.c_logits_real = self.feature_classifier(self.real_features)
			self.c_logits_fake = self.feature_classifier(self.gen_features, reuse=True)

			self.pred = tf.argmax(self.c_logits_real, 1)
			self.label_logits = tf.argmax(self.labels, 1)
			self.correct_pred = tf.equal(self.pred, self.label_logits)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			
			self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_fake, labels=self.target_fake))
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_real, labels=self.target_real))

			# Gradient penalty
			epsilon = tf.random_uniform([], 0.0, 1.0)
			differences = self.gen_features - self.real_features
			interpolates = self.real_features + (epsilon*differences)
			gradients = tf.gradients(self.feature_discriminator(interpolates, reuse=True),[interpolates])[0] + 1e-16
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
			gradient_panelty = tf.reduce_mean((slopes-1.)**2)
			
			self.d_loss += 10*gradient_panelty

			self.c_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.c_logits_fake, labels=self.labels))
			self.c_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.c_logits_real, labels=self.labels))
			self.c_loss = self.c_loss_fake * 0.7 + self.c_loss_real * 0.3
			
			t_vars = tf.trainable_variables()
			d_vars = [var for var in t_vars if 'feature_discriminator' in var.name]
			g_vars = [var for var in t_vars if 'feature_generator' in var.name]
			c_vars = [var for var in t_vars if 'feature_classifier' in var.name]
			
			self.g_adam = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate).minimize(self.g_loss)
			self.d_adam = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate).minimize(self.d_loss)
			self.c_adam_fake = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, beta1=0.5, beta2=0.9).minimize(self.c_loss_fake, var_list=g_vars)
			self.c_adam_real = tf.train.AdamOptimizer(learning_rate=self.classifier_learning_rate, beta1=0.5, beta2=0.9).minimize(self.c_loss_real, var_list=c_vars)
			self.c_adam = tf.train.AdamOptimizer(learning_rate=self.classifier_learning_rate, beta1=0.5, beta2=0.9).minimize(self.c_loss, var_list=c_vars)

		print("model built")
