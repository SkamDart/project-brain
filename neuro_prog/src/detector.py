import tensorflow as tf
import numpy as np
import cPickle

class Detector():
    def __init__(self, weight_file_path, n_labels):
        self.image_mean = [103.939, 116.779, 123.68]
        self.n_labels = n_labels

        # with open(weight_file_path) as f:
        #     self.pretrained_weights = cPickle.load(f)

    def get_weight( self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias( self, layer_name ):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    def get_conv_weight( self, name ):
        f = self.get_weight( name )
        return f.transpose(( 2,3,1,0 ))

    def conv_layer( self, bottom, name ):
        with tf.variable_scope(name) as scope:

            w = self.get_conv_weight(name)
            b = self.get_bias(name)

            conv_weights = tf.get_variable(
                    "W",
                    shape=w.shape,
                    initializer=tf.constant_initializer(w)
                    )
            conv_biases = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b)
                    )

            conv = tf.nn.conv2d( bottom, conv_weights, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add( conv, conv_biases )
            relu = tf.nn.relu( bias, name=name )

        return relu

    def new_conv_layer( self, bottom, filter_shape, pad, name ):
        with tf.variable_scope( name ) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv3d( bottom, w, [1,1,1,1,1], padding=pad)
            bias = tf.nn.bias_add(conv, b)
            relu = tf.nn.relu( bias, name=name )

        return relu

    def fc_layer(self, bottom, name, create=False):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape(bottom, [-1, dim])

        cw = self.get_weight(name)
        b = self.get_bias(name)

        if name == "fc6":
            cw = cw.reshape((4096, 512, 7,7))
            cw = cw.transpose((2,3,1,0))
            cw = cw.reshape((25088,4096))
        else:
            cw = cw.transpose((1,0))

        with tf.variable_scope(name) as scope:
            cw = tf.get_variable(
                    "W",
                    shape=cw.shape,
                    initializer=tf.constant_initializer(cw))
            b = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b))

            fc = tf.nn.bias_add( tf.matmul( x, cw ), b, name=scope)

        return fc

    def new_fc_layer( self, bottom, input_size, output_size, name ):
        shape = bottom.get_shape().to_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])

        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b, name=scope)

        return fc

    def inference( self, input_images):
        #input: [53x30x36x30x1]
        relu1_1 = self.new_conv_layer( input_images, [3,3,3,1,16], 'SAME', "conv1_1" )
        #relu1_1: [53x30x36x30x16]
        relu1_2 = self.new_conv_layer( relu1_1, [3,3,3,16,32], 'SAME', "conv1_2" )
        #relu1_2: [53x30x36x30x32]
        pool1 = tf.nn.max_pool3d(relu1_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                         padding='SAME', name='pool1')
        #pool1: [53x15x18x15x32]
        relu2_1 = self.new_conv_layer(pool1, [3,3,3,32,64], 'SAME', "conv2_1")
        #relu2_1: [53x15x18x15x64]
        relu2_2 = self.new_conv_layer(relu2_1, [3,3,3,64,64], 'SAME', "conv2_2")
        #relu2_2: [53x15x18x15x64]
        pool2 = tf.nn.max_pool3d(relu2_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name='pool2')
        #pool2: [53x8x9x8x64]

        # relu3_1 = self.new_conv_layer( pool2, "conv3_1")
        # relu3_2 = self.new_conv_layer( relu3_1, "conv3_2")
        # relu3_3 = self.new_conv_layer( relu3_2, "conv3_3")
        # pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                        padding='SAME', name='pool3')

        # relu4_1 = self.new_conv_layer( pool3, "conv4_1")
        # relu4_2 = self.new_conv_layer( relu4_1, "conv4_2")
        # relu4_3 = self.new_conv_layer( relu4_2, "conv4_3")
        # pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                        padding='SAME', name='pool4')

        # relu5_1 = self.new_conv_layer( pool4, "conv5_1")
        # relu5_2 = self.new_conv_layer( relu5_1, "conv5_2")
        # relu5_3 = self.new_conv_layer( relu5_2, "conv5_3")

        conv6 = self.new_conv_layer( pool2, [3,3,3,64,128], 'SAME', "conv6")
        #conv6: [53x8x9x8x128]
        gap = tf.reduce_mean( conv6, [1,2,3] )
        #gap: [53x128]

        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable(
                    "W",
                    shape=[128, self.n_labels],
                    initializer=tf.random_normal_initializer(0., 0.01))

        output = tf.matmul( gap, gap_w)
        #output: [53 x n_labels]
        return pool1, pool2, conv6, gap, output

    def get_classmap(self, label, conv6):
        conv6_stack = tf.reduce_mean( conv6, [1] )
        #conv6_stack: [53x9x8x128]
        conv6_resized = tf.image.resize_bilinear( conv6_stack, [36, 30] )
        #conv6_resized: [53x36x30x128]
        with tf.variable_scope("GAP", reuse=True):
            label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
            label_w = tf.reshape( label_w, [-1, 128, 1] ) # [53, 128, 1]

        conv6_resized = tf.reshape(conv6_resized, [-1, 36*30, 128]) # [53, 36*30, 128]

        classmap = tf.matmul( conv6_resized, label_w )
        #classmap: [53, 36*30, 1]
        classmap = tf.reshape( classmap, [-1, 36, 30] )
        return classmap






