import tensorflow as tf
import numpy as np
import pandas as pd

from detector import Detector
from util import load_image
import os


weight_path = '/data/junjiew2/neuro-prog/data/caffe_layers_value.pickle'
model_path = '/data/junjiew2/neuro-prog/models/'
pretrained_model_path = None #'../models/caltech256/model-0'
n_epochs = 10000
init_learning_rate = 0.01
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 53


caltech_path = '/data/junjiew2/neuro-prog/data/caltech'
trainset_path = '/data/junjiew2/neuro-prog/data/caltech/train.pickle'
testset_path = '/data/junjiew2/neuro-prog/data/caltech/test.pickle'
label_dict_path = '/data/junjiew2/neuro-prog/data/caltech/label_dict.pickle'

filepaths = {0: '/data/junjiew2/neuro-prog/data/2BK-0BK_30-36-30.npz', 1: '/data/junjiew2/neuro-prog/data/REL-MATCH_30-36-30.npz'}

if not os.path.exists( trainset_path ):
    if not os.path.exists( caltech_path ):
        os.makedirs( caltech_path )

    labels, label_names = zip([0,'class0'], [1,'class1'])

    label_dict = pd.Series( labels, index=label_names )

    n_labels = len( label_dict )

    image_paths_per_label = [range(1000, 1494), range(2000, 2481)]
    image_paths_train = np.hstack(map(lambda one_class: one_class[:-10], image_paths_per_label))
    #index for testset: [484 485 486 487 488 489 490 491 492 493 471 472 473 474 475 476 477 478 479 480]
    image_paths_test = np.hstack(map(lambda one_class: one_class[-10:], image_paths_per_label))

    trainset = pd.DataFrame({'image_index': image_paths_train})
    testset  = pd.DataFrame({'image_index': image_paths_test })

    trainset['label'] = trainset['image_index'].map(lambda x: int(str(x)[0]) - 1)
    trainset['true_index'] = trainset['image_index'].map(lambda x: int(str(x)[1:]))
    trainset['image_path'] = trainset['label'].map(lambda x: filepaths[x])

    testset['label'] = testset['image_index'].map(lambda x: int(str(x)[0]) - 1)
    testset['true_index'] = testset['image_index'].map(lambda x: int(str(x)[1:]))
    testset['image_path'] = testset['label'].map(lambda x: filepaths[x])

    label_dict.to_pickle(label_dict_path)
    trainset.to_pickle(trainset_path)
    testset.to_pickle(testset_path)
else:
    trainset = pd.read_pickle( trainset_path )
    testset  = pd.read_pickle( testset_path )
    label_dict = pd.read_pickle( label_dict_path )
    n_labels = len(label_dict)

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 30, 36, 30, 1], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(weight_path, n_labels)

p1,p2, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels_tf, logits=output ))

weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
loss_tf += weight_decay


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = .9
sess_config.intra_op_parallelism_threads = 2

sess = tf.InteractiveSession(config = sess_config)
saver = tf.train.Saver( max_to_keep=50 )

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.global_variables_initializer().run()

if pretrained_model_path:
    print "Pretrained"
    saver.restore(sess, pretrained_model_path)

testset.index  = range( len(testset) )
#testset = testset.ix[np.random.permutation( len(testset) )]#[:1000]
#trainset2 = testset[1000:]
#testset = testset[:1000]

#trainset = pd.concat( [trainset, trainset2] )
# We lack the number of training set. Let's use some of the test images

#f_log = open('../results/log.caltech256.txt', 'w')

iterations = 0
loss_list = []
for epoch in range(n_epochs):

    trainset.index = range( len(trainset) )
    trainset = trainset.ix[ np.random.permutation( len(trainset) )]

    for start, end in zip(
        range( 0, len(trainset)-1+batch_size, batch_size),
        range(batch_size, len(trainset)-1+batch_size, batch_size)):

        current_data = trainset[start:end].values
        current_images = np.array(map(lambda x: np.load(x[3])['voxeldata'][:, :, :, x[2]], current_data))
        #making Tensor in shape [batch_size, depth, height, width]
        current_images =  np.swapaxes(current_images, 1, 3)
        current_images = np.reshape(current_images, (batch_size,30,36,30,1))
        current_labels = trainset[start:end]['label'].values

        _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

        loss_list.append( loss_val )

        iterations += 1
        if iterations % 5 == 0:
            print "======================================"
            print "Epoch", epoch, "Iteration", iterations
            print "Processed", start, '/', len(trainset)

            label_predictions = output_val.argmax(axis=1)
            acc = (label_predictions == current_labels).sum()

            print "Accuracy:", acc, '/', len(current_labels)
            print "Training Loss:", np.mean(loss_list)
            print "\n"
            loss_list = []

    n_correct = 0
    n_data = 0
    batch_size_test = 10
    for start, end in zip(
            range(0, len(testset)+batch_size_test, batch_size_test),
            range(batch_size_test, len(testset)+batch_size_test, batch_size_test)
            ):
        current_data = testset[start:end].values
        current_images = np.array(map(lambda x: np.load(x[3])['voxeldata'][:, :, :, x[2]], current_data))
        current_images =  np.swapaxes(current_images, 1, 3)
        current_images = np.reshape(current_images, (batch_size_test,30,36,30,1))
        current_labels = testset[start:end]['label'].values

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == current_labels).sum()

        n_correct += acc
        n_data += len(current_data)

    acc_all = n_correct / float(n_data)
    #f_log.write('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print 'epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)

    init_learning_rate *= 0.99




