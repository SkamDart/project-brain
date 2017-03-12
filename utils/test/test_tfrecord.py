from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
from ..processtools import read_tfrecord, write_tfrecord

""" Test for tfrecord write and read.
    Usage: specify a main function here, and run 'python -m utils.test.test_tfrecord' at root folder"""

def testWrite():
    # Get some image/annotation pairs for example 
    filename_pairs = [('small_TestData/cat.jpg','small_TestData/cat_annotation.png')]
    # Important: We are using PIL to read .png files later.
    # This was done on purpose to read indexed png files
    # in a special way -- only indexes and not map the indexes
    # to actual rgb values. This is specific to PASCAL VOC
    # dataset data. If you don't want thit type of behaviour
    # consider using skimage.io.imread()
    tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

    """generate output file is done here, following codes just check if we can
       retrive original image from our TfRecord file"""
    write_tfrecord.writeToFile(filename_pairs, tfrecords_filename)

    # Let's collect the real images to later on compare
    # to the reconstructed ones
    original_images = []
    reconstructed_images = []

    for img_path, annotation_path in filename_pairs:
        img = np.array(Image.open(img_path))
        annotation = np.array(Image.open(annotation_path))
        # Put in the original images into array
        # Just for future check for correctness
        original_images.append((img, annotation))

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])
        
        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])
        
        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])
        
        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])
        
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
        
        # Annotations don't have depth (3rd dimension)
        reconstructed_annotation = annotation_1d.reshape((height, width, -1))
        reconstructed_images.append((reconstructed_img, reconstructed_annotation))

    # Let's check if the reconstructed images match
    # the original images
    for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
        img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)

        #Should print True if succeed
        print(np.allclose(*img_pair_to_compare))
        print(np.allclose(*annotation_pair_to_compare))


def testReadAndDisplay():
    tfrecords_filename = 'pascal_voc_segmentation.tfrecords'
    # Even when reading in multiple threads, share the filename queue.
    image, annotation = read_tfrecord.read_and_decode(tfrecords_filename)
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Let's read off 3 batches just for example
        for i in xrange(1):
            img, anno = sess.run([image, annotation])
            print(img[0, :, :, :].shape)
            print('current batch')
            # We selected the batch size of two
            # So we should get two image pairs in each batch
            # Let's make sure it is random
            io.imshow(img[0, :, :, :])
            io.show()

            io.imshow(anno[0, :, :, 0])
            io.show()
            
            io.imshow(img[1, :, :, :])
            io.show()

            io.imshow(anno[1, :, :, 0])
            io.show()

        coord.request_stop()
        coord.join(threads)
