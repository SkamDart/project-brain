"""
	Writes image to TF-Record type
	Ref: https://github.com/warmspringwinds/tensorflow_notes/blob/master/tfrecords_guide.ipynb
	Modified: Junjie
"""

from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

def im_read(img_path, annotation_path):
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))

    height = img.shape[0]
    width = img.shape[1]
    
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()

    return (img_raw, annotation_raw, height, width)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def writeToFile(filename_pairs, tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img_path, annotation_path in filename_pairs:
        (img_raw, annotation_raw, height, width) = im_read(img_path, annotation_path)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def testFunction():
    # Get some image/annotation pairs for example 
    filename_pairs = [('../small_testdata/cat.jpg','../small_testdata/cat_annotation.png')]
    # Important: We are using PIL to read .png files later.
    # This was done on purpose to read indexed png files
    # in a special way -- only indexes and not map the indexes
    # to actual rgb values. This is specific to PASCAL VOC
    # dataset data. If you don't want thit type of behaviour
    # consider using skimage.io.imread()
    tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

    """generate output file is done here, following codes just check if we can
       retrive original image from our TfRecord file"""
    writeToFile(filename_pairs, tfrecords_filename)

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
