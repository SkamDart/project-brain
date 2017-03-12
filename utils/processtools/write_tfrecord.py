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
