preprocess.py: utility functions to transform NIfTI images to binary files that is ready to serve as input for tensorflow cifar-10 model.
Reference: the official manual of NiBabel library.

read_tfrecord.py: read a TfRecord file and display its image in a batch.
write_tfrecord.py: take image files and generate TfRecord file.
Reference: https://github.com/warmspringwinds/tensorflow_notes/blob/master/tfrecords_guide.ipynb