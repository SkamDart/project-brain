import nibabel as nib
from ..processtools import *

""" Test for generate binary file from NIfTI files.
    Usage: specify a main function here, and run 'python -m utils.test.test_binary' at root folder
    Under to the HCP Data Use Terms, you could download and use HCP data from:
    https://store.humanconnectome.org/data/data-use-terms/"""

def test_binary(data_dirs0, data_dirs1):
    """ Function to test generating binary file using our sample data. """
    outarr = None

    for filepath in data_dirs0:
        if outarr == None:
            outarr = preprocess.fileToList(filepath, 0)
        else:
            outarr += preprocess.fileToList(filepath, 0)

    for filepath in data_dirs1:
        if outarr == None:
            outarr = preprocess.fileToList(filepath, 1)
        else:
            outarr += preprocess.fileToList(filepath, 1)

    preprocess.generateBinary(outarr, 'out.bin')

def test_display_image(data_dir):
    """ Function to test display image from .nii files. """
    epi_img = nib.load(data_dir)
    epi_img_data = epi_img.get_data()
    slice_1 = epi_img_data[73:106, 93:126, 95]
    slice_2 = epi_img_data[:, :, 95]
    preprocess.show_slices([slice_1, slice_2])
