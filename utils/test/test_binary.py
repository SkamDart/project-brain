import nibabel as nib
from ..processtools import *

""" Test for generate binary file from NIfTI files.
    Usage: specify a main function here, and run 'python -m utils.test.test_binary' at root folder
    Under to the HCP Data Use Terms, you could download and use HCP data from:
    https://store.humanconnectome.org/data/data-use-terms/"""

def testBinary():
    """ Function to test generating binary file using our sample data. """
    outarr = None
    for i in range(5):
        filepath = 'small_TestData/BrainHack-selected/LS400'+str(i+1)+'_T1_std.nii.gz'
        if i == 0:
            outarr = preprocess.fileToList(filepath, 0)
        else:
            outarr += preprocess.fileToList(filepath, 0)

    for i in range(4):
        filepath = 'small_TestData/BrainHack-selected/LS600'+str(i+1)+'_T1_std.nii.gz'
        outarr += preprocess.fileToList(filepath, 0)

    preprocess.generateBinary(outarr, 'out.bin')

def testDisplayImage():
    """ Function to test display image from .nii files. """
    epi_img = nib.load('small_TestData/BrainHack-selected/LS600'+str(2)+'_T1_std.nii.gz')
    epi_img_data = epi_img.get_data()
    epi_img1 = nib.load('small_TestData/BrainHack-selected/LS400'+str(5)+'_T1_std.nii.gz')
    epi_img_data1 = epi_img1.get_data()
    slice_2 = epi_img_data[73:106, 93:126, 95]
    slice_5 = epi_img_data1[73:106, 93:126, 95]
    preprocess.show_slices([slice_2, slice_5])
