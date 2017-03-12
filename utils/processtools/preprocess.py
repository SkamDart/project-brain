import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def fileToList(filepath, label):
    """ Function to transform the image inside a .nii file to a list,  and add its label.
        The format is: <label><R channel pixel><G channel pixel><B channel pixel>.
        The parameters used to slice image is subject to change. """
    epi_img = nib.load(filepath)
    epi_img_data = epi_img.get_data()
    raw_img = 0.33*epi_img_data[75:107, 103:135, 95]
    raw_img = (np.array(raw_img)).flatten()
    return list([label]) + list(raw_img) + list(raw_img) + list(raw_img)

def generateBinary(outputlist, filename):
    """ Functino to generate a binary file from a list of data.
        The output file can be directly used as input for tensorflow cifar-10 module. """
    np.array(outputlist, np.uint8).tofile(filename)

def imageShape(filepath):
    """ Function to return the shape of image in a .nii file. """
    epi_img = nib.load(filepath)
    epi_img_data = epi_img.get_data()
    return epi_img_data.shape

def show_slices(slices):
    """ Function to display row of image slices. """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Center slices for EPI image")
    plt.show()
