import os
import nibabel as nib
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

### random rotation
def random_rotate3D(image_path, out_path, min_angle, max_angle):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    img_numpy = image_array.astype('float32')
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle+1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    img_rot = ndimage.rotate(img_numpy, angle, axes=axes)
    img_rot = nib.Nifti1Image(img_rot, nib.load(image_path).affine)

    nib.save(img_rot, out_path)

### Elestic transformation
def elastic_transform_3D(image_path, out_path, alpha=4, sigma=35, bg_val=0.1):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    img_numpy = image_array.astype('float32')
    assert img_numpy.ndim == 3
    shape = img_numpy.shape
    # Define coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Initialize interpolators
    im_intrps = RegularGridInterpolator(coords, img_numpy,
                                        method="linear",
                                        bounds_error=False,
                                        fill_value=bg_val)
    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                        mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                        mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                        mode="constant", cval=0.) * alpha
    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    indices = np.reshape(x + dx, (-1, 1)), \
        np.reshape(y + dy, (-1, 1)), \
        np.reshape(z + dz, (-1, 1))
    # Interpolate 3D image image
    image = im_intrps(indices).reshape(shape)
    image = nib.Nifti1Image(image, nib.load(image_path).affine)
    nib.save(image, out_path)

### path definition
data_dir = r"D:\CTdata\MaskCrop\bg\A\Train"
subtype1 = ['ChRCC', 'pRCC']

### for rotate of ChRCC, and pRCC
for i in range(0, 3):
    working_dir = os.path.join(data_dir, subtype1[i])
    for file_name in os.listdir(working_dir):
        if file_name.endswith('.nii.gz'):
            image_path = os.path.join(working_dir, file_name)
            out_path = os.path.join(working_dir, file_name[:-7]) + "r.nii.gz"
            random_rotate3D(image_path, out_path, -15, 15)
            print('{} is Done'.format(file_name))

### for elestic transformation of ChRCC, and pRCC
for i in range(0, 3):
    working_dir = os.path.join(data_dir, subtype1[i])
    for file_name in os.listdir(working_dir):
        if file_name.endswith('.nii.gz'):
            image_path = os.path.join(working_dir, file_name)
            out_path = os.path.join(working_dir, file_name[:-7]) + "e.nii.gz"
            elastic_transform_3D(image_path, out_path)
            print('{} is Done'.format(file_name))

subtype2 = ['benign', 'ccRCC']

### for rotate of ccRCC and benign
for i in range(0, 2):
    working_dir = os.path.join(data_dir, subtype2[i])
    for file_name in os.listdir(working_dir):
        if file_name.endswith('_cp10.nii.gz'):
            image_path = os.path.join(working_dir, file_name)
            out_path = os.path.join(working_dir, file_name[:-7]) + "r.nii.gz"
            random_rotate3D(image_path, out_path, -15, 15)
            print('{} is Done'.format(file_name))

### for elestic transformation of ccRCC and benign
for i in range(0, 2):
    working_dir = os.path.join(data_dir, subtype2[i])
    for file_name in os.listdir(working_dir):
        if file_name.endswith('_cp10.nii.gz'):
            image_path = os.path.join(working_dir, file_name)
            out_path = os.path.join(working_dir, file_name[:-7]) + "e.nii.gz"
            elastic_transform_3D(image_path, out_path)
            print('{} is Done'.format(file_name))