import os
import nibabel as nib
import pandas as pd
import numpy as np
import SimpleITK as sitk

def maskcroppingbox(image_path, mask_path, out_path, a=1, b=1, c=1, use2D=False):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array.astype('float32')
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = mask_array.astype('float32')
    mask_array_2 = np.argwhere(mask_array)  # 根据mask在原图找到对应的坐标位置
    # 寻找mask的z轴的上下边界所在层数
    (zstart, ystart, xstart), (zstop, ystop, xstop) = mask_array_2.min(axis=0), mask_array_2.max(axis=0)
    #1: rmask; 2: shrink3; 3: expand3; 5: expand5
    #1: 3,3,3; 2: 3,3,-3; 3: 3,-3,3; 4: -3,3,3; 5: 3,-3,-3; 6: -3,-3,3; 7: -3,3,-3; 8: -3,-3,-3
    roi_image = image_array[zstart+a-1:zstop+a, ystart+b-1:ystop+b, xstart+c-1:xstop+c]
    roi_mask = mask_array[zstart+a-1:zstop+a, ystart+b-1:ystop+b, xstart+c-1:xstop+c]
    # roi_image[roi_mask < 1] = 0
    roi_image = nib.Nifti1Image(roi_image, nib.load(image_path).affine)
    nib.save(roi_image, out_path)

# 路径定义
data_dir = r"D:\CTdata\TestData\Rimage"
mask_dir = r"D:\CTdata\TestData\Rimage"
output_dir = r"D:\CTdata\MaskCrop\bg\C\train"
image_id = pd.read_csv("D:\Data_analysis\DL_3D\data_train_index.csv")
image_id = image_id[image_id.c_phase > 0]

## for files of benign、ChRCC and pRCC
mask_path_s = ['_rmask.nii.gz', '_shrink3.nii.gz', '_expand3.nii.gz', '_expand5.nii.gz']
out_path_s1 = ['benign', 'ChRCC', 'pRCC']
out_path_s2 = ['_cp1', '_cp2', '_cp3', '_cp5']
A = [1, 4, 4, 4, -2, 4, -2, -2, -2]
B = [1, 4, 4, -2, 4, -2, -2, 4, -2]
C = [1, 4, -2, 4, 4, -2, 4, -2, -2]

for i in range(0, 4):
    image_id_s = image_id[image_id.patho_msubtype == out_path_s1[i]]
    index = image_id_s['id'].values.tolist()
    out_path1 = os.path.join(output_dir, out_path_s1[i])
    for l in range(0, len(image_id_s)):
        file_name = index[l]
        image_path = os.path.join(data_dir, file_name) + "_cm.nii.gz"
        for n in range(0, 4):
            mask_path = os.path.join(mask_dir, file_name) + mask_path_s[n]
            for m in range(0, 9):
                out_path = os.path.join(out_path1, file_name) + out_path_s2[n] + str(m) + ".nii.gz"
                maskcroppingbox(image_path, mask_path, out_path, a=A[m], b=B[m], c=C[m], use2D=False)
                print('{} is Done'.format(out_path))

## for files of ccRCC
out_dir_a = r"D:\CTdata\MaskCrop\bg\C\train\ccRCC"
image_id_a = image_id[image_id.patho_msubtype == 'ccRCC']
index_a = image_id_a['id'].values.tolist()
for i in range(0, len(image_id_a)):
    file_name = index_a[i]
    image_path = os.path.join(data_dir, file_name) + "_cm.nii.gz"
    for n in range(0, 4):
        mask_path = os.path.join(mask_dir, file_name) + mask_path_s[n]
        for m in range(0, 1):
            out_path = os.path.join(out_dir_a, file_name) + out_path_s2[n] + str(m) + ".nii.gz"
            maskcroppingbox(image_path, mask_path, out_path, a=A[m], b=B[m], c=C[m], use2D=False)
            print('{} is Done'.format(out_path))

for i in range(0, len(image_id_a)):
    file_name = index_a[i]
    image_path = os.path.join(data_dir, file_name) + "_cm.nii.gz"
    for n in range(0, 1):
        mask_path = os.path.join(mask_dir, file_name) + mask_path_s[n]
        for m in range(1, 9):
            out_path = os.path.join(out_dir_a, file_name) + out_path_s2[n] + str(m) + ".nii.gz"
            maskcroppingbox(image_path, mask_path, out_path, a=A[m], b=B[m], c=C[m], use2D=False)
            print('{} is Done'.format(out_path))

## for validation file
output_dir_v = r"D:\CTdata\TestCrop\bg\A"
image_id_v = pd.read_csv("D:\Data_analysis\DL_3D\Test_age_gender_patho_selected.csv")
image_id_v = image_id_v[image_id_v.c_phase > 0]
out_path_sv = ['benign', 'ccRCC', 'ChRCC', 'OtherMG', 'pRCC']
for i in range(0, 5):
    image_id_s_v = image_id_v[image_id_v.patho_msubtype == out_path_sv[i]]
    index = image_id_s_v['id'].values.tolist()
    out_path1 = os.path.join(output_dir_v, out_path_sv[i])
    for l in range(0, len(image_id_s_v)):
        file_name = index[l]
        image_path = os.path.join(data_dir, file_name) + "_cm.nii.gz"
        for n in range(0, 4):
            mask_path = os.path.join(mask_dir, file_name) + mask_path_s[n]
            for m in range(0, 1):
                out_path = os.path.join(out_path1, file_name) + out_path_s2[n] + str(m) + ".nii.gz"
                maskcroppingbox(image_path, mask_path, out_path, a=A[m], b=B[m], c=C[m], use2D=False)
                print('{} is Done'.format(out_path))