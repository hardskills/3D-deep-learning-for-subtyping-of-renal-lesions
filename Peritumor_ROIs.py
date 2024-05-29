import os
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

def array2nii(image_array, out_path, NIIimage_resample):
    ## image_array是矩阵，out_path是带文件名的路径，NIIimage_resample是sitk_obj
    # 1.构建nrrd阅读器
    image2 = NIIimage_resample
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, out_path)

# 定义原始图像和ROI文件夹路径
dir_path = r"D:\CTdata\TestData\Oimage"
sav_path = r"D:\CTdata\TestData\Rimage1"
data = pd.read_csv(r"D:\Data_analysis\ITHAnalysis\Test_age_gender_patho_selected.csv")

# 批量处理
for i in range (0, 2):
    file_name = data['id'][i]
    print(file_name)
    img_path = os.path.join(dir_path, file_name) + "_A.nii.gz"
    roi_path = os.path.join(dir_path, file_name) + "_mask.nii.gz"
    print(img_path)
    print(roi_path)

    tumorImage = sitk.ReadImage(img_path)
    newSpacing = [1.0, 1.0, 1.0]
    resamplemethod = sitk.sitkLinear
    Niimage_resample = sitk.Resample(tumorImage,
                                     [int(round(tumorImage.GetSize()[i] * tumorImage.GetSpacing()[i] / newSpacing[i]))
                                      for i in range(3)], sitk.Transform(), resamplemethod, tumorImage.GetOrigin(),
                                     newSpacing, tumorImage.GetDirection(), 0.0, tumorImage.GetPixelID())
    image_array = sitk.GetArrayFromImage(Niimage_resample)
    mask_img_resample = sitk.Resample(sitk.ReadImage(roi_path),
                                      [int(round(tumorImage.GetSize()[i] * tumorImage.GetSpacing()[i] / newSpacing[i]))
                                       for i in range(3)], sitk.Transform(), sitk.sitkNearestNeighbor,
                                      tumorImage.GetOrigin(), newSpacing, tumorImage.GetDirection(), 0.0,
                                      sitk.sitkUInt8)

    mask_img_arr = sitk.GetArrayFromImage(mask_img_resample)
    iteration = 3  # newSpacing是1mm,扩展3mm
    mask_img_arr_expand3 = ndimage.binary_dilation(mask_img_arr, iterations=3).astype(mask_img_arr.dtype)
    mask_img_arr_expand5 = ndimage.binary_dilation(mask_img_arr, iterations=5).astype(mask_img_arr.dtype)
    mask_img_arr_shrink3 = ndimage.binary_erosion(mask_img_arr, iterations=3).astype(mask_img_arr.dtype)
    mask_img_arr_border3 = mask_img_arr_expand3 - mask_img_arr
    mask_img_arr_border5 = mask_img_arr_expand5 - mask_img_arr
    mask_img_arr_border3e = mask_img_arr_expand3 - mask_img_arr_shrink3
    mask_img_arr_border5e = mask_img_arr_expand5 - mask_img_arr_shrink3

    output_folder_img = os.path.join(sav_path,file_name) + "_ra.nii.gz"
    output_folder_roi = os.path.join(sav_path,file_name) + "_rmask.nii.gz"
    output_folder_expand3 = os.path.join(sav_path, file_name) + "_expand3.nii.gz"
    output_folder_expand5 = os.path.join(sav_path, file_name) + "_expand5.nii.gz"
    output_folder_shrink3 = os.path.join(sav_path, file_name) + "_shrink3.nii.gz"


    sitk.WriteImage(Niimage_resample, output_folder_img)
    sitk.WriteImage(mask_img_resample, output_folder_roi)
    array2nii(mask_img_arr_expand3, output_folder_expand3, Niimage_resample)
    array2nii(mask_img_arr_expand5, output_folder_expand5, Niimage_resample)
    array2nii(mask_img_arr_shrink3, output_folder_shrink3, Niimage_resample)
