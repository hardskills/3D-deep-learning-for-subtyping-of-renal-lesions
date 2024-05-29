import os
import pandas as pd
import itk

dir_path = "D:\CTdata\TestData\Rimage"
mov_path = "D:\CTdata\TestData\Oimage"
sav_path = "D:\CTdata\TestData\Rimage"
data = pd.read_csv(r"D:\Data_analysis\DL_3D\Test_age_gender_patho_selected.csv")
data = data[data.v_phase > 0]
index = data['id'].values.tolist()

parameter_object = itk.ParameterObject.New()
parameter_object.AddParameterFile("D:\PyProject\Models\Registration\Parameters_Translation.txt")
parameter_object.AddParameterFile("D:\PyProject\Models\Registration\Parameters_Rigid.txt")
parameter_object.AddParameterFile("D:\PyProject\Models\Registration\Parameters_BSpline.txt")
parameter_object.AddParameterFile("D:\PyProject\Models\Registration\Parameters_Affine.txt")

for i in range(473, 700):
    file_name = index[i]
    print(i)
    print(file_name)
    fixed_image_path = os.path.join(dir_path, file_name) + "_ra.nii.gz"
    moving_image_path = os.path.join(mov_path, file_name) + "_V.nii.gz"
    print(fixed_image_path)
    print(moving_image_path)
    saving_image_path = os.path.join(sav_path, file_name) + "_vm.nii.gz"
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)

    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False)

    itk.imwrite(result_image, saving_image_path)