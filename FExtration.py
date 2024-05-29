import os
import pandas as pd
from radiomics import featureextractor

dir_path_i = r"D:\CTdata\TestData\Rimage"
dir_path_m = r"D:\CTdata\TestData\Rimage"
sav_path = "D:\PyProject"
data = pd.read_csv("D:\Data_analysis\DL_3D\Test_age_gender_patho_selected.csv")
data_enroll_c = data[data.v_phase > 0]
df = pd.DataFrame()

settings = {}
settings['binWidth'] = 25
settings['sigma'] = [3, 5]
# settings['Interpolator'] = [1, 1, 1]
settings['voxelArrayShift'] = 1000
settings['normalize'] = True
settings['normalizeScale'] = 100
# settings['correctMask'] = True
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
print('Extraction parameters:\n\t', extractor.settings)

extractor.enableAllImageTypes()
# extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
extractor.enableAllFeatures()
extractor.enableImageTypeByName('LBP2D', False, {})

# for mask in ['_border3', '_border3e', '_border5', '_border5e', '_expand3', '_expand5', '_shrink3']:
for mask in ["_rmask", '_shrink3', '_expand3', '_expand5']:
    df = pd.DataFrame()
    for i in range(0, 680):
        file_name = data_enroll_c['id'].iloc[i]
        image_path = os.path.join(dir_path_i, file_name) + "_vm.nii.gz"
        print(image_path)
        mask_path = os.path.join(dir_path_m, file_name) + mask + ".nii.gz"
        featureVector = extractor.execute(image_path, mask_path)
        df_new = pd.DataFrame.from_dict(featureVector.values()).T
        df_new.columns = featureVector.keys()
        df_new.insert(0, 'imageFile', file_name)
        df = pd.concat([df, df_new])

    df.to_excel(os.path.join(sav_path, 'Test_v_result' + mask + '.xlsx'), index=None)

# dir_path_i = "D:\CTdata\KiTS_Rimage"
# sav_path = "D:\PyProject"
# dir_path_m = "D:\CTdata\KiTS_ITHOupt\SVSplit"
# data = pd.read_csv("D:\Data_analysis\ITHAnalysis\KiTS21\kits_age_gender_patho.csv")
# data_enroll = data[data.enroll > 0]
# # data_enroll_a = data[data.v > 0]
#
# settings = {}
# settings['binWidth'] = 25
# settings['voxelArrayShift'] = 1000
# settings['normalize'] = True
# settings['normalizeScale'] = 100
# settings['minimumROISize'] = 1
# settings['minimumROIDimensions'] = 1
# extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
# print('Extraction parameters:\n\t', extractor.settings)
#
# # extractor.enableAllImageTypes()
# extractor.enableImageTypes(Original={})
# extractor.enableAllFeatures()
# # extractor.enableImageTypeByName('LBP2D', False, {})
#
# for i in range(203, 312):
#     df = pd.DataFrame()
#     file_name = data_enroll['id'].iloc[i]
#     print(i)
#     print(file_name)
#     image_path = os.path.join(dir_path_i, file_name) + "_ra.nii.gz"
#     mask_path = os.path.join(dir_path_m, file_name)
#     mask_files = os.listdir(mask_path)
#     mask_files.sort()
#     for n in range(1,len(mask_files)+1):
#         mask_path = os.path.join(dir_path_m, file_name, file_name) + "_" + str(n) + ".nii.gz"
#         print(mask_path)
#         featureVector = extractor.execute(image_path, mask_path)
#         df_new = pd.DataFrame.from_dict(featureVector.values()).T
#         df_new.columns = featureVector.keys()
#         df_new.insert(0, 'imageFile', file_name + "_" + str(n))
#         df = pd.concat([df, df_new])
#
#     df.to_excel(os.path.join(sav_path, str(i) + '_' + 'ITH.xlsx'), index=None)