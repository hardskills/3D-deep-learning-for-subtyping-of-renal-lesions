import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from autogluon.tabular import TabularPredictor

## combined model ##
data_r = pd.read_csv("D:\Data_analysis\DL_3D\Feature_nb_ACC.csv")
data = data_r.loc[data_r['patho_msubtype'].isin(['ccRCC', 'ChRCC', 'pRCC', 'benign'])]
data = data.loc[data['mask'] == 'cp30']
# basic: 0-15; 3D_A: 15-527; 3D_C: 527-1039; 3D_V: 1039-1551; A: 1551-3332; C: 3332-5113; V: 5113-6894
data_s = data.iloc[:, np.r_[0:15, 15:527, 1039:1551]]
data_s = data_s.dropna()

data_enroll = data_s[data_s.diameter <= 210]
data_patient = data_enroll[['id', 'diameter', 'patho_msubtype', 'index', 'index_new']]
data_v = data_enroll.drop(columns=['id', 'mask', 'gender', 'age',  'laterality', 'diameter', 'a_phase', 'c_phase', 'v_phase', 'image_class_pos', 'image_class_une', 'patho_msubtype', 'patho_auditTime', 'index', 'index_new'])
## end ##

var_thr = VarianceThreshold(threshold=0)
var_thr.fit(data_v)
data_z = data_v.loc[:,var_thr.get_support()]
data = pd.concat([data_patient, data_z], axis=1)

data_train = data[data["index_new"] == 'tra']
data_train_x = data_train.drop(columns=['id', 'diameter', 'patho_msubtype', 'index', 'index_new'])
data_train_y = data_train["patho_msubtype"]

num_fea = 1024
f_selector = SelectKBest(f_classif, k=num_fea)
f_selector.fit(data_train_x, data_train_y)
f_support = f_selector.get_support()
data_train_feature = data_train_x.loc[:, f_support]
data_train = pd.concat([data_train_feature, data_train_y], axis=1)
label = "patho_msubtype"

data_vali = data[data["index"] == 'val']
data_vali_x = data_vali.drop(columns=['id', 'diameter', 'patho_msubtype', 'index', 'index_new'])
data_vali_feature = data_vali_x.loc[:, f_support]
data_vali_y = data_vali["patho_msubtype"]
data_vali = pd.concat([data_vali_feature, data_vali_y], axis=1)

data_test = data[data["index"] == 'tes']
data_test_x = data_test.drop(columns=['id', 'diameter', 'patho_msubtype', 'index', 'index_new'])
data_test_feature = data_test_x.loc[:, f_support]
data_test_y = data_test["patho_msubtype"]
data_test = pd.concat([data_test_feature, data_test_y], axis=1)

data_vali_test = pd.concat([data_vali, data_test])

save_path = r"D:\PyProject\Models\AutoGluon\ST\ACC\3D_nb_cp30_M"
# predictor = TabularPredictor(label=label, path=save_path, eval_metric='balanced_accuracy').fit(data_train, hyperparameters={'NN_TORCH':{}})
predictor = TabularPredictor.load(save_path)
results = predictor.fit_summary()
predictor.features()

# fea_imp = predictor.feature_importance(data_vali_test)
# pd.DataFrame(fea_imp).to_csv("D:\PyProject\ReTumor\AutoGluon\Results_ST\RAD\FeaSel_importance_RA_cp50_M01.csv")

y_train_prob = predictor.predict_proba(data_train_feature)
y_train_cate = predictor.predict(data_train_feature)

y_vali_prob = predictor.predict_proba(data_vali_feature)
y_vali_cate = predictor.predict(data_vali_feature)

y_test_prob = predictor.predict_proba(data_test_feature)
y_test_cate = predictor.predict(data_test_feature)

perf_vali = predictor.evaluate_predictions(y_true=data_vali_y, y_pred=y_vali_prob, auxiliary_metrics=True)
print(perf_vali)
print(predictor.leaderboard(data_vali, silent=True))
print(predictor.leaderboard(extra_info=True, silent=True))

perf_test = predictor.evaluate_predictions(y_true=data_test_y, y_pred=y_test_prob, auxiliary_metrics=True)
print(perf_test)
print(predictor.leaderboard(data_test, silent=True))
print(predictor.leaderboard(extra_info=True, silent=True))

# pd.DataFrame(y_vali_prob).to_csv("D:\PyProject\ReTumor\AutoGluon\Results_ST\ACC\data_3D_nb_vali_cp20_M_prob.csv")
# pd.DataFrame(y_vali_cate).to_csv("D:\PyProject\ReTumor\AutoGluon\Results_ST\ACC\data_3D_nb_vali_cp20_M_cate.csv")
#
# pd.DataFrame(y_test_prob).to_csv("D:\PyProject\ReTumor\AutoGluon\Results_ST\ACC\data_3D_nb_test_cp20_M_prob.csv")
# pd.DataFrame(y_test_cate).to_csv("D:\PyProject\ReTumor\AutoGluon\Results_ST\ACC\data_3D_nb_test_cp20_M_cate.csv")