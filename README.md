# 3D-deep-learning-for-subtyping-of-renal-lesions
The mask of ITR-3mm (ITR with 3 mm shrink), ITR+3mm (ITR with 3 mm expansion) and ITR+5mm 5 mm (ITR with 3 mm expansion) expansion was obtained by using the code of Peritumor_ROIs.py.
Co-registration of non-contrast and venous phase images to the arterial phase image was by the code of Registration.py.
The ROIs was cropped by the code of Crope_ROIs.py.
Radiomic features was extracted with the code of Radiomic_feature_extraction.py.
Data augmentation was the code of Augmentation_rotation_transformation.py.
3D feature extration was by the code of Train_3D_deep_learning_model & Feature_extraction.py
The classifier was constructed with the code of AutoGluon_classifiers.
