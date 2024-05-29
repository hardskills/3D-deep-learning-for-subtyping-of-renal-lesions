# 3D-deep-learning-for-subtyping-of-renal-lesions
1. The mask of ITR-3mm (ITR with 3 mm shrink), ITR+3mm (ITR with 3 mm expansion) and ITR+5mm (ITR with 5 mm expansion) expansion was obtained by using the code of Peritumor_ROIs.py.

2. Co-registration of non-contrast and venous phase images to the arterial phase image was by the code of Registration.py.

3. The ROIs was cropped by the code of Crope_ROIs.py.

4. Radiomic features was extracted with the code of Radiomic_feature_extraction.py.

5. Data augmentation was the code of Augmentation_rotation_transformation.py.

6. 3D feature extration was by the code of Train_3D_deep_learning_model & Feature_extraction.py

7. The classifier was constructed with the code of AutoGluon_classifiers.
