# Preprocessing
### Projection - Multi_view_projection folder
1. projection_final.py (Projection 54 images per healthy subject)
2. projection_final_TB.py (Projection 54 images per tuberculosis subject)
3. projection_final_PN.py (Projection 54 images per Pneumonia subject)

# Training
### Training - Anomaly_detection folder
1. patchcore_AD_each_view_args_1.py (Training for the first projection plane) - bash run_one.sh
2. patchcore_AD_each_view_args_2.py (Training for the second projection plane) - bash run_two.sh
3. patchcore_AD_each_view_args_3.py (Training for the third projection plane) - bash run_three.sh

# Inference
### Inference - inference folder
1. patchcore_AD_each_view_args_1.py (Inference for the first projection plane) - bash test_one.sh
2. patchcore_AD_each_view_args_2.py (Inference for the second projection plane) - bash test_two.sh
3. patchcore_AD_each_view_args_3.py (Inference for the third projection plane) - bash test_three.sh

### Evaluation
1. auc_cal_final_3x3x3.py (3x3x3 score)

# Visualization
### SMC dataset - Anomaly_detection
1. generate_3d_map.py (Generate anomaly score map for tuberculosis and pneumonia subjects)



