work_dir=work_dirs/masking_strategy/version1/mask_l_defaultm_deform_attn_1_ca
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_mask.py 4 --work-dir $work_dir
for failures in 'beam_reduction' 'camera_stuck' 'camera_view_drop' 'lidar_stuck' 'limited_fov' 'object_failure' 'spatial_misalignment' 'lidar_drop' 'occlusion'
do
    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
done