work_dir=work_dirs/mask_grid_patch_15_freq_025
bash tools/dist_train.sh  ./projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_mask_grid_patch.py 4 --work-dir $work_dir
for failures in 'beam_reduction' 'camera_view_drop' 'limited_fov' 'object_failure' 'spatial_misalignment' 'lidar_drop' 'occlusion'
do
    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_residual/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
done
