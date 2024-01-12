bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_CA_pers.py 4 --work-dir work_dirs/CA_residual_freeze_img_pers
#bash tools/dist_test.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_CA.py work_dirs/CA_residual/epoch_5.pth 4 --work-dir work_dirs/CA_residual
#bash tools/dist_test.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_failure.py work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_5.pth 4 --work-dir work_dirs/baseline_7107/lidar_stuck

# for failures in 'beam_reduction' 'camera_stuck' 'camera_view_drop' 'lidar_stuck' 'limited_fov' 'object_failure' 'spatial_misalignment'
# do
#     #python3 tools/test.py projects/BEVFusion/configs/failure/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py checkpoint/bevfusion_7107.pth --work-dir work_dirs/baseline_7107/$failures
#     python3 tools/test.py projects/BEVFusion/configs/CA_failure/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py checkpoint/CA_residual_freeze_img.pth --work-dir work_dirs/CA_residual_freeze_img/$failures
# done