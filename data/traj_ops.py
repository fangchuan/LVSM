from typing import List

import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from scene.cameras import Camera

def interp_poses(w2c_poses: List[np.ndarray], num_frames: int = 24) -> List[np.ndarray]:
    """interpolate two camera poses"""
    v_rotation_in = np.zeros([0, 4])
    v_pos_x_in = []
    v_pos_y_in = []
    v_pos_z_in = []
    for i, pose in enumerate(w2c_poses):
        v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(pose[:3, :3]).as_quat()], axis=0)
        v_pos_x_in.append(pose[0, 3])
        v_pos_y_in.append(pose[1, 3])
        v_pos_z_in.append(pose[2, 3])

    in_times = np.arange(0, len(v_rotation_in)).tolist()
    out_times = np.linspace(0, len(v_rotation_in) - 1, num_frames).tolist()
    v_rotation_in = Rotation.from_quat(v_rotation_in)
    slerp = Slerp(in_times, v_rotation_in)
    v_interp_rotation = slerp(out_times)
    fx = interp1d(in_times, np.array(v_pos_x_in), kind="linear")
    fy = interp1d(in_times, np.array(v_pos_y_in), kind="linear")
    fz = interp1d(in_times, np.array(v_pos_z_in), kind="linear")
    v_interp_xs = fx(out_times)
    v_interp_ys = fy(out_times)
    v_interp_zs = fz(out_times)

    target_poses = []
    for idx in range(len(out_times)):
        rot_matrix = v_interp_rotation[idx].as_matrix()
        trans = np.array([v_interp_xs[idx], v_interp_ys[idx], v_interp_zs[idx]])
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = rot_matrix
        T_c2w[:3, 3] = trans

        target_poses.append(T_c2w)
    return target_poses
def gs_interpolate_trajectory(gs_views: List[Camera], nframes_interval:int=-1) -> List[Camera]:

    num_views = len(gs_views)


    # interpolate
    if nframes_interval == -1:
        nframes_interval = 24

    interp_gs_views = []
    for idx in range(num_views - 1):
        
        two_views = gs_views[idx:idx + 2]
        new_view = Camera
        # c2w pose
        w2c_poses = [view.world_view_transform.transpose(0, 1).detach().cpu().numpy() for view in two_views]
        c2w_poses = [np.linalg.inv(pose) for pose in w2c_poses]
        interp_c2w_poses = interp_poses(c2w_poses, nframes_interval)
        
        for interp_c2w_pose in interp_c2w_poses:
            R_c2w = interp_c2w_pose[:3, :3]
            t_w2c = np.linalg.inv(interp_c2w_pose)[:3, 3]
            new_view = Camera(
                colmap_id=two_views[0].colmap_id,
                R=R_c2w,
                T=t_w2c,
                FoVx=two_views[0].FoVx,
                FoVy=two_views[0].FoVy,
                image=two_views[0].original_image,
                depth=two_views[0].depth,
                gt_alpha_mask=None,
                image_name=str(len(interp_gs_views)),
                uid=len(interp_gs_views),
                warp_mask=None,
                K=None, 
                src_R=None, 
                src_T=None,
                src_uid=len(interp_gs_views)
            )
            
            interp_gs_views.append(new_view)
            
    return interp_gs_views