# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import PIL
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F

from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt, Int64
from torch import Tensor


class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths


    def __len__(self):
        return len(self.all_scene_paths)


    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def __getitem__(self, idx):
        # try:
        scene_path = self.all_scene_paths[idx].strip()
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices= current_view_idx["context"] + current_view_idx["target"]
        else:
            # sample input and target views
            image_indices = self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
    
        # except:
        #     traceback.print_exc()
        #     print(f"error loading")
        #     print(image_indices)
        #     print(image_paths_chosen)
        #     return self.__getitem__(random.randint(0, len(self) - 1))


        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene_name
        }

class RoomverseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_ids = f.read().splitlines()
            self.all_scene_ids = [scene for scene in self.all_scene_ids if scene.strip()]
            self.all_scene_paths = [os.path.join(self.config.training.dataset_root, scene_id) for scene_id in self.all_scene_ids]
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        # if self.inference:
        #     self.view_idx_list = dict()
        #     if self.config.inference.get("test_split_file_path", None) is not None:
        #         if os.path.exists(self.config.inference.test_split_file_path):
        #             with open(self.config.inference.view_idx_file_path, 'r') as f:
        #                 self.test_scene_ids = f.read().splitlines()
        #                 # filter out None values, i.e. scenes that don't have specified input and targetviews
        #                 self.test_scene_ids = [scene for scene in self.test_scene_ids if scene.strip()]
        #             filtered_scene_paths = []
        #             for scene in self.all_scene_paths:
        #                 file_name = scene.split("/")[-1]
        #                 scene_name = file_name.split(".")[0]
        #                 if scene_name in self.view_idx_list_filtered:
        #                     filtered_scene_paths.append(scene)

        #             self.all_scene_paths = filtered_scene_paths

    def __len__(self):
        return len(self.all_scene_paths)

    # resize image size, return the image in [0, 1]
    def resize_image(self, image: Float[Tensor, "1 3 H W"], image_type: str='rgb', depth_scale: float = 1000.0) -> Float[Tensor, "1 3 H W"]:
        interpolation = cv2.INTER_NEAREST if image_type in ['depth', 'normal'] else cv2.INTER_LINEAR
        if image.shape[-2:] != (self.config.model.image_tokenizer.image_size, self.config.model.image_tokenizer.image_size):
            if image_type == 'depth':
                pers_img = torch.from_numpy(
                        cv2.resize(
                            image[0].permute(1, 2, 0).cpu().numpy(),
                            (self.config.model.image_tokenizer.image_size, self.config.model.image_tokenizer.image_size),
                            interpolation=interpolation,
                        )
                    )[None, :, :]          
            else:
                pers_img = torch.from_numpy(
                        cv2.resize(
                            image[0].permute(1, 2, 0).cpu().numpy(),
                            (self.config.model.image_tokenizer.image_size, self.config.model.image_tokenizer.image_size),
                            interpolation=interpolation,
                        )
                    ).permute(2, 0, 1)
        else:
            pers_img = image[0]
            
        if image_type in ['rgb', 'semantic']:
            # normalize rgb to [-1, 1]
            pers_img = pers_img.float() / 255.0
            pers_img = pers_img.clip(0.0, 1.0)
        elif image_type in ['depth']:
            pers_img = pers_img.float() / depth_scale
        elif image_type in ['normal']:
            # convert normal to [-1, 1]
            normal_img = pers_img.permute(1, 2, 0).cpu().numpy()
            normal = np.clip((normal_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
            normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
            # save normal in camera space, flip to make +z upward
            pers_img = torch.from_numpy(normal).permute(2, 0, 1).float()  #[3, 256, 256]
        return pers_img
    
    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices


    @staticmethod
    def _load_pil_image(file_path: str) -> Float[Tensor, "1 C H W"]:
        """Load image , Pillow version"""
        img = Image.open(file_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis].astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def __getitem__(self, idx):
        # try:
        scene_folderpath = self.all_scene_paths[idx].strip()
        room_uid = "/".join(scene_folderpath.split("/")[-4:])
        rgb_dir = os.path.join(scene_folderpath, "rgb")
        # depth_dir = os.path.join(scene_folderpath, "depth")
        # normal_dir = os.path.join(scene_folderpath, "normal")
        # semantic_dir = os.path.join(scene_folderpath, "semantic")
        cam_meta_filepath = os.path.join(scene_folderpath, "cameras.json")

        # load camera poses
        cameras_meta = json.load(open(cam_meta_filepath, 'r'))
        total_valid_cam_keys = list(cameras_meta["cameras"].keys())

        # random the num of input views
        num_sample_views = 16
        T_in = 1
        T_out = num_sample_views - T_in


        # use the sequential sampling
        # start_index = np.random.randint(0, len(total_valid_cam_keys))
        start_index = 0
        valid_frame_ids = np.roll(total_valid_cam_keys, -start_index)
        sample_keys = valid_frame_ids[:num_sample_views]

        if self.inference:
            image_indices= [int(k) for k in sample_keys]
        else:
            # TODO: add training dataset
            raise NotImplementedError("Training dataset for Roomverse is not implemented yet.")
            
        # c2w poses
        input_c2ws = [torch.from_numpy(np.array(cameras_meta["cameras"][view_key]).reshape(4, 4)).float() for view_key in sample_keys]
        input_c2ws: Float[Tensor, "N 4 4"] = torch.stack(input_c2ws, dim=0)  # [v, 4, 4]
        
        # RRGBs
        image_paths_chosen = [os.path.join(rgb_dir, f"frame_{ic}.jpg") for ic in image_indices]
        input_images = [self._load_pil_image(rgb_image_filepath) for rgb_image_filepath in image_paths_chosen]
        input_images = [self.resize_image(img, image_type='rgb') for img in input_images]
        input_images: Float[Tensor, "N 3 H W"] = torch.stack(input_images, dim=0)  # [v, C, H, W]
            
        # intrinsics
        intrinsic_mat = torch.tensor(cameras_meta["intrinsic"], dtype=torch.float32).reshape(3, 3)
        ori_img_height, ori_img_width = cameras_meta["height"], cameras_meta["width"]
        # resize intrinsic matrix w.r.t the hight
        if self.config.model.image_tokenizer.image_size != ori_img_height or self.config.model.image_tokenizer.image_size != ori_img_width:
            intrinsic_mat[0, 0] *= self.config.model.image_tokenizer.image_size / ori_img_height
            intrinsic_mat[1, 1] *= self.config.model.image_tokenizer.image_size / ori_img_height
            intrinsic_mat[0, 2] = self.config.model.image_tokenizer.image_size / 2.0
            intrinsic_mat[1, 2] = self.config.model.image_tokenizer.image_size / 2.0
        input_intrinsics: Float[Tensor, "N 4"] = torch.tensor([
            intrinsic_mat[0, 0], intrinsic_mat[1, 1],
            intrinsic_mat[0, 2], intrinsic_mat[1, 2]
        ], dtype=torch.float32).repeat(len(image_indices), 1)


        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        print(f"input-images: {input_images.shape}, input-c2ws: {input_c2ws.shape}, input-intrinsics: {input_intrinsics.shape}, indices: {indices.shape}, room_uid: {room_uid}")
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": room_uid
        }
class SpatialGenDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            # 确保 scene_path 是包含 npz 文件的目录
            self.scene_path = self.config.training.npz_path
            
            # 加载场景数据
            room_data = self._load_scene_data(self.scene_path)
            frames = self._build_frames(room_data)
            frames.sort(key=lambda x: x["image_name"])  # 按图像名称排序
            # print(frames[0]["image"].shape)
            # 创建所有上下文对
            self.all_context_pairs = []
            for i in range(len(frames) - 1):
                self.all_context_pairs.append({
                    "frame1": frames[i],
                    "frame2": frames[i + 1]
                })

        except Exception as e:
            print(f"Error reading dataset from '{self.config.training.npz_path}': {str(e)}")
            raise e

        self.inference = self.config.inference.get("if_inference", False)

    def __len__(self):
        return len(self.all_context_pairs)

    def _load_scene_data(self, scene_path: str) -> Dict[str, Any]:
        """从 NPZ 文件加载场景数据"""
        npz_path = os.path.join(scene_path, "inference_results.npz")
        data_dict = np.load(npz_path, allow_pickle=True)
        room_data = next(iter(data_dict.values())).item()
        return room_data

    def _build_frames(self, room_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从场景数据构建帧字典"""
        frames = []
        intrinsic = room_data['intrinsic']
        fxfycxcy = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        
        # 处理输入视图
        for i in range(len(room_data['input_rgbs'])):
            c2w = room_data['input_poses'][i]
            frames.append({
                "image": room_data['input_rgbs'][i],
                "c2w": c2w,
                "w2c": np.linalg.inv(c2w),
                "fxfycxcy": fxfycxcy.copy(),
                "image_name": f"input_{i}",
            })
        
        # 处理目标视图
        for i in range(len(room_data['target_rgbs'])):
            c2w = room_data['target_poses'][i]
            frames.append({
                "image": room_data['target_rgbs'][i],
                "c2w": c2w,
                "w2c": np.linalg.inv(c2w),
                "fxfycxcy": fxfycxcy.copy(),
                "image_name": f"target_{i}",
            })
            
        return frames

    def preprocess_frames(self, frames: List[Dict[str, Any]]) -> Tuple[Tensor, Tensor, Tensor]:
        """预处理帧（调整大小、裁剪和内参调整）"""
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)
        
        images, intrinsics, c2ws = [], [], []
        
        for frame in frames:
            img_arr = frame["image"]

            h, w = img_arr.shape[:2]
            img_arr = (img_arr).astype(np.uint8)

            # print("-"*64)
            # print(img_arr.shape,img_arr.min(), img_arr.max(), img_arr.dtype)# 結果是( 512, 512, 3) 0 255 uint8
            # print("-"*64)

            img = Image.fromarray(img_arr)
            
            # 保持宽高比调整宽度
            new_w = int(resize_h / h * w)
            new_w = int(round(new_w / patch_size) * patch_size)
            img = img.resize((new_w, resize_h), Image.LANCZOS)
            
            # 方形裁剪
            if square_crop:
                min_size = min(resize_h, new_w)
                start_h = (resize_h - min_size) // 2
                start_w = (new_w - min_size) // 2
                img = img.crop((start_w, start_h, start_w + min_size, start_h + min_size))
            
            # 转换为张量并归一化
            img_tensor = torch.from_numpy(np.array(img) / 255.0).permute(2, 0, 1).float()
            
            # 调整内参
            fx, fy, cx, cy = frame["fxfycxcy"]
            resize_ratio_x = new_w / w
            resize_ratio_y = resize_h / h
            fx, fy, cx, cy = (
                fx * resize_ratio_x,
                fy * resize_ratio_y,
                cx * resize_ratio_x,
                cy * resize_ratio_y
            )
            
            if square_crop:
                cx -= start_w
                cy -= start_h
                
            images.append(img_tensor)
            intrinsics.append(torch.tensor([fx, fy, cx, cy]).float())
            c2ws.append(torch.from_numpy(frame["c2w"]).float())
            
        return (
            torch.stack(images, dim=0),
            torch.stack(intrinsics, dim=0),
            torch.stack(c2ws, dim=0)
        )

    def preprocess_poses(
        self,
        c2ws: Tensor,
        scene_scale_factor: float = 1.35
    ) -> Tensor:
        """将位姿归一化到公共坐标系和尺度"""
        # 中心和对齐坐标系
        center = c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right), dim=-1)
        
        # 创建平均位姿
        avg_pose = torch.eye(4, device=c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)
        
        # 变换位姿
        c2ws = avg_pose @ c2ws
        
        # 重新缩放场景
        scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
        c2ws[:, :3, 3] /= scene_scale * scene_scale_factor
        
        return c2ws

    def interp_poses(self, w2c_poses: List[np.ndarray], num_frames: int = 24) -> List[np.ndarray]:
        """在两个相机位姿之间插值"""
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

    def save_debug_data(self, data: Dict[str, Any], scene_name: str, idx: int):
        """保存调试数据到./tmp目录"""
        # if not self.debug_enabled or self.debug_counter >= self.max_debug_samples:
        #     return
            
        try:
            # 创建场景专属目录
            scene_dir = os.path.join("./tmp/debug", f"{scene_name}_{idx}")
            os.makedirs(scene_dir, exist_ok=True)
            
            # 保存图像
            images_dir = os.path.join(scene_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            images = data["image"].numpy()
            for i, img_arr in enumerate(images):
                img_arr = np.transpose(img_arr, (1, 2, 0))  # [C, H, W] -> [H, W, C]  
                img = Image.fromarray((img_arr*255).astype(np.uint8))
                img.save(os.path.join(images_dir, f"frame_{i}.png"))
            
            # 保存位姿数据
            poses = {
                # "c2w_before_normalization": data["c2w_before_normalization"].detach().cpu().numpy().tolist(),
                "c2w_after_normalization": data["c2w"].detach().cpu().numpy().tolist(),
                "intrinsics": data["fxfycxcy"].detach().cpu().numpy().tolist()
            }
            
            with open(os.path.join(scene_dir, "poses.json"), "w") as f:
                json.dump(poses, f, indent=4)
            
            # 保存其他元数据
            metadata = {
                "scene_name": scene_name,
                "data_index": idx,
                "num_frames": len(images),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else str(self.config)
            }
            
            with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Debug data saved to: {scene_dir}")
            
            
        except Exception as e:
            print(f"Error saving debug data: {str(e)}")
            traceback.print_exc()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            # 获取上下文帧对
            context_pair = self.all_context_pairs[idx]
            frame1 = context_pair["frame1"]
            frame2 = context_pair["frame2"]
            
            # 准备所有帧数据（2个上下文帧 + 24个插值帧）
            all_frames = [frame1, frame2]
            
            # 在原始位姿之间插值
            w2c_poses = [frame1["w2c"], frame2["w2c"]]
            interp_c2ws = self.interp_poses(w2c_poses, num_frames=24)
            
            # 为插值帧创建帧数据
            for i, c2w in enumerate(interp_c2ws):
                all_frames.append({
                    "image": frame1["image"],  # 使用第一帧的图像作为占位
                    "c2w": c2w,
                    "w2c": np.linalg.inv(c2w),
                    "fxfycxcy": frame1["fxfycxcy"].copy(),  # 复制内参
                    "image_name": f"interp_{i}"
                })
            
            # 预处理所有帧
            images, intrinsics, c2ws = self.preprocess_frames(all_frames)
            
            # 归一化所有位姿
            normalized_c2ws = self.preprocess_poses(
                c2ws, 
                self.config.training.get("scene_scale_factor", 1.35)
            )
            
            # 准备索引张量
            num_frames = len(all_frames)  # 26 (2 + 24)
            image_indices_t = torch.arange(0, num_frames).long().unsqueeze(-1)
            scene_indices_t = torch.full_like(image_indices_t, idx)
            indices = torch.cat([image_indices_t, scene_indices_t], dim=-1)
            
            # 创建场景名称
            scene_name = f"{os.path.basename(self.scene_path)}_{frame1['image_name']}_{frame2['image_name']}"
            result = {
                "image": images,
                "c2w": normalized_c2ws,
                "fxfycxcy": intrinsics,
                "index": indices,
                "scene_name": scene_name
            }

            # self.save_debug_data(result, scene_name, idx)
            return result
            
        except Exception as e:
            print(f"Error loading scene {self.scene_path}, idx {idx}: {str(e)}")
            traceback.print_exc()
            # 重试随机样本
            # return self.__getitem__(random.randint(0, len(self) - 1))