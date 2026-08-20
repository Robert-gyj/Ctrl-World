import mediapy
import os
from diffusers.models import AutoencoderKL
import mediapy
import torch
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
import mediapy
from torch.utils.data import Dataset

import pandas as pd
from accelerate import Accelerator


class EncodeLatentDataset(Dataset): 
    def __init__(self, old_path, new_path, svd_path, device, size=(192, 320), rgb_skip=3):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))  # 使用 json.loads() 解析单行
            return data

        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        instruction = traj_data['tasks'][0]
        traj_id = traj_data['episode_index']
        chunk_id = int(traj_id/1000)

        data_type = 'val' if traj_id%100 == 99 else 'train'
        # if os.path.exists(f"{self.new_path}/videos/{data_type}/{traj_id}") and os.path.exists(f"{self.new_path}/latent_videos/{data_type}/{traj_id}"):
        #     # print(f"Skipping trajectory {traj_id}, already processed.")
        #     return 0

        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        length = len(df['observation.state.cartesian_position'])

        obs_car = []
        obs_joint =[]
        obs_gripper = []
        action_car = []
        action_joint = []
        action_gripper = []
        action_joint_vel = []

        for i in range(length):
            obs_car.append(df['observation.state.cartesian_position'][i].tolist())
            obs_joint.append(df['observation.state.joint_position'][i].tolist())
            obs_gripper.append(df['observation.state.gripper_position'][i].tolist())
            action_car.append(df['action.cartesian_position'][i].tolist())
            action_joint.append(df['action.joint_position'][i].tolist())
            action_gripper.append(df['action.gripper_position'][i].tolist())
            action_joint_vel.append(df['action.joint_velocity'][i].tolist())
        success = df['is_episode_successful'][0]
        video_paths = [
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_1_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_2_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.wrist_left/episode_{traj_id:06d}.mp4']
        traj_info = {'success': success,
                     'observation.state.cartesian_position': obs_car,
                     'observation.state.joint_position': obs_joint,
                     'observation.state.gripper_position': obs_gripper,
                     'action.cartesian_position': action_car,
                     'action.joint_position': action_joint,
                     'action.gripper_position': action_gripper,
                     'action.joint_velocity': action_joint_vel,
                    }
        

        # if f"{save_root}/videos/{data_type}/{traj_id}" exist, skip this trajectory
        try:
            self.process_traj(video_paths, traj_info, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, rgb_skip=self.skip, device=self.vae.device)
        except:
            print(f"Error processing trajectory {traj_id}, skipping...")
            return 0
    
        return 0


    def process_traj(self, video_paths, traj_info, instruction, save_root,traj_id=0,data_type='val', size=(192,320), rgb_skip=3, device='cuda'):
        for video_id, video_path in enumerate(video_paths):
            # load and resize video and save
            video = mediapy.read_video(video_path)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
            frames = frames[::rgb_skip]  # Skip frames to save memory here!!!
            x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=5)

            # save svd latent
            x = x.to(device)
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")
        
        # record cartesain aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_gripper = np.array(traj_info['observation.state.gripper_position'])[:,None]
        # print(cartesian_pose.shape, cartesian_gripper.shape)
        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper),axis=-1)[::rgb_skip].tolist()

        # record joints aligned with video frames; rollout scripts read anno['joints']
        joint_pose = np.array(traj_info['observation.state.joint_position'])
        joint_gripper = np.array(traj_info['observation.state.gripper_position'])[:,None]
        joint_states = np.concatenate((joint_pose, joint_gripper),axis=-1)[::rgb_skip].tolist()
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": frames.shape[0],
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"}
            ],
            'states': cartesian_states,
            'joints': joint_states,
            'observation.state.cartesian_position': traj_info['observation.state.cartesian_position'],
            'observation.state.joint_position': traj_info['observation.state.joint_position'],
            'observation.state.gripper_position': traj_info['observation.state.gripper_position'],
            'action.cartesian_position': traj_info['action.cartesian_position'],
            'action.joint_position': traj_info['action.joint_position'],
            'action.gripper_position': traj_info['action.gripper_position'],
            'action.joint_velocity': traj_info['action.joint_velocity'],
            }
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_hf_path', type=str, default='/cephfs/shared/droid_hf/droid_1.0.1')
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_subset')
    parser.add_argument('--svd_path', type=str, default='/cephfs/shared/llm/stable-video-diffusion-img2vid')
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=args.droid_hf_path,
        new_path= args.droid_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=3, #  to downsample 15hz video to 5hz video
    )
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Precomputed {idx} samples")

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug

