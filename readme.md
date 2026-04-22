<div align="center">
<h2><center>👉 Ctrl-World: A Controllable Generative World Model for Robot Manipulation </h2>

[Yanjiang Guo*](https://robert-gyj.github.io), [Lucy Xiaoyang Shi*](https://lucys0.github.io),  [Jianyu Chen](http://people.iiis.tsinghua.edu.cn/~jychen/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)

 \*Equal contribution; Stanford University, Tsinghua University

ICLR 2026

<a href='https://arxiv.org/abs/2510.10125'><img src='https://img.shields.io/badge/ArXiv-2510.10125-red'></a> 
<a href='https://ctrl-world.github.io/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This repo includes the official PyTorch implementation for ICLR 2026 [**Ctrl-World**](https://sites.google.com/view/ctrl-world) paper. And also include the world model post-training process in [**VLAW**](https://sites.google.com/view/vlaw-arxiv) paper.

**TL; DR:** Ctrl-World is an action-conditioned world model compatible with modern VLA policies and enables policy-in-the-loop rollouts entirely in imagination, which can be used to evaluate and improve the **instruction following** ability of VLA. 

<p>
    <img src="synthetic_traj/gallery/ctrl_world.jpg" alt="wild-data" width="100%" />
</p>
<!-- synthetic_traj/gallery/ctrl_world.jpg -->



##  Content
**[2026.02] New: add initial conditiones use in paper [here](https://github.com/Robert-gyj/Ctrl-World?tab=readme-ov-file#-3-new-interact-with-pi_05-model-within-world-model-with-initial-conditions-in-the-paper) and wm post-training [here](https://github.com/Robert-gyj/Ctrl-World?tab=readme-ov-file#-3-new-post-train-world-model-on-down-stream-tasks)**

[2025.10] 1. Generate synthetic trajectory via replaying the recorded actions in DROID dataset.

[2025.10] 2. Generate synthetic trajectory via keyboard interactions.

[2025.10] 3. Generate synthetic trajectory via interaction with advanced VLA model $\pi_{0.5}$.

[2025.10] 4. A training pipeline of Ctrl-World on DROID dataset.


## Installation 🛠️


```bash
conda create -n ctrl-world python==3.11
conda activate ctrl-world
pip install -r requirements.txt

#  If you want to use ctrl-world to interact with $\pi_{0.5}$ model, following the pi official repo to install the pi model dependencies. Otherwise you can skip it.
# (from https://github.com/Physical-Intelligence/openpi/tree/main)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


## CheckPoint and Dataset 📷


| Ckpt name     | Training type | Size |
|---------------|------------------|---------|
| [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)  | CLIP text and image encoder    |  ~600M   |
| [svd](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)  | Pretrained SVD video diffusion model   | ~8G    |
| [Ctrl-World](https://huggingface.co/yjguo/Ctrl-World) |   Ctrl-World model trained on DROID dataset  | ~8G   |
| [DROID Dataset](https://huggingface.co/datasets/cadene/droid_1.0.1) |   Opensourced DROID dataset, ~95k traj, 564 scene    |  ~370G  |


<!-- **📊 Replay opensourced trajectory:** If you want to replay 

**📊 Replicate results on calvin abc:** If you want to replicate results on calvin abc, download the svd-robot-calvin model.

**📊  Train VPP in cunstom environments**: If you want to run VPP algorithm on your own robot, download the svd-robot model and follow instructions in the training section. -->



## Ctrl-World Inference 📊
### 📊 (1) Replay the recorded trajectories within world model.
**Task Description:** We start from an initial observation sampled from the recorded trajectories and then generate long trajectories by replaying the recorded actions. At each interaction step, a 1-second action chunk is provided to the world model, and the interaction is repeated multiple times to produce the full rollout. 

We provide a very small subset of DROID dataset in `dataset_example/droid_subset`. After download the ckpt in section 1, you can directly run the following command to replay some long trajectories:


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt}
```
The rollout configuration can be found in `config.py` in function `__post_init__`.
If you want to replay more trajectories, you need to download and process the original DROID datasets following the instructions in training section.

*Tip: One interaction step takes around ~10s on A100 or ~5s on H100.*

### 📊 (2) Interact with world model via keyboard control.
**Task Description:** We begin from an initial observation sampled from the recorded trajectories and use keyboard commands to control the robot interactively.

Each keyboard command is converted into an action chunk, and the set of valid commands includes:
{ l: left, r: right, f: forward, b: backward, u: up, d: down, o: open gripper, c: close gripper }.

You can input multiple commands at once, and the system will execute them sequentially in an autoregressive manner.
For example, you can run the following command:


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --task_type keyboard --keyboard lllrrr
```

### 📊 (3) Interact with $\pi_{0.5}$ model within world model

**Task Description:** We take some snapshot from a new DROID setup and perform policy-in-the-loop rollouts inside world model. Both $\pi_{0.5}$ and Ctrl-World need to zero-shot transferr to new setups.

We also need to download official $\pi_{0.5}$-DROID checkpoint following [official openpi repo](https://github.com/Physical-Intelligence/openpi). We provide some snapshots in `dataset_example/droid_new_setup`. These snapshot are from new DROID setups out of opensourced dataset. we tried tasks including `task_types = ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','stack']`. 

*Claims: We only train Ctrl-World on opensourced DROID dataset and zero-shot transferred to our new DROID setups. The model can evaluate a policy’s instruction-following capability but also can be imprecise in modeling physical interactions.*

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${pickplace}
```
Alternatively, you can configure all parameters in config.py and run `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py`. Since the official $\pi_{0.5}$ policies are implemented in JAX, we need to set XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 to prevent JAX from pre-allocating too much GPU memory.

### 📊 (3) <span style="color:red;">New</span>: Interact with $\pi_{0.5}$ model within world model with initial conditions in the paper
In the paper, we run each category of task for 20 times. Each category of task may have 5 or 10 initial configurations and repeat for 2 or 4 times (20 times in total). You can run following command by settng `task_type` you want. All initial condition is in `dataset_example/droid_new_setup_full`.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi_eval.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path ${path to clip folder} --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${fold_tower}
```



## Pre-Training/Post-training Ctrl-World 📊

In this section, we provide detailed instructions on how to train Ctrl-World on DROID dataset. If you want to train with custum datasets, you can also follow this instructions with neccesary modifications.


### 🛸 (0) Requirements for training on whole droid dataset
Our experiments are run on one/two nodes each with 8 A100/H100 cards.

### 🛸 (1) Prepare dataset
(1) Since the video diffusion model are run in latent space of image encoder, we first extract the latent space of the video to improve training efficiency. After download the [huggingface DROID datasets](https://huggingface.co/datasets/cadene/droid_1.0.1), you can run the following command to extract latent in parrallel:
```bash
accelerate launch dataset_example/extract_latent.py --droid_hf_path ${path to droid} --droid_output_path dataset_example/droid --svd_path ${path to svd}
```
The processed data will be saved at `dataset_example/droid`. The structure of this dataset should be same as `dataset_example/droid_subset`, we already included some trajectories in it.


(2) After extract the video latent, we can prepare dataset meta information, which create a json file include all items and calculate the normalization of states and actions, which are required during training.
```bash
python dataset_meta_info/create_meta_info.py --droid_output_path ${path to processed droid data} --dataset_name droid
```

### 🛸 (2) Launch training
After prepare the datasets, you can launch training. You can first test the environment with a small subset of droid we provided in the repo:
```bash
WANDB_MODE=offline accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset
```
Then you can launch the training process with whole dataset:
```bash
accelerate launch --main_process_port 29501 scripts/train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid
```

### 🛸 (3) <span style="color:red;">New</span>: Post-train world model on down-stream tasks
Pretrained world model may not accurate enough in contact-rich or deformable object tasks. Following the pipeline in (1)(2), you can also post-train world model on down-stream tasks as in paper [VLAW](https://arxiv.org/abs/2602.12063).

Post-trained ctrl-world can support long-horizon policy-in-the-loop rollout and generate realistic long videos. 
Some examples is shown in below. Starting form the same initial condition, we rollout policy in **both real world and world model** for 20 seconds. The Top row is real world and bottom row is world model. More videos [**here**](https://sites.google.com/view/vlaw-arxiv).
<p>
    <img src="synthetic_traj/gallery/post-trained_1.gif" alt="wild-data" width="40%" />
    <img src="synthetic_traj/gallery/post-trained_2.gif" alt="wild-data" width="40%" />
</p>
<p>
    
</p>


## Acknowledgement

Ctrl-World is developed from the opensourced video foundation model [Stable-Video-Diffusion](https://github.com/Stability-AI/generative-models). The VLA model used in this repo is from [openpi](https://github.com/Physical-Intelligence/openpi). We thank the authors for their efforts!


## Bibtex 
If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@article{guo2025ctrl,
  title={Ctrl-world: A controllable generative world model for robot manipulation},
  author={Guo, Yanjiang and Shi, Lucy Xiaoyang and Chen, Jianyu and Finn, Chelsea},
  journal={arXiv preprint arXiv:2510.10125},
  year={2025}
}
@article{guo2026vlaw,
  title={Vlaw: Iterative co-improvement of vision-language-action policy and world model},
  author={Guo, Yanjiang and Lee, Tony and Shi, Lucy Xiaoyang and Chen, Jianyu and Liang, Percy and Finn, Chelsea},
  journal={arXiv preprint arXiv:2602.12063},
  year={2026}
}
```
