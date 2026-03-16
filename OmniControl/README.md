# Finetuning OmniControl with PhysMoDPO 

# Description
This codebase is a modified version from [OmniControl](https://github.com/neu-vi/OmniControl/tree/main) to perform DPO data generation and finetuning for [PhysMoDPO](https://github.com/Mael-zys/PhysMoDPO) project.

## Table of Content
* [Installation](#installation): environment and pretrained models
* [Training](#train): pretraining and PhysMoDPO training
* [Evaluation](#evaluation): evaluation with SMPL, G1 and H1 robots.


# Installation

This code requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)
* All the scripts should be executed in this folder
* Please first setup the [prerequisites](https://github.com/Mael-zys/PhysMoDPO?tab=readme-ov-file#prerequisites) before training and evaluation.

## Environment

Install ffmpeg for visualization(if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```

Setup conda env:
```bash
cd OmniControl
conda env create -f environment.yml
conda activate omnicontrol
python -m spacy download en_core_web_sm
```

## Download dependencies:

Download smpl, glove and t2m evaluators:
```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

## Dataset setup

For experiments with OmniControl baseline, we follow [stmc](https://github.com/nv-tlabs/stmc?tab=readme-ov-file#motions) project to process data into SMPL-based representation to facilitate SMPL mesh visualization and full-body motion tracking.

Please download the processed [HumanML3D data](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQB7qLs_a1uRQogJTeA-vTKJASGsI43giNUYGLLE_igwf0g?e=zuUQt4) and [OMOMO data](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQB7qLs_a1uRQogJTeA-vTKJASGsI43giNUYGLLE_igwf0g?e=zuUQt4) 

Or follow the detail instructions [stmc](https://github.com/nv-tlabs/stmc?tab=readme-ov-file#motions) project to process the raw dataset.


## Pretrained models

Download and put [ckpt files](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQASt8p_-5iGT45Fanej3YPuARI8cM5pdj-q-7m8vww5cGk?e=hch8oU) in ```save/ckpt``` folder:
- OmniControl original control setting
    - Pretraining ckpt on HumanML3D dataset: `pretraining_smpl_original.pt`
    - DPO finetuned on HumanML3D dataset: `DPO_hml3d_original.pt`
    - DPO finetuned on OMOMO dataset: `DPO_omomo_original.pt`
- OmniControl cross control setting
    - Pretraining ckpt on HumanML3D dataset: `pretraining_smpl_cross.pt`
    - DPO finetuned on HumanML3D dataset: `DPO_hml3d_cross.pt`
    - DPO finetuned on OMOMO dataset: `DPO_omomo_cross.pt`

# Training

## Pretraining with SMPL format data

As we change the data format, we train OmniControl from scratch.

For the original control setting (training with random one joint), please run with the following script to train your own model:
```bash
bash script/train_smpl.sh
```

For the cross control setting (training with random number of joints), please change mask_type to `--mask_type 'cross'` to train from scratch or finetune previous model with cross setting with the following script to train your own model:
```bash
bash script/train_smpl_cross.sh
```


## Preference data generation

Now we generate DPO data with the previous models:
```bash
bash script/dpo_data_generation.sh
```

- In this example script, we use the model to generate 12 samples for each input prompt and the generated data can be found in ```save/omnicontrol_smpl/inference_rep12_cross```.

- Note that, we apply physics-based method here to calculate the rewards, therefore, we remove the motions which require object support.

- To generate data on OMOMO dataset, please add `--omomo`.

## PhysMoDPO finetuning

Based on the rewards, we run the training script to select the preference pair and then perform DPO finetuning:
```bash
bash script/train_dpo.sh
```

- To perform finetuning on OMOMO dataset, please add `--omomo` and change `--dpo_data_root` to the corresponding generated folder.

- Note that, this generation-finetuning can be done iteratively. The provided ckpt is finetuned with 3 rounds.

# Evaluation
## Evaluation on SMPL robot
Change `model_path` to your ckpt path and run this script for cross control evaluation on HumanML3D dataset:
```bash
bash script/eval_smpl.sh
```
- This will first generate motion with the pretrained model, and then run motion tracking with SMPL robot in simulator. We will evaluate all the metrics on the human motions after tracking.
- Add `--omomo` for evaluation on OMOMO dataset


## Zero-shot evaluation

First download the retargeted GT test motion on G1 and H1 format for the FID and TMR evaluation metrics. Put them into ```save/cross_GT_hint``` and ```save/cross_GT_OMOMO_hint``` folder
- [Retargeted HumanML3D](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQClWWQlOmQISabJGhtHKVLwAQGmXrU2QVtz8AosXoNHfqQ?e=l7QOIK)
- [Retargeted OMOMO](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQDgPIjLNxbvQovXNx8xfHR0AXgoeq1ODF8t32n0tnqWwZ4?e=sG0wGu)

### G1 evaluation

- Copy the absolute path of the evaluation folder with SMPL robot, such as ```/home/xxx/PhysMoDPO/OmniControl/save/omnicontrol_smpl/eval_humanml__last_gscale2.5_omnicontrol_ddim_masktypecross_joint0_density100_condboth_text_spatial_partflat_ground_humanml3d_2026-02-18_20-22-03```. 

- We will re-use the human motion files before SMPL simulation, and then retarget them to g1 for the evaluation. Please run the following script with your own absolute path 
```bash
bash script/eval_g1.sh \
/home/xxx/PhysMoDPO/OmniControl/save/omnicontrol_smpl/eval_humanml__last_gscale2.5_omnicontrol_ddim_masktypecross_joint0_density100_condboth_text_spatial_partflat_ground_humanml3d_2026-02-18_20-22-03  \
save/cross_GT_hint
```

For evaluation on OMOMO dataset, please change both the input folder and the GT test folder with your own path.

### H1 evaluation

coming soon

# References
This project repository builds upon [OmniControl](https://github.com/neu-vi/OmniControl/tree/main).
