# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', control_joint=0, density=100, use_omomo=False, 
                use_dpo=False,
                mask_type='original', dpo_root_dir=None, multi_text=False, select_sample=None, sample_strategy='best_worst',
                  dpo_final_metric_weight_list=None, dpo_final_metric_margin_list=None, 
                  dpo_final_threshold=0.03, dpo_random_threshold=0.02,
                  use_smpl=False, cond_mode=None, data_part='all', dpo_pair_diff_percentile=1.0):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, control_joint=control_joint, density=density, \
                       use_omomo=use_omomo, use_dpo=use_dpo, mask_type=mask_type, dpo_root_dir=dpo_root_dir, 
                        multi_text=multi_text, select_sample=select_sample, sample_strategy=sample_strategy,
                        dpo_final_metric_weight_list=dpo_final_metric_weight_list, dpo_final_metric_margin_list=dpo_final_metric_margin_list,
                        dpo_final_threshold=dpo_final_threshold, dpo_random_threshold=dpo_random_threshold, use_smpl=use_smpl, cond_mode=cond_mode, data_part=data_part,
                        dpo_pair_diff_percentile=dpo_pair_diff_percentile)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', control_joint=0, density=100,
                       use_omomo=False, use_dpo=False, mask_type='original', dpo_data_root=None, multi_text=False, select_sample=None, 
                       sample_strategy='best_worst', num_workers=4, 
                       dpo_final_metric_weight_list=None, dpo_final_metric_margin_list=None,
                       dpo_final_threshold=0.03, dpo_random_threshold=0.02,
                       use_smpl=False, cond_mode=None, data_part='all', dpo_pair_diff_percentile=1.0, shuffle=True):
    dataset = get_dataset(name, num_frames, split, hml_mode, control_joint, density, use_omomo,
                          use_dpo, mask_type, dpo_root_dir=dpo_data_root, multi_text=multi_text, 
                          select_sample=select_sample, sample_strategy=sample_strategy,
                          dpo_final_metric_weight_list=dpo_final_metric_weight_list, dpo_final_metric_margin_list=dpo_final_metric_margin_list,
                          dpo_final_threshold=dpo_final_threshold, dpo_random_threshold=dpo_random_threshold, use_smpl=use_smpl, cond_mode=cond_mode, data_part=data_part,
                          dpo_pair_diff_percentile=dpo_pair_diff_percentile)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True if not select_sample else False, collate_fn=collate,
    )

    return loader
