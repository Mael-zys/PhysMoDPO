from torch.utils.data import DataLoader, Dataset
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import CompMDMGeneratedDataset

import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch


def collate_fn(batch):
    # Filter out entries that are entirely None.
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    batch.sort(key=lambda x: x[3], reverse=True)
    
    # Handle fields in the batch that may contain None values (for example,
    # the hint field) by processing each field independently.
    batch_items = list(zip(*batch))
    collated = []
    
    for i, items in enumerate(batch_items):
        # Check whether all elements are None.
        if all(item is None for item in items):
            collated.append(None)
        # Check whether some elements are None.
        elif any(item is None for item in items):
            # Keep fields with None values as lists.
            collated.append(list(items))
        else:
            # Use default_collate when the field contains no None values.
            collated.append(default_collate(items))
    
    return collated


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            # We don't need the following logic because our sample func generates the full tensor anyway:
            # if len(motion) < self.opt.max_motion_length:
            #     motion = np.concatenate([motion,
            #                              np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
            #                              ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


# our loader
def get_mdm_loader(model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, 
                   max_motion_length, num_samples_limit, scale, generate_motion=True, use_smpl=False, joints_value_from='joints',
                   eval_after_simulation=False, output_path=None, sim_gpu=0, eval_mode='omnicontrol',
                   maskedmimic_init_mode='zero_pose'):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # dataset = CompMDMGeneratedDataset(opt, ground_truth_dataset, ground_truth_dataset.w_vectorizer, mm_num_samples, mm_num_repeats)
    if eval_mode == 'omnicontrol':
        dataset = CompMDMGeneratedDataset(model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, 
                                      max_motion_length, num_samples_limit, scale, generate_motion, 
                                      use_smpl=use_smpl, joints_value_from=joints_value_from,
                                      eval_after_simulation=eval_after_simulation, output_path=output_path, sim_gpu=sim_gpu)

    mm_dataset = MMGeneratedDataset(opt, dataset, ground_truth_loader.dataset.w_vectorizer)

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True if batch_size == 32 else False, num_workers=min(4, batch_size))
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader
