import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

import json
import math
from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from ..scripts.motion_process import recover_root_rot_pos, recover_from_ric
from data_loaders.humanml.utils.metrics import cross_combination_joints, cross_combination_joints_my
from tools.extract_joints import extract_joints
from tools.guofeats import joints_to_guofeats
# import spacy

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))
    
def collate_fn(batch):
    if batch[0][-1] is None:
        batch = [b[:-1] for b in batch]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode, control_joint=0, density=100, multi_text=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.ignore_max_motion_length = opt.ignore_max_motion_length
        self.mode = mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' and not opt.ignore_max_motion_length else 24
        self.control_joint = control_joint
        self.density = density
        self.mask_type = opt.mask_type
        self.multi_text = multi_text

        data_dict = {}
        id_list = []
        self.split_file = split_file
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        max_len_filter = 201 if self.opt.use_smpl else 200 # as smpl data has one more frame, this is to make sure we use the same data for fair comparison
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))[:, :263] # for behave data
                if (len(motion)) < min_motion_len or (len(motion) >= max_len_filter and not self.ignore_max_motion_length):
                    continue
                motion = motion[:199] # as smpl data has one more frame
                text_data = []
                flag = False
                skip_flag_flat_ground = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        if line.strip() == '' or line.strip().startswith('#'):
                            print(f"empty line or comment line in {pjoin(opt.text_dir, name + '.txt')}, continue")
                            continue
                        line_split = line.strip().split('#')
                        caption = line_split[0]

                        if opt.data_part == 'flat_ground':
                            # further filter
                            object_support_action = [' sits ', ' sit ', ' sitting ', ' seat ', 
                                                     'upstairs', 'downstairs', 'chair', ' bench ',
                                                     ' ladder ', ' climb ', ' climbing ', ' climbs ', ' stairs']
                            if any([x in caption.lower() for x in object_support_action]):
                                skip_flag_flat_ground = True
                                break

                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= max_len_filter and not self.ignore_max_motion_length):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break
                
                if opt.data_part == 'flat_ground' and skip_flag_flat_ground:
                    continue
                
                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        # Currently omomo data is only used for testing with smpl
        if 'HumanML3D' in opt.data_root or 'omomo' in opt.data_root:
            if self.opt.use_smpl:
                spatial_norm_path = './dataset/HumanML3D_amass'
            else:
                spatial_norm_path = './dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            if self.opt.use_smpl:
                raise NotImplementedError('SMPL not supported for KIT dataset yet')
            else:
                spatial_norm_path = './dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')
        print('dataset: ', spatial_norm_path)
        print('number of samples for %s: %d'%(mode, len(name_list)))
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        # assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def random_mask_cross_my_random(self, joints, n_joints=22, density=1):
        length = joints.shape[0]
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        
        mask_seq = np.zeros((length, n_joints, 1)).astype(np.bool)

        ratio = choose_seq_num / length
        mask_seq[:,[0, 10, 11, 15, 20, 21], :] = np.random.rand(length, 6, 1) < ratio

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_cross_my_each(self, joints, n_joints=22, density=1):
        # cross_joints = cross_combination_joints()
        cross_joints = cross_combination_joints_my()
        choose = np.random.choice(len(cross_joints), 1).item()
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)
        for cj in choose_joint:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_cross_my(self, joints, n_joints=22, density=1):
        # cross_joints = cross_combination_joints()
        cross_joints = cross_combination_joints_my()
        choose = np.random.choice(len(cross_joints), 1).item()
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_cross(self, joints, n_joints=22, density=1):
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask(self, joints, n_joints=22, density=1):
        if n_joints != 21:
            # humanml3d
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            # kit
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        choose_joint = [self.control_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        # density = 100
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train(self, joints, n_joints=22):
        if n_joints != 21:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        num_joints = len(controllable_joints)
        # joints: length, 22, 3
        num_joints_control = np.random.choice(num_joints, 1)
        # only use one joint during training
        num_joints_control = 1
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = controllable_joints[choose_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_hands(self, joints, n_joints=22, density=1):
        choose_joint = [20,21]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        # density = 100
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_hands_train(self, joints, n_joints=22):
        choose_joint = [20,21]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_train_cross_my_random(self, joints, n_joints=22):
        length = joints.shape[0]
        
        mask_seq = np.zeros((length, n_joints, 1)).astype(np.bool)

        mask_seq[:,[0, 10, 11, 15, 20, 21], :] = np.random.rand(length, 6, 1) < 0.5

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_train_cross_my_each(self, joints, n_joints=22):
        cross_joints = cross_combination_joints_my()
        choose = np.random.choice(len(cross_joints), 1).item()
        # choose = -1
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            choose_seq_num = np.random.choice(length - 1, 1) + 1
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_train_cross_my(self, joints, n_joints=22):
        cross_joints = cross_combination_joints_my()
        choose = np.random.choice(len(cross_joints), 1).item()
        # choose = -1
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask_train_cross(self, joints, n_joints=22):
        from data_loaders.humanml.utils.metrics import cross_combination_joints
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        # choose = -1
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        if self.opt.use_smpl:
            n_joints = 24
            joints = extract_joints(torch.from_numpy(motion).float(), 
                                    "smplrifke",
                                    fps=20,
                                    value_from="joints",
                                    smpl_layer=None,)
            joints = joints["joints"]
        else:
            n_joints = 22 if motion.shape[-1] == 263 else 21
            # hint is global position of the controllable joints
            joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
            joints = joints.numpy()

        if 'spatial' not in self.opt.cond_mode:
            hint = None
        else:
            # control any joints at any time
            # note that for validation, we use the same masking strategy as training
            if self.mode == 'train' or 'val' in self.split_file:
                if self.mask_type == 'random':
                    hint = self.random_mask_train(joints, n_joints)
                elif self.mask_type == 'cross':
                    # hint = self.random_mask_train_cross_my(joints, n_joints)
                    hint = self.random_mask_train_cross(joints, n_joints)
                elif self.mask_type == 'hands':
                    hint = self.random_mask_hands_train(joints, n_joints)
                else:
                    hint = self.random_mask_train(joints, n_joints)
            else:
                if self.mask_type == 'random':
                    hint = self.random_mask_train(joints, n_joints)
                elif self.mask_type == 'cross':
                    # hint = self.random_mask_cross_my(joints, n_joints)
                    hint = self.random_mask_cross(joints, n_joints)
                elif self.mask_type == 'hands':
                    hint = self.random_mask_hands(joints, n_joints)
                else:
                    hint = self.random_mask(joints, n_joints)

            hint = hint.reshape(hint.shape[0], -1)
            if m_length < self.max_motion_length:
                hint = np.concatenate([hint,
                                    np.zeros((self.max_motion_length - m_length, hint.shape[1]))
                                        ], axis=0)

        if self.opt.use_smpl and self.mode == 'gt':
            # only used in evaluation to compute FID/CLIP scores
            # convert joints to guofeats
            x, y, z = T(joints)
            temp = T(np.stack((x, z, -y), axis=0))
            motion = joints_to_guofeats(temp[:m_length])
            m_length -= 1

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-8)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        # only for generation
        if self.multi_text:
            caption_all = []
            for text_data in text_list:
                caption, tokens = text_data['caption'], text_data['tokens']
                caption_all.append(caption)
            caption = ' ||| '.join(caption_all)
        
        if 'text' not in self.opt.cond_mode:
            caption = ''
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint


class Text2MotionDatasetV2_DPO(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode, control_joint=0,
                 density=100, dpo_root_dir='', multi_text=False, select_sample=None, sample_strategy='best_worst', 
                 omnicontrol_metric_filename='omnicontrol_sorted.json',
                 allow_sparse_metric_reps=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.ignore_max_motion_length = opt.ignore_max_motion_length
        self.mode = mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' and not opt.ignore_max_motion_length else 24
        self.control_joint = control_joint
        self.density = density
        self.mask_type = opt.mask_type
        self.multi_text = multi_text
        self.select_sample = select_sample
        self.sample_strategy = sample_strategy

        data_dict = {}
        self.dpo_root_dir = dpo_root_dir
        motion_dir = os.path.join(dpo_root_dir, 'omnicontrol_output')
        omnicontrol_metric_json = os.path.join(dpo_root_dir, omnicontrol_metric_filename)
        deepmimic_metric_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_gt_err_0.json')
        deepmimic_metric_max_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_gt_err_max_0.json')
        power_metric_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_power_0.json')

        final_metric_list = ['mean_error', 'skate_ratio', 'dp_mpjpe', 'dp_mpjpe_max', 'power', 'feet_height', 'jerk', 'm2t_score', 'm2m_score']
        # Read hyperparameters from opt
        final_metric_weight_list = getattr(opt, 'dpo_final_metric_weight_list', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        final_metric_margin_list = getattr(opt, 'dpo_final_metric_margin_list', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        final_threshold = getattr(opt, 'dpo_final_threshold', 0.03)
        random_threshold = getattr(opt, 'dpo_random_threshold', 0.02)
        # win_sample_percentile: only consider samples in the top X% as potential winners
        win_sample_percentile = getattr(opt, 'dpo_win_sample_percentile', 1.0)
        # dpo_pair_diff_percentile: keep top X% of pairs by pair_diff (0-1, 1 keeps all)
        pair_diff_percentile = getattr(opt, 'dpo_pair_diff_percentile', 1.0)
        if pair_diff_percentile is None:
            pair_diff_percentile = 1.0
        pair_diff_percentile = min(max(pair_diff_percentile, 0.0), 1.0)

        # load json file
        with open(omnicontrol_metric_json, 'r') as f:
            omnicontrol_metric = json.load(f)

        has_deepmimic_metric = os.path.exists(deepmimic_metric_json)
        has_deepmimic_metric_max = os.path.exists(deepmimic_metric_max_json)
        has_power_metric = os.path.exists(power_metric_json)
        deepmimic_metric = {}
        deepmimic_metric_max = {}
        power_metric = {}

        if has_deepmimic_metric:
            with open(deepmimic_metric_json, 'r') as f:
                deepmimic_metric = json.load(f)
        
        if has_deepmimic_metric_max:
            with open(deepmimic_metric_max_json, 'r') as f:
                deepmimic_metric_max = json.load(f)
        
        if has_power_metric:
            with open(power_metric_json, 'r') as f:
                power_metric = json.load(f)

        omnicontrol_only_metrics = {'mean_error', 'skate_ratio', 'feet_height', 'jerk', 'm2t_score', 'm2m_score'}

        def metric_available(metric):
            if metric in omnicontrol_only_metrics:
                return metric in omnicontrol_metric
            if metric == 'dp_mpjpe':
                return (metric in omnicontrol_metric or has_deepmimic_metric)
            if metric == 'dp_mpjpe_max':
                return (metric in omnicontrol_metric or has_deepmimic_metric_max)
            if metric == 'power':
                return (metric in omnicontrol_metric or has_power_metric)
            return False

        def get_metric_value(metric, idx_name, sample_name, rep_idx):
            if metric in ['mean_error', 'skate_ratio', 'feet_height', 'jerk', 'm2t_score']:
                if metric not in omnicontrol_metric:
                    return None
                if allow_sparse_metric_reps and idx_name not in omnicontrol_metric[metric]:
                    return None
                metric_value = round(float(omnicontrol_metric[metric][idx_name]), 5)
                if metric == 'm2t_score':
                    metric_value = -metric_value
                return metric_value

            if metric == 'm2m_score':
                if metric not in omnicontrol_metric:
                    return None
                if idx_name not in omnicontrol_metric[metric]:
                    # Legacy behavior keeps missing m2m at zero; in sparse G1 mode, skip missing reps.
                    return None if allow_sparse_metric_reps else 0
                return -round(float(omnicontrol_metric[metric][idx_name]), 5)

            if metric == 'dp_mpjpe':
                if metric in omnicontrol_metric:
                    if idx_name not in omnicontrol_metric[metric]:
                        if allow_sparse_metric_reps:
                            return None
                    return round(float(omnicontrol_metric[metric][idx_name]), 5)
                deepmimic_key = f'ik_{sample_name}_{rep_idx}'
                if allow_sparse_metric_reps and deepmimic_key not in deepmimic_metric:
                    return None
                return round(float(deepmimic_metric[deepmimic_key]), 5)

            if metric == 'dp_mpjpe_max':
                if metric in omnicontrol_metric:
                    if idx_name not in omnicontrol_metric[metric]:
                        if allow_sparse_metric_reps:
                            return None
                    return round(float(omnicontrol_metric[metric][idx_name]), 5)
                deepmimic_max_key = f'ik_{sample_name}_{rep_idx}'
                if allow_sparse_metric_reps and deepmimic_max_key not in deepmimic_metric_max:
                    return None
                return round(float(deepmimic_metric_max[deepmimic_max_key]), 5)

            if metric == 'power':
                if metric in omnicontrol_metric:
                    if idx_name not in omnicontrol_metric[metric]:
                        if allow_sparse_metric_reps:
                            return None
                    return round(float(omnicontrol_metric[metric][idx_name]), 5)
                power_key = f'ik_{sample_name}_{rep_idx}'
                if allow_sparse_metric_reps and power_key not in power_metric:
                    return None
                return round(float(power_metric[power_key]), 5)

            raise NotImplementedError('unknown metric')

        id_list = [os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if f.endswith('.npy')]

        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                batch_data = np.load(name, allow_pickle=True).item()
                num_repetitions = batch_data['num_repetitions']
                batch_size = batch_data['output_vectors'].shape[0] // num_repetitions
                
                for bid in range(batch_size):
                    name = bid + batch_data['dataset_start_idx']
                    
                    final_score_list = []
                    data_pair = []

                    try:
                        if opt.data_part == 'flat_ground':
                            # further filter
                            object_support_action = [' sits ', ' sit ', ' sitting ', ' seat ', 
                                                        'upstairs', 'downstairs', 'chair', ' bench ',
                                                        ' ladder ', ' climb ', ' climbing ', ' climbs ', ' stairs']
                            caption = batch_data['text'][bid]
                            if any([x in caption.lower() for x in object_support_action]):
                                continue

                        # Store individual metric values for each sample
                        metric_values_per_sample = []
                        valid_rep_indices = []
                        best_dp_mpjpe_max = float('inf')
                        has_dp_mpjpe_max_value = False
                        for i in range(num_repetitions):
                            idx_name = f'sample_sample_{name}_rep_{i}'
                            sample_metrics = {}
                            
                            for metric in final_metric_list:
                                if not metric_available(metric):
                                    continue

                                metric_value = get_metric_value(metric, idx_name, name, i)
                                if metric_value is None:
                                    continue

                                sample_metrics[metric] = metric_value
                                if metric == 'dp_mpjpe_max' and metric_value < best_dp_mpjpe_max:
                                    best_dp_mpjpe_max = metric_value
                                    has_dp_mpjpe_max_value = True

                            if allow_sparse_metric_reps and len(sample_metrics) == 0:
                                continue

                            metric_values_per_sample.append(sample_metrics)
                            valid_rep_indices.append(i)
                        
                        for sample_metrics in metric_values_per_sample:
                            score_idx_name = 0
                            for mettric_idx, metric in enumerate(final_metric_list):
                                if metric in sample_metrics:
                                    weight = final_metric_weight_list[mettric_idx]
                                    score_idx_name += weight * sample_metrics[metric]
                            final_score_list.append(score_idx_name)

                        # remove samples that cannot be followed by deepmimic
                        if allow_sparse_metric_reps:
                            if has_dp_mpjpe_max_value and best_dp_mpjpe_max > 0.5:
                                print("skip sample %d due to best dp_mpjpe_max %.4f"%(name, best_dp_mpjpe_max))
                                continue
                        else:
                            if ('dp_mpjpe_max' in omnicontrol_metric or has_deepmimic_metric_max) and best_dp_mpjpe_max > 0.5:
                                print("skip sample %d due to best dp_mpjpe_max %.4f"%(name, best_dp_mpjpe_max))
                                continue
                    
                    except:
                        continue
                    
                    if allow_sparse_metric_reps and len(metric_values_per_sample) == 0:
                        continue

                    # add thresholding to filter similar motions
                    if allow_sparse_metric_reps:
                        if len(final_score_list) < 2 or max(final_score_list) - min(final_score_list) < final_threshold:
                            continue
                    else:
                        if max(final_score_list) - min(final_score_list) < final_threshold:
                            continue
                    
                    if sample_strategy != 'best_worst':
                        # generate all valid pairs where winner is better in ALL metrics
                        mpjpe_res_sorted = sorted(enumerate(final_score_list), key=lambda x: x[1])
                        
                        # Calculate the cutoff index for win samples based on percentile
                        max_win_idx = max(1, int(len(mpjpe_res_sorted) * win_sample_percentile))
                        
                        pos_ptr = 0  
                        
                        # collect unique indices that will be used
                        used_indices = set()

                        
                        while pos_ptr < max_win_idx:  # Only consider top X% as potential winners
                            pos_idx = mpjpe_res_sorted[pos_ptr][0]
                            pos_value = mpjpe_res_sorted[pos_ptr][1]
                            pos_metrics = metric_values_per_sample[pos_idx]
                            
                            neg_ptr = len(mpjpe_res_sorted) - 1
                            while neg_ptr > pos_ptr:
                                neg_idx = mpjpe_res_sorted[neg_ptr][0]
                                neg_value = mpjpe_res_sorted[neg_ptr][1]
                                neg_metrics = metric_values_per_sample[neg_idx]
                                
                                # Check if pos sample is better in ALL metrics with threshold
                                all_metrics_better = True
                                has_compared_metric = False
                                for mettric_idx, metric in enumerate(final_metric_list):
                                    if not metric_available(metric):
                                        continue

                                    if allow_sparse_metric_reps:
                                        if metric not in pos_metrics or metric not in neg_metrics:
                                            continue
                                        pos_metric_value = pos_metrics[metric]
                                        neg_metric_value = neg_metrics[metric]
                                    else:
                                        pos_metric_value = pos_metrics.get(metric, float('inf'))
                                        neg_metric_value = neg_metrics.get(metric, float('inf'))

                                    has_compared_metric = True
                                    # Lower is better for all metrics
                                    # Winner must be better than loser by at least random_threshold * weight
                                    if pos_metric_value >= neg_metric_value - final_metric_margin_list[mettric_idx]:
                                        all_metrics_better = False
                                        break

                                if allow_sparse_metric_reps and not has_compared_metric:
                                    all_metrics_better = False
                                
                                if all_metrics_better and neg_value - pos_value >= random_threshold:
                                    data_pair.append((pos_idx, neg_idx))
                                    used_indices.add(pos_idx)
                                    used_indices.add(neg_idx)
                                neg_ptr -= 1
                            
                            pos_ptr += 1

                        # only store selected indices data
                        if len(data_pair) > 0:
                            # create index mapping from old to new
                            used_indices_list = sorted(list(used_indices))
                            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices_list)}
                            
                            # only keep selected motions, texts and scores
                            selected_rep_indices = [valid_rep_indices[idx] for idx in used_indices_list]
                            selected_motions = batch_data['output_vectors'][bid::batch_size][selected_rep_indices]
                            selected_scores = [final_score_list[idx] for idx in used_indices_list]
                            
                            # remap data_pair indices
                            remapped_data_pair = [(old_to_new_idx[pos], old_to_new_idx[neg]) for pos, neg in data_pair]
                            remapped_pair_diffs = [selected_scores[neg_new] - selected_scores[pos_new] for pos_new, neg_new in remapped_data_pair]

                            if pair_diff_percentile < 1.0:
                                if pair_diff_percentile <= 0.0:
                                    continue
                                sorted_pairs = sorted(
                                    zip(remapped_data_pair, remapped_pair_diffs),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                                keep_count = int(math.ceil(len(sorted_pairs) * pair_diff_percentile))
                                if keep_count <= 0:
                                    continue
                                sorted_pairs = sorted_pairs[:keep_count]
                                remapped_data_pair, remapped_pair_diffs = zip(*sorted_pairs)
                                remapped_data_pair = list(remapped_data_pair)
                                remapped_pair_diffs = list(remapped_pair_diffs)
                            
                            if self.multi_text:
                                selected_texts = [batch_data['text'][bid::batch_size][idx] for idx in selected_rep_indices]
                            else:
                                selected_texts = batch_data['text'][bid]
                            
                            data_dict[name] = {'motion': selected_motions,
                                            #    'noise': batch_data['noises'][bid::batch_size],
                                                'length': batch_data['lengths'][bid],
                                                'text': selected_texts,
                                                'tokens': batch_data['tokens'][bid],
                                                'hint': batch_data['hint'][bid],
                                                'original_motion': batch_data['original_motion'][bid] if select_sample is None or select_sample == 'gt' else None,
                                                'data_pair': remapped_data_pair,
                                                'pair_diffs': remapped_pair_diffs,
                                                'final_score_list': selected_scores,
                            }
                        else:
                            # skip if no valid pairs
                            continue
                    
                    else:
                        # For best_worst strategy, find all valid pairs where winner is better in ALL metrics
                        # Then select the pair with maximum total score difference
                        mpjpe_res_sorted = sorted(enumerate(final_score_list), key=lambda x: x[1])
                        
                        best_pair = None
                        max_score_diff = -1
                        
                        # Iterate through all possible pairs
                        for pos_ptr in range(len(mpjpe_res_sorted)):
                            pos_idx = mpjpe_res_sorted[pos_ptr][0]
                            pos_value = mpjpe_res_sorted[pos_ptr][1]
                            pos_metrics = metric_values_per_sample[pos_idx]
                            
                            for neg_ptr in range(len(mpjpe_res_sorted) - 1, pos_ptr, -1):
                                neg_idx = mpjpe_res_sorted[neg_ptr][0]
                                neg_value = mpjpe_res_sorted[neg_ptr][1]
                                neg_metrics = metric_values_per_sample[neg_idx]
                                
                                # Check if pos sample is better in ALL metrics with threshold
                                all_metrics_better = True
                                has_compared_metric = False
                                for mettric_idx, metric in enumerate(final_metric_list):
                                    if not metric_available(metric):
                                        continue

                                    if allow_sparse_metric_reps:
                                        if metric not in pos_metrics or metric not in neg_metrics:
                                            continue
                                        pos_metric_value = pos_metrics[metric]
                                        neg_metric_value = neg_metrics[metric]
                                    else:
                                        pos_metric_value = pos_metrics.get(metric, float('inf'))
                                        neg_metric_value = neg_metrics.get(metric, float('inf'))

                                    has_compared_metric = True
                                    # Lower is better for all metrics
                                    # Winner must be better than loser by at least final_threshold * weight
                                    if pos_metric_value >= neg_metric_value - final_metric_margin_list[mettric_idx]:
                                        all_metrics_better = False
                                        break

                                if allow_sparse_metric_reps and not has_compared_metric:
                                    all_metrics_better = False
                                
                                # If this pair is valid and has larger score difference, update best_pair
                                if all_metrics_better:
                                    score_diff = neg_value - pos_value
                                    if score_diff > max_score_diff and score_diff >= final_threshold:
                                        max_score_diff = score_diff
                                        best_pair = (pos_idx, neg_idx)
                        
                        # Only add to dataset if we found a valid pair
                        if best_pair is None:
                            continue
                        
                        select_ids_winner, select_ids_loser = best_pair
                        best_worst_diff = max_score_diff
                        selected_rep_indices = [valid_rep_indices[select_ids_winner], valid_rep_indices[select_ids_loser]]
                        
                        # only keep the win and lose sample into the dataset
                        data_dict[name] = {'motion': batch_data['output_vectors'][bid::batch_size][selected_rep_indices],
                                        #    'noise': batch_data['noises'][bid::batch_size],
                                            'length': batch_data['lengths'][bid],
                                            'text': [batch_data['text'][bid::batch_size][idx] for idx in selected_rep_indices] if self.multi_text else batch_data['text'][bid],
                                            'tokens': batch_data['tokens'][bid],
                                            'hint': batch_data['hint'][bid],
                                            'original_motion': batch_data['original_motion'][bid] if select_sample is None or select_sample == 'gt' else None,
                                            'data_pair': [(0, 1)],
                                            'pair_diffs': [best_worst_diff],
                                            'final_score_list': [final_score_list[select_ids_winner], final_score_list[select_ids_loser]],
                        }

                    new_name_list.append(name)
                    length_list.append(batch_data['lengths'][bid])
            except:
                print('error in loading ', name)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        # Currently omomo data is only used for testing with smpl
        if 'HumanML3D' in opt.data_root or 'omomo' in opt.data_root:
            if self.opt.use_smpl:
                spatial_norm_path = './dataset/HumanML3D_amass'
            else:
                spatial_norm_path = './dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            if self.opt.use_smpl:
                raise NotImplementedError('SMPL not supported for KIT dataset yet')
            else:
                spatial_norm_path = './dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')
        print('dataset: ', spatial_norm_path)
        print('number of samples for %s: %d'%(mode, len(name_list)))
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        # assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def _select_pair(self, data):
        data_pair = data['data_pair']
        if len(data_pair) == 0:
            raise ValueError("No data_pair available for sampling.")
        return random.choice(data_pair)
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        select_ids_winner = None
        select_ids_loser = None
        hint, m_length, tokens = data['hint'], data['length'], data['tokens']
        tokens = tokens.split('_')

        pos_one_hots = []
        word_embeddings = []
        sent_len = len(tokens)
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.sample_strategy != 'best_worst':
            select_ids_winner, select_ids_loser = self._select_pair(data)
            # indices are already remapped, so we can use them directly
        else:
            # use best and worst
            if select_ids_winner is None or select_ids_loser is None:
                select_ids_winner = np.argmin(data['final_score_list']) 
                select_ids_loser = np.argmax(data['final_score_list'])
            
        metrics1 = data['final_score_list'][select_ids_winner]
        metrics2 = data['final_score_list'][select_ids_loser]
            # print(metrics1, metrics2, select_ids_winner, select_ids_loser)

        motion_winner = data['motion'][select_ids_winner]
        motion_loser = data['motion'][select_ids_loser]
        # noise_winner = data['noise'][select_ids_winner]
        # noise_loser = data['noise'][select_ids_loser]

        metrics_ratio = metrics2 - metrics1

        if self.select_sample == 'best':
            motion = motion_winner
        elif self.select_sample == 'worst':
            motion = motion_loser
        elif self.select_sample == 'gt':
            original_motion = data['original_motion']
            motion = original_motion
        else:
            original_motion = data['original_motion']
            # motion = np.concatenate([motion_winner, motion_loser, noise_winner, noise_loser, original_motion], axis=1)
            motion = np.concatenate([motion_winner, motion_loser, original_motion], axis=1)

        if self.opt.use_smpl:
            n_joints = 24
        else:
            n_joints = 22 if motion_winner.shape[-1] == 263 else 21

        if 'spatial' not in self.opt.cond_mode:
            hint = None
        else:
            mask_seq = (hint.reshape(self.max_motion_length, n_joints, 3) != 0).any(-1, keepdims=True)
            hint = (hint - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
            hint = hint * mask_seq
            hint = hint.reshape(hint.shape[0], -1)

        if self.multi_text:
            caption_winner = data['text'][select_ids_winner]
            caption_loser = data['text'][select_ids_loser]
            if self.select_sample == 'best':
                caption = caption_winner
            elif self.select_sample == 'worst':
                caption = caption_loser
            elif self.select_sample == 'gt':
                caption = caption_winner
            else:
                caption = caption_winner + ' ||| ' + caption_loser
        else:
            caption = data['text']

        if 'text' not in self.opt.cond_mode:
            caption = ''
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), metrics_ratio, hint


class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None, None


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, density=100, use_omomo=False, \
                 use_dpo=False,
                 mask_type='original', dpo_root_dir='', multi_text=False, select_sample=None, 
                 sample_strategy='best_worst', dpo_final_metric_weight_list=None,
                 dpo_final_metric_margin_list=None, dpo_final_threshold=0.03, dpo_random_threshold=0.02, dpo_win_sample_percentile=0.3,
                 dpo_pair_diff_percentile=1.0,
                 use_smpl=False, cond_mode=None, data_part='all', **kwargs):
        self.mode = mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.mask_type = mask_type
        opt.ignore_max_motion_length = False
        opt.cond_mode=cond_mode
        opt.data_part=data_part
        
        # DPO hyperparameters
        if dpo_final_metric_weight_list is None:
            dpo_final_metric_weight_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        opt.dpo_final_metric_weight_list = dpo_final_metric_weight_list
        if dpo_final_metric_margin_list is None:
            dpo_final_metric_margin_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        opt.dpo_final_metric_margin_list = dpo_final_metric_margin_list
        opt.dpo_final_threshold = dpo_final_threshold
        opt.dpo_random_threshold = dpo_random_threshold
        opt.dpo_win_sample_percentile = dpo_win_sample_percentile
        opt.dpo_pair_diff_percentile = dpo_pair_diff_percentile

        opt.use_smpl = False
        if use_smpl:
            opt.data_root = 'dataset/HumanML3D_amass'
            if use_omomo:
                opt.motion_dir = 'dataset/omomo_amass/new_joint_vecs'
                opt.text_dir = 'dataset/omomo_amass/texts'
            else:
                opt.motion_dir = 'dataset/HumanML3D_amass/new_joint_vecs'
                opt.text_dir = 'dataset/HumanML3D_amass/texts'
            opt.use_smpl = True

        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)
        print('data: ', opt.data_root, opt.motion_dir, opt.text_dir)
        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        if use_smpl:
            # omomo data
            if use_omomo:
                self.split_file = pjoin('dataset/omomo_amass', f'{split}.txt')
            # humanml3d data
            elif data_part == 'flat_ground':
                self.split_file = pjoin('dataset/HumanML3D_amass', f'{split}_final.txt')
            else:
                # only use amass part
                self.split_file = pjoin('dataset/HumanML3D_amass', f'{split}.txt')
        else:
            if data_part == 'flat_ground':
                self.split_file = pjoin(opt.data_root, f'{split}_final.txt')
            elif data_part == 'amass':
                self.split_file = pjoin(opt.data_root, f'{split}_filtered.txt')
            else:
                self.split_file = pjoin(opt.data_root, f'{split}.txt')
        
        
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            if use_omomo and not use_smpl:
                print('we only finetune with dpo data for omomo')
                exit()
            elif use_dpo:
                self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2MotionDatasetV2_DPO(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode,
                                                            control_joint, density, dpo_root_dir, multi_text, select_sample, sample_strategy)
                self.num_actions = 1 # dummy placeholder
            else:
                self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode, control_joint, density, multi_text)
                self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 0, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)
