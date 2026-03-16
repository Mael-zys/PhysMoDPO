import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_util
from tools.guofeats import joints_to_guofeats
from tools.extract_joints import extract_joints, extract_joints_batch
from tools.smplrifke_feats import smplrifkefeats_to_smpldata_batch
import os
import subprocess 
import yaml
import numpy as np
from tools.fix_fps import interpolate_fps_poses, interpolate_fps_trans
from utils.runtime_paths import PROTO_MOTIONS_ROOT, resolve_omnicontrol_path


def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))
    
class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length,
                 num_samples_limit, scale=1., generate_motion=True, use_smpl=False, joints_value_from='joints', 
                 eval_after_simulation=False, output_path=None, sim_gpu=0):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        
        self.use_smpl = use_smpl
        self.output_format = 'hml_vec' # or 'smpl'
        self.output_gt_joints = False # output gt joints for visualization purpose
        self.eval_after_simulation = eval_after_simulation
        self.output_path = output_path

        if self.eval_after_simulation and self.use_smpl:
            os.makedirs(self.output_path, exist_ok=True)

        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        
        if joints_value_from == 'smpl' or self.eval_after_simulation:
            from tools.smpl_layer import SMPLH
            self.smplh_dict = {
                gender: SMPLH(
                    path="body_models/smplh",
                    jointstype="both",
                    input_pose_rep="axisangle",
                    gender=gender,
                )
                for gender in ["neutral", "male", "female"]
            }
            for smpl_layer in self.smplh_dict.values():
                smpl_layer = smpl_layer.eval()
                # Freeze SMPL layer parameters but allow gradient flow
                for param in smpl_layer.parameters():
                    param.requires_grad = False

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                for k, v in model_kwargs['y'].items():
                    if torch.is_tensor(v):
                        model_kwargs['y'][k] = v.to(dist_util.dev())

                if num_samples_limit is not None and len(generated_motion) > num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    gt_sample = motion.to(dist_util.dev()).clone() # only for visualization purpose

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    original_joints = None
                    gt_joints_temp = None
                    smpldata = {}
                    if use_smpl:
                        sample = sample.permute(0, 3, 2, 1).contiguous()
                        sample = sample.squeeze(2)
                        sample = sample * torch.from_numpy(self.dataset.std).to(sample.device) + torch.from_numpy(self.dataset.mean).to(sample.device)

                        smpldata = smplrifkefeats_to_smpldata_batch(sample.to(dist_util.dev()).clone(), first_angle=np.pi)

                        if self.eval_after_simulation:
                            os.makedirs(os.path.join(self.output_path, f'{i}_{t}', 'amass_format', 'ik'), exist_ok=True)
                            for bs_i in range(dataloader.batch_size):
                                length = model_kwargs['y']['lengths'][bs_i]
                                save_path = os.path.join(self.output_path, f'{i}_{t}', 'amass_format', 'ik', f'ik_{bs_i}.npz')
                                np.savez(save_path, poses=smpldata['poses'][bs_i].cpu().numpy()[:length], 
                                    trans=smpldata['trans'][bs_i].cpu().numpy()[:length],
                                    betas=np.zeros(10), num_betas=10, gender='neutral',
                                        mocap_frame_rate=20, text=model_kwargs['y']['text'][bs_i],
                                        hint=model_kwargs['y']['hint'][bs_i].cpu().numpy() if 'hint' in model_kwargs['y'] else None)

                            absolute_output_path = resolve_omnicontrol_path(self.output_path, f'{i}_{t}')

                            # use specified GPU for simulation evaluation
                            gpu_id = sim_gpu
                            subprocess.call(
                                ["bash", str(PROTO_MOTIONS_ROOT / "run_deepmimic.sh"), absolute_output_path, str(gpu_id)],
                                cwd=str(PROTO_MOTIONS_ROOT),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )

                            simulation_motion_folder = os.path.join(absolute_output_path, 'deepmimic_output')
                            mask_mimic_id_list = yaml.load(open(os.path.join(absolute_output_path, 'amass_format', 'data_list.yaml'), "r"), Loader=yaml.FullLoader)["motions"]
                            mask_mimic_name2inx = {}
                            for mask_mimic_file in mask_mimic_id_list:
                                mask_mimic_file_name = mask_mimic_file['file'].split('/')[-1].replace('.npy', '')
                                mask_mimic_idx = mask_mimic_file['idx']
                                mask_mimic_name2inx[mask_mimic_file_name] = mask_mimic_idx

                            model_kwargs['y']['lengths'] -= 2
                            for bs_i in range(dataloader.batch_size):
                                length = model_kwargs['y']['lengths'][bs_i]
                                maskmimic_id = mask_mimic_name2inx[f'ik_{bs_i}']
                                maskmimic_file = os.path.join(simulation_motion_folder, f'trajectory_pose_aa_{maskmimic_id}_0.npz')
                                sim_smpldata = np.load(maskmimic_file)
                                
                                smpldata['poses'][bs_i,:length,:] = interpolate_fps_poses(torch.from_numpy(sim_smpldata['pose'][0]).float(), 30.0, 20.0)[:length,:66].to(sample.device)
                                smpldata['trans'][bs_i,:length,:] = interpolate_fps_trans(torch.from_numpy(sim_smpldata['trans'][0]).float(), 30.0, 20.0)[:length].to(sample.device)

                        gt_sample = gt_sample.permute(0, 3, 2, 1).contiguous()
                        gt_sample = gt_sample.squeeze(2)
                        gt_sample = gt_sample * torch.from_numpy(self.dataset.std).to(gt_sample.device) + torch.from_numpy(self.dataset.mean).to(gt_sample.device)

                        from tools.smpl_layer import SMPLH
                        if self.eval_after_simulation:
                            joints = extract_joints_batch(smpldata.copy(),
                                            "smpldata",
                                            fps=20,
                                            value_from="smpl",
                                            smpl_layer=self.smplh_dict['neutral'], joints_only=True)
                        else:
                            joints = extract_joints_batch(sample.clone(),
                                            "smplrifke",
                                            fps=20,
                                            value_from=joints_value_from,
                                            smpl_layer=self.smplh_dict['neutral'] if joints_value_from == 'smpl' or self.eval_after_simulation else None, joints_only=True)
                        gt_joints = extract_joints_batch(gt_sample.clone(),
                                        "smplrifke",
                                        fps=20,
                                        value_from=joints_value_from,
                                        smpl_layer=self.smplh_dict['neutral'] if joints_value_from == 'smpl' or self.eval_after_simulation else None, joints_only=True)
                            
                        joints = joints["joints"]
                        gt_joints = gt_joints["joints"]

                        original_joints = joints.copy()
                        gt_joints_temp = gt_joints.copy()

                        # convert back to hml_vec for evaluation
                        x, y, z = T(joints)
                        motions = T(np.stack((x, z, -y), axis=0))

                        bs, seq_len = motions.shape[0], motions.shape[1]
                        output = np.zeros((bs, 1, seq_len, 263))
                        for bs_i in range(dataloader.batch_size):
                            length = model_kwargs['y']['lengths'][bs_i]
                            output[bs_i,0,:length-1] = joints_to_guofeats(motions[bs_i][:length])

                        sample = torch.from_numpy(output).to(dist_util.dev()).float()
                        sample = sample.permute(0,1,3,2)


                        # convert GT back to hml_vec for evaluation
                        x, y, z = T(gt_joints)
                        motions = T(np.stack((x, z, -y), axis=0))

                        bs, seq_len = motions.shape[0], motions.shape[1]
                        output = np.zeros((bs, 1, seq_len, 263))
                        for bs_i in range(dataloader.batch_size):
                            length = model_kwargs['y']['lengths'][bs_i]
                            output[bs_i,0,:length-1] = joints_to_guofeats(motions[bs_i][:length])

                        gt_sample = torch.from_numpy(output).to(dist_util.dev()).float()
                        gt_sample = gt_sample.permute(0,1,3,2)

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                      'gt_motion': gt_sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'hint': model_kwargs['y']['hint'][bs_i].cpu().numpy() if 'hint' in model_kwargs['y'] else None,
                                    'tokens': tokens[bs_i],
                                    'cap_len': tokens[bs_i].index('eos/OTHER') + 1,
                                    'original_joints': original_joints[bs_i] if original_joints is not None else None,
                                    'gt_joints': gt_joints_temp[bs_i] if gt_joints_temp is not None else None,
                                    'poses': smpldata['poses'][bs_i].cpu().numpy() if 'poses' in smpldata else None,
                                    'trans': smpldata['trans'][bs_i].cpu().numpy() if 'trans' in smpldata else None,
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        hint = data['hint'] if 'hint' in data else None
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.dataset.mode == 'eval' and self.output_format == 'hml_vec':
            normed_motion = motion
            if self.use_smpl:
                denormed_motion = motion
                m_length -= 1  # because guofeats has one less frame
            else:
                denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

            if self.output_gt_joints:
                gt_motion = data['gt_motion']
                if self.use_smpl:
                    gt_denormed_motion = gt_motion
                else:
                    gt_denormed_motion = self.dataset.t2m_dataset.inv_transform(gt_motion)
                gt_renormed_motion = (gt_denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
                gt_motion = gt_renormed_motion
                motion = np.concatenate([motion, gt_motion], axis=0)
        else:
            motion = data['original_joints'] # output original smpl output for control eval
            if self.output_gt_joints:
                gt_motion = data['gt_joints']
                motion = np.concatenate([motion, gt_motion], axis=0)  # first half is gen, second half is gt
            
            # dirty codes, temporarily make use of word_embeddings and pos_one_hots to store smpl data
            if 'poses' in data and data['poses'] is not None:
                word_embeddings = data['poses']
            if 'trans' in data and data['trans'] is not None:
                pos_one_hots = data['trans']
                
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint
