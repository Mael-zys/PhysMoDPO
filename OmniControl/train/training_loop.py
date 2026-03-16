# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import functools
import os
from copy import deepcopy

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval.eval_humanml import evaluation
import numpy as np
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data, eval_data, gt_data_for_eval=None, 
                 eval_data_best=None, eval_data_worst=None, eval_data_gt=None):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        # Note: ref_model will be created after loading checkpoint
        self.ref_model = None
        
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.warmup_steps = args.warmup_steps if hasattr(args, 'warmup_steps') else 0

        self.step = 0
        self.resume_step = 0
        self.current_epoch = 0  # Track current epoch for data iteration
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.sync_cuda = torch.cuda.is_available()
        
        # Will be set in _load_and_sync_parameters if resuming from same directory
        self.resuming_same_dir = False

        self._load_and_sync_parameters()
        
        # Get gradient clipping threshold
        self.grad_clip = args.grad_clip if hasattr(args, 'grad_clip') else 0.0
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            grad_clip=self.grad_clip,
        )
        
        if self.grad_clip > 0:
            logger.log(f"Gradient clipping enabled with threshold: {self.grad_clip}")

        self.opt = AdamW(
            self.mp_trainer.trainable_param, lr=self.lr, weight_decay=self.weight_decay
        )
        
        # If resuming from the same directory, load optimizer and other states
        if self.resuming_same_dir:
            self._load_optimizer_and_training_state()
        else:
            # If using warmup and starting from scratch (not resuming same dir), set initial lr to 0
            if self.warmup_steps > 0:
                for param_group in self.opt.param_groups:
                    param_group["lr"] = 0.0
                logger.log(f"Warmup enabled: Starting with lr=0, will warmup to lr={self.lr} over {self.warmup_steps} steps")

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.use_ddp = False
        self.ddp_model = self.model

        self.eval_data = eval_data
        self.eval_gt_data = gt_data_for_eval
        self.eval_data_best = eval_data_best
        self.eval_data_worst = eval_data_worst
        self.eval_data_gt = eval_data_gt
        self.best_eval_loss = float('inf')
        self.best_fid = {}
        self.best_control = {}
        self.best_simulation = {}
        self.best_power = {}
        self.best_jerk = {}
        self.best_m2t_score = {}
        self.best_m2m_score = {}
        self.best_fuse_score = {}

        self.tmr_forward = load_tmr_model_easy(self.device)
        
        # Metric names for DPO fuse score calculation
        # Order matches dpo_fusescore_selection_list: ['Control_l2', 'Skating Ratio', 'Simulation Error', 'Simulation Error Max', 'Power', 'Feet Height', 'Jerk']
        self.metric_names_for_fuse = ['Control_l2', 'Skating Ratio', 'Simulation Error', 'Simulation Error Max', 'Power', 'Feet Height', 'Jerk', 'M2T score', 'M2M score']
        # DPO training data metrics stats
        if self.args.use_dpo:
            import json
            metrics_stats_path = os.path.join(self.args.dpo_data_root, 'metric_stats.json')
            
            # map metric_names_for_fuse to metrics_stats keys
            self.metrics_map = {
                'Control_l2': 'control_l2_dist',
                'Skating Ratio': 'skating_ratio',
                'Simulation Error': 'simulation_error',
                'Simulation Error Max': 'simulation_error_max',
                'Power': 'power',
                'Feet Height': 'feet_height',
                'Jerk': 'jerk',
                'M2T score': 'm2t_score',
                'M2M score': 'm2m_score',
            }
            with open(metrics_stats_path, 'r') as f:
                self.metrics_stats = json.load(f)

        # Early stopping for Control_l2 metric
        self.early_stop_enabled = hasattr(args, 'early_stop_patience') and args.early_stop_patience > 0
        if self.early_stop_enabled:
            self.early_stop_patience = args.early_stop_patience
            self.early_stop_threshold = args.early_stop_threshold
            self.control_l2_baseline = {}  # Will be set per eval loader at step 0 evaluation
            self.early_stop_counter = {}  # Consecutive evaluations above threshold per loader
            self.should_early_stop = False
            logger.log(f"Early stopping enabled: patience={self.early_stop_patience}, threshold={self.early_stop_threshold*100:.1f}%")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            checkpoint_data = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev()
            )
            
            # Check if this is a new-style checkpoint with metadata
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                # New-style checkpoint with full state
                logger.log("Found new-style checkpoint with full training state.")
                self.model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
                
                # Check if resuming from the same directory - use absolute paths for comparison
                checkpoint_dir = os.path.abspath(os.path.dirname(resume_checkpoint))
                save_dir = os.path.abspath(self.save_dir)
                self.resuming_same_dir = (checkpoint_dir == save_dir)
                
                if self.resuming_same_dir:
                    logger.log(f"Resuming from same directory: {save_dir}")
                    logger.log("Will restore full training state (optimizer, step, random state, etc.)")
                    # Directly set self.step from checkpoint
                    if 'step' in checkpoint_data:
                        self.step = checkpoint_data['step']
                        logger.log(f"Loaded step from checkpoint: {self.step}")
                    else:
                        self.step = parse_resume_step_from_filename(resume_checkpoint)
                        logger.log(f"Parsed step from filename: {self.step}")
                    self.resume_step = 0  # No offset needed when resuming from same dir
                else:
                    logger.log(f"Resuming from different directory:")
                    logger.log(f"  Checkpoint dir: {checkpoint_dir}")
                    logger.log(f"  Output dir: {save_dir}")
                    logger.log("Will only load model parameters, not optimizer or training state.")
                    self.resume_step = 0
            else:
                # Old-style checkpoint with only model state_dict
                logger.log("Found old-style checkpoint (model parameters only).")
                self.model.load_state_dict(checkpoint_data, strict=False)
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
                self.resuming_same_dir = False
                logger.log(f"Parsed resume_step from filename: {self.resume_step}")
        
        # Create ref_model AFTER loading checkpoint so it has the same weights as model
        if self.args.use_dpo:
            logger.log("Creating reference model for DPO (deepcopy after loading checkpoint)...")
            self.ref_model = deepcopy(self.model)
            # self.ref_model.eval()  # disable random masking
            # Freeze ref_model parameters
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.log("Reference model created and frozen.")

    def _load_optimizer_and_training_state(self):
        """Load optimizer state and other training states when resuming from same directory."""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if not resume_checkpoint:
            return
        
        checkpoint_data = dist_util.load_state_dict(
            resume_checkpoint, map_location=dist_util.dev()
        )
        
        # Check if this is a new-style checkpoint
        if not isinstance(checkpoint_data, dict) or 'model_state_dict' not in checkpoint_data:
            logger.log("Old-style checkpoint detected, cannot restore optimizer and training state.")
            return
        
        missing_states = []
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint_data:
            logger.log("Loading optimizer state...")
            try:
                self.opt.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.log("✓ Optimizer state loaded successfully")
            except Exception as e:
                logger.log(f"✗ Failed to load optimizer state: {e}")
                missing_states.append('optimizer_state_dict')
        else:
            logger.log("✗ Optimizer state not found in checkpoint")
            missing_states.append('optimizer_state_dict')
        
        # Load step (already loaded in _load_and_sync_parameters, but verify)
        if 'step' in checkpoint_data:
            logger.log(f"✓ Step restored: {self.resume_step}")
        else:
            logger.log("✗ Step not found in checkpoint")
            missing_states.append('step')
        
        # Load best metrics
        if 'best_eval_loss' in checkpoint_data:
            self.best_eval_loss = checkpoint_data['best_eval_loss']
            logger.log(f"✓ Best eval loss restored: {self.best_eval_loss}")
        else:
            logger.log("✗ Best eval loss not found in checkpoint")
            missing_states.append('best_eval_loss')
        
        if 'best_fid' in checkpoint_data:
            self.best_fid = checkpoint_data['best_fid']
            logger.log(f"✓ Best FID restored: {self.best_fid}")
        else:
            logger.log("✗ Best FID not found in checkpoint")
            missing_states.append('best_fid')
        
        if 'best_control' in checkpoint_data:
            self.best_control = checkpoint_data['best_control']
            logger.log(f"✓ Best control metrics restored: {self.best_control}")
        else:
            logger.log("✗ Best control metrics not found in checkpoint")
            missing_states.append('best_control')
        
        if 'best_simulation' in checkpoint_data:
            self.best_simulation = checkpoint_data['best_simulation']
            logger.log(f"✓ Best simulation metrics restored: {self.best_simulation}")
        else:
            logger.log("✗ Best simulation metrics not found in checkpoint")
            missing_states.append('best_simulation')
        
        if 'best_fuse_score' in checkpoint_data:
            self.best_fuse_score = checkpoint_data['best_fuse_score']
            logger.log(f"✓ Best fuse score restored: {self.best_fuse_score}")
        else:
            logger.log("✗ Best fuse score not found in checkpoint")
            missing_states.append('best_fuse_score')
        
        if 'best_m2t_score' in checkpoint_data:
            self.best_m2t_score = checkpoint_data['best_m2t_score']
            logger.log(f"✓ Best M2T score restored: {self.best_m2t_score}")
        else:
            self.best_m2t_score = {}
            logger.log("✗ Best M2T score not found in checkpoint")
            missing_states.append('best_m2t_score')
        
        if 'best_m2m_score' in checkpoint_data:
            self.best_m2m_score = checkpoint_data['best_m2m_score']
            logger.log(f"✓ Best M2M score restored: {self.best_m2m_score}")
        else:
            self.best_m2m_score = {}
            logger.log("✗ Best M2M score not found in checkpoint")
            missing_states.append('best_m2m_score')
        
        # Load early stopping state
        if self.early_stop_enabled:
            if 'control_l2_baseline' in checkpoint_data:
                control_l2_baseline = checkpoint_data['control_l2_baseline']
                if isinstance(control_l2_baseline, dict):
                    self.control_l2_baseline = control_l2_baseline
                else:
                    self.control_l2_baseline = {'vald': control_l2_baseline}
                logger.log(f"✓ Control_l2 baseline restored: {self.control_l2_baseline}")
            else:
                logger.log("✗ Control_l2 baseline not found in checkpoint")
                missing_states.append('control_l2_baseline')
            
            if 'early_stop_counter' in checkpoint_data:
                early_stop_counter = checkpoint_data['early_stop_counter']
                if isinstance(early_stop_counter, dict):
                    self.early_stop_counter = early_stop_counter
                else:
                    self.early_stop_counter = {'vald': early_stop_counter}
                logger.log(f"✓ Early stop counter restored: {self.early_stop_counter}")
            else:
                logger.log("✗ Early stop counter not found in checkpoint")
                missing_states.append('early_stop_counter')

        # Load random states
        if 'random_state' in checkpoint_data:
            import random
            try:
                random.setstate(checkpoint_data['random_state']['python'])
                logger.log("✓ Python random state restored")
            except Exception as e:
                logger.log(f"✗ Failed to restore Python random state: {e}")
                missing_states.append('random_state.python')
        else:
            logger.log("✗ Python random state not found in checkpoint")
            missing_states.append('random_state')
        
        if 'numpy_random_state' in checkpoint_data:
            try:
                np.random.set_state(checkpoint_data['numpy_random_state'])
                logger.log("✓ NumPy random state restored")
            except Exception as e:
                logger.log(f"✗ Failed to restore NumPy random state: {e}")
                missing_states.append('numpy_random_state')
        else:
            logger.log("✗ NumPy random state not found in checkpoint")
            missing_states.append('numpy_random_state')
        
        if 'torch_random_state' in checkpoint_data:
            try:
                # Ensure the state is on CPU before setting
                state = checkpoint_data['torch_random_state']
                if state.is_cuda:
                    state = state.cpu()
                torch.set_rng_state(state)
                logger.log("✓ PyTorch CPU random state restored")
            except Exception as e:
                logger.log(f"✗ Failed to restore PyTorch CPU random state: {e}")
                missing_states.append('torch_random_state')
        else:
            logger.log("✗ PyTorch CPU random state not found in checkpoint")
            missing_states.append('torch_random_state')
        
        if 'torch_cuda_random_state' in checkpoint_data and torch.cuda.is_available():
            try:
                # Ensure the state is on CPU before setting
                state = checkpoint_data['torch_cuda_random_state']
                if state.is_cuda:
                    state = state.cpu()
                torch.cuda.set_rng_state(state)
                logger.log("✓ PyTorch CUDA random state restored")
            except Exception as e:
                logger.log(f"✗ Failed to restore PyTorch CUDA random state: {e}")
                missing_states.append('torch_cuda_random_state')
        elif torch.cuda.is_available():
            logger.log("✗ PyTorch CUDA random state not found in checkpoint")
            missing_states.append('torch_cuda_random_state')
        
        # Load epoch state
        if 'current_epoch' in checkpoint_data:
            self.current_epoch = checkpoint_data['current_epoch']
            logger.log(f"✓ Current epoch restored: {self.current_epoch}")
        else:
            logger.log("✗ Current epoch not found in checkpoint")
            missing_states.append('current_epoch')
        
        # Load DataLoader worker RNG states if available
        if 'dataloader_worker_rng_states' in checkpoint_data:
            # Store for later use when recreating DataLoader
            self.dataloader_worker_rng_states = checkpoint_data['dataloader_worker_rng_states']
            logger.log(f"✓ DataLoader worker RNG states restored ({len(self.dataloader_worker_rng_states)} workers)")
        else:
            logger.log("✗ DataLoader worker RNG states not found in checkpoint")
            missing_states.append('dataloader_worker_rng_states')
            self.dataloader_worker_rng_states = None
        
        # Print summary
        if missing_states:
            logger.log(f"\nWarning: The following states were not restored: {missing_states}")
        else:
            logger.log("\n✓ All training states restored successfully!")

    def run_loop(self):

        for epoch in range(self.current_epoch, self.num_epochs):
            print(f'Starting epoch {epoch}')
            self.current_epoch = epoch
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate(losses_only = not self.args.eval_during_training)
                    self.model.train()
                    
                    # Check early stopping
                    if self.early_stop_enabled and self.should_early_stop:
                        logger.log("Early stopping: Training terminated.")
                        print("\n🛑 Training stopped early due to Control_l2 degradation.")
                        return

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    @torch.no_grad()
    def evaluate(self, losses_only=False):
        if self.eval_data is None:
            logger.log("No evaluation data provided, skipping evaluation.")
            self.save(last=False)
            return
        print("Start evaluation.")
        if not losses_only:
            num_samples_limit = 256 - 1  # reduce for faster eval, None means no limit (eval over all dataset)
            run_mm = False
            mm_num_samples = 0
            mm_num_repeats = 0
            mm_num_times = 0
            diversity_times = 50 # reduce for faster eval
            replication_times = 1  # about 3 Hrs
            guidance_param = 2.5 # default value to evaluation
            # copy the model for classifier free sampling
            eval_model = ClassifierFreeSampleModel(deepcopy(self.model))
            
            log_file = os.path.join(self.save_dir, f'eval_{self.step}.log')
            log_folder = log_file.replace('.log','')
            os.makedirs(log_folder, exist_ok=True)

            eval_motion_loaders = {
                # ################
                # ## HumanML3D Dataset##
                # ################

                # # # only evaluate after simulation
                # # 'vald': lambda: get_mdm_loader(
                # #     eval_model, self.diffusion, 32,
                # #     self.eval_data, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                # #     num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from, generate_motion=False, 
                # # ),
                # 'vald_sim': lambda: get_mdm_loader(
                #     eval_model, self.diffusion, 32,
                #     self.eval_data, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                #     num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from, generate_motion=False, 
                #     eval_after_simulation=True, sim_gpu=self.args.sim_gpu, output_path=log_folder
                # )
            }

            if self.args.use_dpo:
                eval_motion_loaders['vald_sim'] = lambda: get_mdm_loader(
                        eval_model, self.diffusion, 32,
                        self.eval_data, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                        num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from, generate_motion=False, 
                        eval_after_simulation=True, sim_gpu=self.args.sim_gpu, output_path=log_folder
                )
            else:
                eval_motion_loaders['vald'] = lambda: get_mdm_loader(
                        eval_model, self.diffusion, 32,
                        self.eval_data, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                        num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from, generate_motion=False, 
                )


            if self.args.use_dpo:
                if self.eval_data_best is not None:
                    num_samples_limit = min(128, len(self.eval_data_best.dataset)) - 1  # reduce for faster eval, None means no limit (eval over all dataset)
                    diversity_times = min(50, len(self.eval_data_best.dataset))  # reduce for faster eval
                    eval_motion_loaders['vald_best'] = lambda: get_mdm_loader(
                        eval_model, self.diffusion, min(32, len(self.eval_data_best.dataset)),
                        self.eval_data_best, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                        num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from
                    )
                
                if self.eval_data_worst is not None:
                    num_samples_limit = min(128, len(self.eval_data_worst.dataset)) - 1  # reduce for faster eval, None means no limit (eval over all dataset)
                    diversity_times = min(50, len(self.eval_data_worst.dataset))  # reduce for faster eval
                    eval_motion_loaders['vald_worst'] = lambda: get_mdm_loader(
                        eval_model, self.diffusion, min(32, len(self.eval_data_worst.dataset)),
                        self.eval_data_worst, mm_num_samples, mm_num_repeats, self.eval_gt_data.dataset.opt.max_motion_length, 
                        num_samples_limit, guidance_param, use_smpl=self.args.use_smpl, joints_value_from=self.args.joints_value_from
                    )          

            eval_wrapper = EvaluatorMDMWrapper(self.args.dataset, dist_util.dev())
            mean_dict = evaluation(eval_wrapper, self.eval_gt_data, eval_motion_loaders, 
                                   os.path.join(self.save_dir, f'eval_{self.step}.log'), replication_times, 
                                   diversity_times, mm_num_times, run_mm=run_mm, train_platform=self.train_platform, 
                                   training_args=self.args, training_step=self.step,
                                   visualize=self.args.visualize_during_training,
                                   use_smpl=self.args.use_smpl, sim_gpu=self.args.sim_gpu if hasattr(self.args, 'sim_gpu') else 0, 
                                   tmr_forward=self.tmr_forward)
            
            for key in mean_dict:
                if 'vald_gt' in key or 'vald_best_gt' in key:
                    continue
                
                if isinstance(mean_dict[key], np.float64) or isinstance(mean_dict[key], np.float32):
                    self.train_platform.report_scalar(name=key, value=mean_dict[key], iteration=self.step, group_name='Evaluation Metrics')
                    if 'vald' in key:
                        if 'FID' in key and key not in self.best_fid:
                            self.best_fid[key] = float('inf')
                        if 'Control_l2' in key and key not in self.best_control:
                            self.best_control[key] = float('inf')
                        if 'Simulation' in key and 'Simulation Error Max' not in key and key not in self.best_simulation:
                            self.best_simulation[key] = float('inf')
                        if 'Power' in key and key not in self.best_power:
                            self.best_power[key] = float('inf')
                        if 'Jerk' in key and key not in self.best_jerk:
                            self.best_jerk[key] = float('inf')
                        if 'M2T score' in key and key not in self.best_m2t_score:
                            self.best_m2t_score[key] = -float('inf')  # M2T score: higher is better
                        if 'M2M score' in key and key not in self.best_m2m_score:
                            self.best_m2m_score[key] = -float('inf')  # M2M score: higher is better
                            
                        if 'FID' in key and mean_dict[key] < self.best_fid[key]:
                            self.best_fid[key] = mean_dict[key]
                            print(f"Best FID: {self.best_fid[key]:.4f}")
                            self.save(last=False, metrics=key)
                        if 'Control_l2' in key and mean_dict[key] < self.best_control[key]:
                            self.best_control[key] = mean_dict[key]
                            print(f"Best Control_l2 Score: {self.best_control[key]:.4f}")
                            self.save(last=False, metrics=key)
                        if 'Simulation' in key and 'Simulation Error Max' not in key and mean_dict[key] < self.best_simulation.get(key, float('inf')):
                            self.best_simulation[key] = mean_dict[key]
                            print(f"Best Simulation: {self.best_simulation[key]:.4f}")
                            self.save(last=False, metrics=key)
                        if 'M2T score' in key and mean_dict[key] > self.best_m2t_score[key]:  # Higher is better
                            self.best_m2t_score[key] = mean_dict[key]
                            print(f"Best M2T Score: {self.best_m2t_score[key]:.4f}")
                            self.save(last=False, metrics=key)
                        if 'M2M score' in key and mean_dict[key] > self.best_m2m_score[key]:  # Higher is better
                            self.best_m2m_score[key] = mean_dict[key]
                            print(f"Best M2M Score: {self.best_m2m_score[key]:.4f}")
                            self.save(last=False, metrics=key)
                        # if 'Power' in key and mean_dict[key] < self.best_power[key]:
                        #     self.best_power[key] = mean_dict[key]
                        #     print(f"Best Power: {self.best_power[key]:.4f}")
                        #     self.save(last=False, metrics=key)
                        # if 'Jerk' in key and mean_dict[key] < self.best_jerk[key]:
                        #     self.best_jerk[key] = mean_dict[key]
                        #     print(f"Best Jerk: {self.best_jerk[key]:.4f}")
                        #     self.save(last=False, metrics=key)

                elif 'Trajectory Error' in key:
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
                    for i in range(len(mean_dict[key])): # zip(traj_err_key, mean):
                        self.train_platform.report_scalar(name=f'{key}_{traj_err_key[i]}', value=mean_dict[key][i], iteration=self.step, group_name='Evaluation Metrics')
                
                elif isinstance(mean_dict[key], np.ndarray):
                    for i in range(len(mean_dict[key])):
                        self.train_platform.report_scalar(name=f'{key}_{i}', value=mean_dict[key][i], iteration=self.step, group_name='Evaluation Metrics')
            
            # Calculate fuse score when DPO is enabled (after processing all metrics)
            if self.args.use_dpo and hasattr(self.args, 'dpo_fusescore_selection_list'):
                fuse_score_loaders = [name for name in eval_motion_loaders.keys() if name in ('vald', 'vald_sim')]
                for loader_name in fuse_score_loaders:
                    # Extract metric values for fuse score calculation
                    metric_values = []
                    for metric_name in self.metric_names_for_fuse:
                        # Key format is typically: '{metric_name}_{loader_name}', e.g., 'Control_l2_vald'
                        matching_key = f"{metric_name}_{loader_name}"
                        if matching_key in mean_dict:
                            # normalize the metric value based on self.metrics_stats
                            if self.metrics_map[metric_name] in self.metrics_stats:
                                mean_val = self.metrics_stats[self.metrics_map[metric_name]]['mean']
                                std_val = self.metrics_stats[self.metrics_map[metric_name]]['std']
                                if std_val > 0:
                                    normalized_value = (mean_dict[matching_key] - mean_val) / std_val
                                else:
                                    normalized_value = mean_dict[matching_key]
                            else:
                                normalized_value = mean_dict[matching_key]

                            # for 'M2T score', 'M2M score' we want to maximize, so invert the value
                            if metric_name in ['M2T score', 'M2M score']:
                                metric_values.append(-normalized_value)  # Invert for maximization
                            else:
                                metric_values.append(normalized_value)
                        else:
                            metric_values.append(0.0)  # Use 0 if metric not found
                    
                    # Calculate fuse score using weights
                    if len(metric_values) == len(self.args.dpo_fusescore_selection_list):
                        fuse_score = sum(w * v for w, v in zip(self.args.dpo_fusescore_selection_list, metric_values))
                        
                        # Use a fixed key per loader for fuse score
                        fuse_score_key = f"fuse_score_{loader_name}"
                        
                        # Initialize best_fuse_score if not exists
                        if fuse_score_key not in self.best_fuse_score:
                            self.best_fuse_score[fuse_score_key] = float('inf')
                        
                        # Report fuse score to training platform
                        self.train_platform.report_scalar(name=fuse_score_key, value=fuse_score, iteration=self.step, group_name='Evaluation Metrics')
                        
                        # Save if this is the best (lowest) fuse score
                        if fuse_score < self.best_fuse_score[fuse_score_key]:
                            self.best_fuse_score[fuse_score_key] = fuse_score
                            print(f"Best Fuse Score ({loader_name}): {self.best_fuse_score[fuse_score_key]:.4f}")
                            print(f"  Metric values ({loader_name}): {dict(zip(self.metric_names_for_fuse, metric_values))}")
                            print(f"  Weights: {self.args.dpo_fusescore_selection_list}")
                            self.save(last=False, metrics=fuse_score_key)
            
            # Early stopping check for Control_l2
            if self.early_stop_enabled:
                early_stop_loaders = [name for name in eval_motion_loaders.keys() if name in ('vald', 'vald_sim')]
                for loader_name in early_stop_loaders:
                    # Find Control_l2 for each loader
                    control_l2_key = f"Control_l2_{loader_name}"
                    if control_l2_key in mean_dict:
                        current_control_l2 = mean_dict[control_l2_key]
                        baseline = self.control_l2_baseline.get(loader_name)
                        
                        # Set baseline at first evaluation (step 0 or first save_interval)
                        if baseline is None:
                            self.control_l2_baseline[loader_name] = current_control_l2
                            self.early_stop_counter[loader_name] = 0
                            logger.log(f"Early stopping baseline set ({loader_name}): Control_l2 = {self.control_l2_baseline[loader_name]:.4f}")
                            print(f"Early stopping baseline set ({loader_name}): Control_l2 = {self.control_l2_baseline[loader_name]:.4f}")
                        else:
                            # Check if current metric exceeds threshold
                            threshold_value = baseline * (1 + self.early_stop_threshold)
                            current_counter = self.early_stop_counter.get(loader_name, 0)
                            if current_control_l2 > threshold_value:
                                current_counter += 1
                                self.early_stop_counter[loader_name] = current_counter
                                logger.log(f"Early stopping ({loader_name}): Control_l2 ({current_control_l2:.4f}) > threshold ({threshold_value:.4f}), counter: {current_counter}/{self.early_stop_patience}")
                                print(f"⚠️  Early stopping warning ({loader_name}): Control_l2 ({current_control_l2:.4f}) > threshold ({threshold_value:.4f}), counter: {current_counter}/{self.early_stop_patience}")
                                
                                if current_counter >= self.early_stop_patience:
                                    self.should_early_stop = True
                                    logger.log(f"Early stopping triggered ({loader_name})! Control_l2 has been above threshold for {self.early_stop_patience} consecutive evaluations.")
                                    print(f"\n{'='*60}")
                                    print(f"🛑 EARLY STOPPING TRIGGERED ({loader_name})!")
                                    print(f"Control_l2 has been above threshold for {self.early_stop_patience} consecutive evaluations.")
                                    print(f"Baseline: {baseline:.4f}, Threshold: {threshold_value:.4f}, Current: {current_control_l2:.4f}")
                                    print(f"{'='*60}\n")
                            else:
                                # Reset counter if metric improves
                                if current_counter > 0:
                                    logger.log(f"Early stopping ({loader_name}): Control_l2 ({current_control_l2:.4f}) improved, resetting counter from {current_counter} to 0")
                                    print(f"✅ Early stopping counter reset ({loader_name}): Control_l2 ({current_control_l2:.4f}) < threshold ({threshold_value:.4f})")
                                self.early_stop_counter[loader_name] = 0


        total_loss = 0
        total_count = 0
        for motion, cond in tqdm(self.eval_data):
            batch = motion.to(self.device)
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            for i in range(0, batch.shape[0], 32):
                # Eliminates the microbatch feature
                assert i == 0
                micro = batch
                micro_cond = cond
                last_batch = (i + 32) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=self.eval_data.dataset
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )
                
                loss = (losses["loss"] * weights).mean()
                total_loss += loss.item() * batch.shape[0]
                total_count += batch.shape[0]

        print(f"Evaluation loss: {total_loss / total_count:.4f}")
        self.train_platform.report_scalar(name='loss', value=total_loss/total_count, iteration=self.step, group_name='Evaluation Loss')
        if total_loss / total_count < self.best_eval_loss:
            self.best_eval_loss = total_loss / total_count
            print(f"Best evaluation loss: {self.best_eval_loss:.4f}")
            self.save(last=False, metrics='eval_loss')
        return

    def run_step(self, batch, cond):
        self._anneal_lr()  # Set learning rate BEFORE training
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch # bs, 263, 1, seq_len
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            if self.args.use_dpo:
                # micro # (bs, 263*2, 1, seq_len) -> (bs, 263, 1, seq_len; bs, 263, 1, seq_len) -> (bs*2, 263, 1, seq_len)
                feat_num_dpo = micro.shape[1] // 3
                micro_win = micro[:, :feat_num_dpo].clone()
                micro_lose = micro[:, feat_num_dpo:feat_num_dpo*2].clone()

                # micro_noise_win = micro[:, 2*feat_num_dpo:3*feat_num_dpo].clone()
                # micro_noise_lose = micro[:, 3*feat_num_dpo:4*feat_num_dpo].clone()

                original_motion = micro[:, 2*feat_num_dpo:].clone()  # (bs, 263, 1, seq_len)
                original_motion = original_motion.repeat(2, 1, 1, 1)  # (bs*2, 263, 1, seq_len)
                
                micro = torch.cat([micro_win, micro_lose], dim=0)
                # micro_noise = torch.cat([micro_noise_win, micro_noise_lose], dim=0)
                micro_noise = None

                # Repeat condition for both win and lose samples
                micro_cond_new = {'y': {}}
                for k, v in micro_cond['y'].items():
                    if k == 'text':  # Special handling for text data
                        text_batch = v.copy()
                        text_batch_split_win = [text_sample.split(' ||| ')[0] for text_sample in text_batch]
                        text_batch_split_lose = [text_sample.split(' ||| ')[-1] for text_sample in text_batch]
                        micro_cond_new['y'][k] = text_batch_split_win + text_batch_split_lose
                        continue
                    if torch.is_tensor(v) and k != 'metric_ratio':
                        # Handle tensors based on their dimensions
                        if v.dim() == 4:
                            micro_cond_new['y'][k] = v.repeat(2, 1, 1, 1)
                        elif v.dim() == 3:
                            micro_cond_new['y'][k] = v.repeat(2, 1, 1)
                        elif v.dim() == 2:
                            micro_cond_new['y'][k] = v.repeat(2, 1)
                        elif v.dim() == 1:
                            micro_cond_new['y'][k] = v.repeat(2)
                        else:
                            micro_cond_new['y'][k] = v.repeat(2, *[1]*(v.dim()-1))
                    elif isinstance(v, list):
                        # Duplicate list for both win and lose
                        micro_cond_new['y'][k] = v + v
                    else:
                        # For other types (str, int, etc.), keep as is
                        micro_cond_new['y'][k] = v
                micro_cond = micro_cond_new
                
                # Also repeat timesteps and weights
                t = t.repeat(2)
                metrics_ratio = None
                if 'metric_ratio' in micro_cond['y']:
                    metrics_ratio = micro_cond['y']['metric_ratio']
                    if torch.is_tensor(metrics_ratio):
                        metrics_ratio = metrics_ratio.to(t.device).reshape(-1)
                    else:
                        metrics_ratio = torch.as_tensor(metrics_ratio, device=t.device)

                compute_losses = functools.partial(
                    self.diffusion.training_losses_dpo,
                    self.ddp_model,
                    self.ref_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=self.data.dataset,
                    noise=micro_noise,
                    sft_scale=self.args.sft_scale, beta_dpo=self.args.beta_dpo,
                    original_motion=original_motion,
                    metrics_ratio=1.0,
                    training_step=self.step,
                    dpo_loss_scale=self.args.dpo_loss_scale,
                    dpo_loss_type=getattr(self.args, 'dpo_loss_type', 'dpo'),
                )

                t = t[:micro.shape[0]//2]  # Use only the first half of t for logging
            else:
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=self.data.dataset
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            # print(loss.item())
            if torch.isnan(loss) or torch.isinf(loss):
                print("Skipping batch due to NaN loss")
                # import pdb
                # pdb.set_trace()
                continue 
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        """
        Apply learning rate schedule with optional warmup and annealing.
        
        Warmup phase (if warmup_steps > 0):
            - Linearly increase lr from 0 to initial lr over warmup_steps
        
        Annealing phase (if lr_anneal_steps > 0):
            - Linearly decrease lr from initial lr to 0 over lr_anneal_steps
        """
        current_step = self.step
        
        # Warmup phase: linearly increase learning rate from 0 to self.lr
        if not self.resuming_same_dir and self.warmup_steps > 0 and current_step < self.warmup_steps:
            # At step 0: warmup_frac = 0, lr = 0
            # At step warmup_steps-1: warmup_frac ≈ 1, lr ≈ self.lr
            warmup_frac = (current_step + 1) / self.warmup_steps  # +1 so we reach self.lr at warmup_steps
            warmup_frac = min(1.0, warmup_frac)  # Clamp to max 1.0
            lr = self.lr * warmup_frac
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            return
        
        current_step = self.step + self.resume_step
        # Constant phase: maintain self.lr if warmup is done but annealing hasn't started
        if self.warmup_steps > 0 and current_step < (self.lr_anneal_steps if self.lr_anneal_steps > 0 else float('inf')):
            # After warmup, before annealing
            for param_group in self.opt.param_groups:
                param_group["lr"] = self.lr
            return
        
        # Annealing phase: linearly decrease learning rate from self.lr to 0
        if not self.lr_anneal_steps:
            # No annealing, keep self.lr
            return
        
        frac_done = (current_step - self.warmup_steps) / (self.lr_anneal_steps - self.warmup_steps)
        frac_done = max(0.0, min(1.0, frac_done))  # Clamp to [0, 1]
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self, last=True, metrics=''):
        """Save complete training state including model, optimizer, step, random states, and best metrics."""
        import random
        
        save_name = 'last' if last else 'best'
        
        # Get model state dict
        model_state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        
        # Do not save CLIP weights
        clip_weights = [e for e in model_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del model_state_dict[e]
        
        # Prepare complete checkpoint data
        checkpoint_data = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.opt.state_dict(),
            'step': self.step + self.resume_step,
            'current_epoch': self.current_epoch,
            'best_eval_loss': self.best_eval_loss,
            'best_fid': self.best_fid,
            'best_control': self.best_control,
            'best_simulation': self.best_simulation,
            'best_m2t_score': self.best_m2t_score,
            'best_m2m_score': self.best_m2m_score,
            'best_fuse_score': self.best_fuse_score,
            'random_state': {
                'python': random.getstate(),
            },
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        
        # Add early stopping state if enabled
        if self.early_stop_enabled:
            checkpoint_data['control_l2_baseline'] = self.control_l2_baseline
            checkpoint_data['early_stop_counter'] = self.early_stop_counter
        
        # Add CUDA random state if available
        if torch.cuda.is_available():
            checkpoint_data['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        
        # Try to capture DataLoader worker RNG states
        # Note: This is best-effort and may not perfectly restore worker states
        try:
            if hasattr(self.data, 'worker_init_fn'):
                checkpoint_data['dataloader_worker_rng_states'] = {
                    'worker_init_fn': str(self.data.worker_init_fn)
                }
        except:
            pass  # DataLoader worker states are optional
        
        logger.log(f"Saving complete training state to {save_name}{metrics}.pt...")
        logger.log(f"  Step: {checkpoint_data['step']}")
        logger.log(f"  Epoch: {checkpoint_data['current_epoch']}")
        logger.log(f"  Best eval loss: {self.best_eval_loss}")
        logger.log(f"  Best FID: {self.best_fid}")
        logger.log(f"  Best control: {self.best_control}")
        logger.log(f"  Best simulation: {self.best_simulation}")
        logger.log(f"  Best M2T score: {self.best_m2t_score}")
        logger.log(f"  Best M2M score: {self.best_m2m_score}")
        logger.log(f"  Best fuse score: {self.best_fuse_score}")
        
        # Save complete checkpoint
        checkpoint_path = bf.join(self.save_dir, f'model_{save_name}{metrics}.pt')
        with bf.BlobFile(checkpoint_path, "wb") as f:
            torch.save(checkpoint_data, f)
        
        logger.log(f"✓ Checkpoint saved successfully to {checkpoint_path}")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
