import os
import json
from tqdm import tqdm
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Select best and worst metric')
    parser.add_argument('--dpo_root_dir', type=str, 
                        default='save/omnicontrol_ckpt/inference_rep10_original_fix_wo_ik_new_multi',
                        help='Root directory for DPO inference results')
    return parser.parse_args()

args = parse_args()
dpo_root_dir = args.dpo_root_dir
motion_dir = os.path.join(dpo_root_dir, 'omnicontrol_output')
omnicontrol_metric_json = os.path.join(dpo_root_dir, 'omnicontrol_sorted.json')
deepmimic_metric_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_gt_err_0.json')
deepmimic_metric_max_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_gt_err_max_0.json')
power_metric_json = os.path.join(dpo_root_dir, 'deepmimic_output/all_motions_with_power_0.json')


# load json file
with open(omnicontrol_metric_json, 'r') as f:
    omnicontrol_metric = json.load(f)

deepmimic_metric = {}
if os.path.exists(deepmimic_metric_json):
    with open(deepmimic_metric_json, 'r') as f:
        deepmimic_metric = json.load(f)

deepmimic_metric_max = {}
if os.path.exists(deepmimic_metric_max_json):
    with open(deepmimic_metric_max_json, 'r') as f:
        deepmimic_metric_max = json.load(f)

power_metric = {}
if os.path.exists(power_metric_json):
    with open(power_metric_json, 'r') as f:
        power_metric = json.load(f)

id_list = [os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if f.endswith('.npy')]

sliding_res_list = []
mpjpe_res_list = []
dp_mpjpe_res_list = []
dp_mpjpe_max_res_list = []
power_res_list = []
feet_height_res_list = []
jerk_res_list = []
m2t_res_list = []
m2m_res_list = []

# Check if feet_height and jerk exist in omnicontrol_metric
has_feet_height = 'feet_height' in omnicontrol_metric
has_jerk = 'jerk' in omnicontrol_metric
has_m2t = 'm2t_score' in omnicontrol_metric
has_m2m = 'm2m_score' in omnicontrol_metric
has_dp_mpjpe = 'dp_mpjpe' in omnicontrol_metric or os.path.exists(deepmimic_metric_json)
has_dp_mpjpe_max = 'dp_mpjpe_max' in omnicontrol_metric or os.path.exists(deepmimic_metric_max_json)
has_power = 'power' in omnicontrol_metric or os.path.exists(power_metric_json)

def _best_worst_diff(res_array, higher_is_better=False):
    if higher_is_better:
        best_vals = np.nanmax(res_array, axis=1)
        worst_vals = np.nanmin(res_array, axis=1)
        diff_vals = best_vals - worst_vals
    else:
        best_vals = np.nanmin(res_array, axis=1)
        worst_vals = np.nanmax(res_array, axis=1)
        diff_vals = worst_vals - best_vals
    return best_vals, worst_vals, diff_vals

def _extreme_indices(diff_vals):
    valid = np.isfinite(diff_vals)
    if not np.any(valid):
        return None, None
    valid_indices = np.nonzero(valid)[0]
    sorted_indices = valid_indices[np.argsort(diff_vals[valid])]
    return sorted_indices[:1], sorted_indices[-1:]

for name in tqdm(id_list):
    try:
        batch_data = np.load(name, allow_pickle=True).item()
        num_repetitions = batch_data['num_repetitions']
        batch_size = batch_data['output_vectors'].shape[0] // num_repetitions
        
        for bid in range(batch_size):
            name = bid + batch_data['dataset_start_idx']
            sliding_res = []
            mpjpe_res = []
            dp_mpjpe_res = []
            dp_mpjpe_max_res = []
            power_res = []
            feet_height_res = []
            jerk_res = []
            m2t_res = []
            m2m_res = []
            
            for i in range(num_repetitions):
                idx_name = f'sample_sample_{name}_rep_{i}'
                dp_idx_name = f'ik_{name}_{i}'
                sliding_res.append(round(float(omnicontrol_metric['skate_ratio'][idx_name]), 5))
                mpjpe_res.append(round(float(omnicontrol_metric['mean_error'][idx_name]), 5))
                if has_dp_mpjpe:
                    if 'dp_mpjpe' in omnicontrol_metric and idx_name in omnicontrol_metric['dp_mpjpe']:
                        dp_mpjpe_res.append(round(float(omnicontrol_metric['dp_mpjpe'][idx_name]), 5))
                    elif dp_idx_name in deepmimic_metric:
                        dp_mpjpe_res.append(round(float(deepmimic_metric[dp_idx_name]), 5))
                    else:
                        dp_mpjpe_res.append(np.nan)
                if has_dp_mpjpe_max:
                    if 'dp_mpjpe_max' in omnicontrol_metric and idx_name in omnicontrol_metric['dp_mpjpe_max']:
                        dp_mpjpe_max_res.append(round(float(omnicontrol_metric['dp_mpjpe_max'][idx_name]), 5))
                    elif dp_idx_name in deepmimic_metric_max:
                        dp_mpjpe_max_res.append(round(float(deepmimic_metric_max[dp_idx_name]), 5))
                    else:
                        dp_mpjpe_max_res.append(np.nan)
                if has_power:
                    if 'power' in omnicontrol_metric and idx_name in omnicontrol_metric['power']:
                        power_res.append(round(float(omnicontrol_metric['power'][idx_name]), 5))
                    elif dp_idx_name in power_metric:
                        power_res.append(round(float(power_metric[dp_idx_name]), 5))
                    else:
                        power_res.append(np.nan)
                if has_feet_height:
                    feet_height_res.append(round(float(omnicontrol_metric['feet_height'].get(idx_name, np.nan)), 5))
                if has_jerk:
                    jerk_res.append(round(float(omnicontrol_metric['jerk'].get(idx_name, np.nan)), 5))
                if has_m2t:
                    m2t_res.append(round(float(omnicontrol_metric['m2t_score'].get(idx_name, np.nan)), 5))
                if has_m2m:
                    m2m_res.append(round(float(omnicontrol_metric['m2m_score'].get(idx_name, np.nan)), 5))
            
            sliding_res_list.append(sliding_res)
            mpjpe_res_list.append(mpjpe_res)
            if has_dp_mpjpe:
                dp_mpjpe_res_list.append(dp_mpjpe_res)
            if has_dp_mpjpe_max:
                dp_mpjpe_max_res_list.append(dp_mpjpe_max_res)
            if has_power:
                power_res_list.append(power_res)
            if has_feet_height:
                feet_height_res_list.append(feet_height_res)
            if has_jerk:
                jerk_res_list.append(jerk_res)
            if has_m2t:
                m2t_res_list.append(m2t_res)
            if has_m2m:
                m2m_res_list.append(m2m_res)
    except:
        pass

# convert to numpy array
sliding_res_array = np.array(sliding_res_list)  # (num_samples, num_repetitions)
mpjpe_res_array = np.array(mpjpe_res_list)  # (num_samples, num_repetitions)
if has_dp_mpjpe:
    dp_mpjpe_res_array = np.array(dp_mpjpe_res_list)  # (num_samples, num_repetitions)
if has_dp_mpjpe_max:
    dp_mpjpe_max_res_array = np.array(dp_mpjpe_max_res_list)  # (num_samples, num_repetitions)
if has_power:
    power_res_array = np.array(power_res_list)  # (num_samples, num_repetitions)
if has_feet_height:
    feet_height_res_array = np.array(feet_height_res_list)  # (num_samples, num_repetitions)
if has_jerk:
    jerk_res_array = np.array(jerk_res_list)  # (num_samples, num_repetitions)
if has_m2t:
    m2t_res_array = np.array(m2t_res_list)  # (num_samples, num_repetitions)
if has_m2m:
    m2m_res_array = np.array(m2m_res_list)  # (num_samples, num_repetitions)

# select best and worst res for each sample
best_sliding_ids, worst_sliding_ids, sliding_diffs = _best_worst_diff(sliding_res_array)
best_mpjpe_ids, worst_mpjpe_ids, mpjpe_diffs = _best_worst_diff(mpjpe_res_array)
if has_dp_mpjpe:
    best_dp_mpjpe_ids, worst_dp_mpjpe_ids, dp_mpjpe_diffs = _best_worst_diff(dp_mpjpe_res_array)
if has_dp_mpjpe_max:
    best_dp_mpjpe_max_ids, worst_dp_mpjpe_max_ids, dp_mpjpe_max_diffs = _best_worst_diff(dp_mpjpe_max_res_array)
if has_power:
    best_power_ids, worst_power_ids, power_diffs = _best_worst_diff(power_res_array)
if has_feet_height:
    best_feet_height_ids, worst_feet_height_ids, feet_height_diffs = _best_worst_diff(feet_height_res_array)
if has_jerk:
    best_jerk_ids, worst_jerk_ids, jerk_diffs = _best_worst_diff(jerk_res_array)
if has_m2t:
    best_m2t_ids, worst_m2t_ids, m2t_diffs = _best_worst_diff(m2t_res_array, higher_is_better=True)
if has_m2m:
    best_m2m_ids, worst_m2m_ids, m2m_diffs = _best_worst_diff(m2m_res_array, higher_is_better=True)

print('Best and worst sliding mean differences:', round(np.nanmean(sliding_diffs), 4), 'difference median:', round(np.nanmedian(sliding_diffs), 4), 'best mean:', round(np.nanmean(best_sliding_ids), 4), 'worst mean:', round(np.nanmean(worst_sliding_ids), 4), 'mean of all samples:', round(np.nanmean(sliding_res_array), 4))
print('Best and worst MPJPE mean differences:', round(np.nanmean(mpjpe_diffs), 4), 'difference median:', round(np.nanmedian(mpjpe_diffs), 4), 'best mean:', round(np.nanmean(best_mpjpe_ids), 4), 'worst mean:', round(np.nanmean(worst_mpjpe_ids), 4), 'mean of all samples:', round(np.nanmean(mpjpe_res_array), 4))
if has_dp_mpjpe and np.isfinite(dp_mpjpe_res_array).any():
    print('Best and worst DeepMimic MPJPE mean differences:', round(np.nanmean(dp_mpjpe_diffs), 4), 'median:', round(np.nanmedian(dp_mpjpe_diffs), 4), 'best mean:', round(np.nanmean(best_dp_mpjpe_ids), 4), 'worst mean:', round(np.nanmean(worst_dp_mpjpe_ids), 4), 'mean of all samples:', round(np.nanmean(dp_mpjpe_res_array), 4))
if has_dp_mpjpe_max and np.isfinite(dp_mpjpe_max_res_array).any():
    print('Best and worst DeepMimic MPJPE Max mean differences:', round(np.nanmean(dp_mpjpe_max_diffs), 4), 'median:', round(np.nanmedian(dp_mpjpe_max_diffs), 4), 'best mean:', round(np.nanmean(best_dp_mpjpe_max_ids), 4), 'worst mean:', round(np.nanmean(worst_dp_mpjpe_max_ids), 4), 'mean of all samples:', round(np.nanmean(dp_mpjpe_max_res_array), 4))
if has_power and np.isfinite(power_res_array).any():
    print('Best and worst Power mean differences:', round(np.nanmean(power_diffs), 4), 'median:', round(np.nanmedian(power_diffs), 4), 'best mean:', round(np.nanmean(best_power_ids), 4), 'worst mean:', round(np.nanmean(worst_power_ids), 4), 'mean of all samples:', round(np.nanmean(power_res_array), 4))
if has_feet_height and np.isfinite(feet_height_res_array).any():
    print('Best and worst Feet Height mean differences:', round(np.nanmean(feet_height_diffs), 4), 'difference median:', round(np.nanmedian(feet_height_diffs), 4), 'best mean:', round(np.nanmean(best_feet_height_ids), 4), 'worst mean:', round(np.nanmean(worst_feet_height_ids), 4), 'mean of all samples:', round(np.nanmean(feet_height_res_array), 4))
if has_jerk and np.isfinite(jerk_res_array).any():
    print('Best and worst Jerk mean differences:', round(np.nanmean(jerk_diffs), 4), 'difference median:', round(np.nanmedian(jerk_diffs), 4), 'best mean:', round(np.nanmean(best_jerk_ids), 4), 'worst mean:', round(np.nanmean(worst_jerk_ids), 4), 'mean of all samples:', round(np.nanmean(jerk_res_array), 4))
if has_m2t and np.isfinite(m2t_res_array).any():
    print('Best and worst M2T score mean differences:', round(np.nanmean(m2t_diffs), 4), 'difference median:', round(np.nanmedian(m2t_diffs), 4), 'best mean:', round(np.nanmean(best_m2t_ids), 4), 'worst mean:', round(np.nanmean(worst_m2t_ids), 4), 'mean of all samples:', round(np.nanmean(m2t_res_array), 4))
if has_m2m and np.isfinite(m2m_res_array).any():
    print('Best and worst M2M score mean differences:', round(np.nanmean(m2m_diffs), 4), 'difference median:', round(np.nanmedian(m2m_diffs), 4), 'best mean:', round(np.nanmean(best_m2m_ids), 4), 'worst mean:', round(np.nanmean(worst_m2m_ids), 4), 'mean of all samples:', round(np.nanmean(m2m_res_array), 4))

# print most closer and more different samples
best_sliding_closer_ids, worst_sliding_closer_ids = _extreme_indices(sliding_diffs)
best_mpjpe_closer_ids, worst_mpjpe_closer_ids = _extreme_indices(mpjpe_diffs)
if has_dp_mpjpe and np.isfinite(dp_mpjpe_res_array).any():
    best_dp_mpjpe_closer_ids, worst_dp_mpjpe_closer_ids = _extreme_indices(dp_mpjpe_diffs)
if has_dp_mpjpe_max and np.isfinite(dp_mpjpe_max_res_array).any():
    best_dp_mpjpe_max_closer_ids, worst_dp_mpjpe_max_closer_ids = _extreme_indices(dp_mpjpe_max_diffs)
if has_power and np.isfinite(power_res_array).any():
    best_power_closer_ids, worst_power_closer_ids = _extreme_indices(power_diffs)
if has_feet_height and np.isfinite(feet_height_res_array).any():
    best_feet_height_closer_ids, worst_feet_height_closer_ids = _extreme_indices(feet_height_diffs)
if has_jerk and np.isfinite(jerk_res_array).any():
    best_jerk_closer_ids, worst_jerk_closer_ids = _extreme_indices(jerk_diffs)
if has_m2t and np.isfinite(m2t_res_array).any():
    best_m2t_closer_ids, worst_m2t_closer_ids = _extreme_indices(m2t_diffs)
if has_m2m and np.isfinite(m2m_res_array).any():
    best_m2m_closer_ids, worst_m2m_closer_ids = _extreme_indices(m2m_diffs)

if best_sliding_closer_ids is not None:
    print('Most closer sliding samples:', best_sliding_closer_ids, sliding_diffs[best_sliding_closer_ids])
    print('Most different sliding samples:', worst_sliding_closer_ids, sliding_diffs[worst_sliding_closer_ids])
if best_mpjpe_closer_ids is not None:
    print('Most closer MPJPE samples:', best_mpjpe_closer_ids, mpjpe_diffs[best_mpjpe_closer_ids])
    print('Most different MPJPE samples:', worst_mpjpe_closer_ids, mpjpe_diffs[worst_mpjpe_closer_ids])
if has_dp_mpjpe and np.isfinite(dp_mpjpe_res_array).any() and best_dp_mpjpe_closer_ids is not None:
    print('Most closer DeepMimic MPJPE samples:', best_dp_mpjpe_closer_ids, dp_mpjpe_diffs[best_dp_mpjpe_closer_ids])
    print('Most different DeepMimic MPJPE samples:', worst_dp_mpjpe_closer_ids, dp_mpjpe_diffs[worst_dp_mpjpe_closer_ids])
if has_dp_mpjpe_max and np.isfinite(dp_mpjpe_max_res_array).any() and best_dp_mpjpe_max_closer_ids is not None:
    print('Most closer DeepMimic MPJPE Max samples:', best_dp_mpjpe_max_closer_ids, dp_mpjpe_max_diffs[best_dp_mpjpe_max_closer_ids])
    print('Most different DeepMimic MPJPE Max samples:', worst_dp_mpjpe_max_closer_ids, dp_mpjpe_max_diffs[worst_dp_mpjpe_max_closer_ids])
if has_power and np.isfinite(power_res_array).any() and best_power_closer_ids is not None:
    print('Most closer Power samples:', best_power_closer_ids, power_diffs[best_power_closer_ids])
    print('Most different Power samples:', worst_power_closer_ids, power_diffs[worst_power_closer_ids])
if has_feet_height and np.isfinite(feet_height_res_array).any() and best_feet_height_closer_ids is not None:
    print('Most closer Feet Height samples:', best_feet_height_closer_ids, feet_height_diffs[best_feet_height_closer_ids])
    print('Most different Feet Height samples:', worst_feet_height_closer_ids, feet_height_diffs[worst_feet_height_closer_ids])
if has_jerk and np.isfinite(jerk_res_array).any() and best_jerk_closer_ids is not None:
    print('Most closer Jerk samples:', best_jerk_closer_ids, jerk_diffs[best_jerk_closer_ids])
    print('Most different Jerk samples:', worst_jerk_closer_ids, jerk_diffs[worst_jerk_closer_ids])
if has_m2t and np.isfinite(m2t_res_array).any() and best_m2t_closer_ids is not None:
    print('Most closer M2T score samples:', best_m2t_closer_ids, m2t_diffs[best_m2t_closer_ids])
    print('Most different M2T score samples:', worst_m2t_closer_ids, m2t_diffs[worst_m2t_closer_ids])
if has_m2m and np.isfinite(m2m_res_array).any() and best_m2m_closer_ids is not None:
    print('Most closer M2M score samples:', best_m2m_closer_ids, m2m_diffs[best_m2m_closer_ids])
    print('Most different M2M score samples:', worst_m2m_closer_ids, m2m_diffs[worst_m2m_closer_ids])

# Save mean and std of all samples to JSON
metric_stats = {
    'skating_ratio': {
        'mean': float(np.nanmean(sliding_res_array)),
        'std': float(np.nanstd(sliding_res_array))
    },
    'control_l2_dist': {
        'mean': float(np.nanmean(mpjpe_res_array)),
        'std': float(np.nanstd(mpjpe_res_array))
    }
}

if has_dp_mpjpe and np.isfinite(dp_mpjpe_res_array).any():
    metric_stats['simulation_error'] = {
        'mean': float(np.nanmean(dp_mpjpe_res_array)),
        'std': float(np.nanstd(dp_mpjpe_res_array))
    }

if has_dp_mpjpe_max and np.isfinite(dp_mpjpe_max_res_array).any():
    metric_stats['simulation_error_max'] = {
        'mean': float(np.nanmean(dp_mpjpe_max_res_array)),
        'std': float(np.nanstd(dp_mpjpe_max_res_array))
    }

if has_power and np.isfinite(power_res_array).any():
    metric_stats['power'] = {
        'mean': float(np.nanmean(power_res_array)),
        'std': float(np.nanstd(power_res_array))
    }

if has_feet_height and np.isfinite(feet_height_res_array).any():
    metric_stats['feet_height'] = {
        'mean': float(np.nanmean(feet_height_res_array)),
        'std': float(np.nanstd(feet_height_res_array))
    }

if has_jerk and np.isfinite(jerk_res_array).any():
    metric_stats['jerk'] = {
        'mean': float(np.nanmean(jerk_res_array)),
        'std': float(np.nanstd(jerk_res_array))
    }

if has_m2t and np.isfinite(m2t_res_array).any():
    metric_stats['m2t_score'] = {
        'mean': float(np.nanmean(m2t_res_array)),
        'std': float(np.nanstd(m2t_res_array))
    }

if has_m2m and np.isfinite(m2m_res_array).any():
    metric_stats['m2m_score'] = {
        'mean': float(np.nanmean(m2m_res_array)),
        'std': float(np.nanstd(m2m_res_array))
    }

# Save to JSON file
metric_stats_path = os.path.join(dpo_root_dir, 'metric_stats.json')
with open(metric_stats_path, 'w') as f:
    json.dump(metric_stats, f, indent=4)

print(f'\n{"="*60}')
print(f'Metric statistics saved to: {metric_stats_path}')
print(f'{"="*60}')
