import sys
import os
from pathlib import Path

# Add project root to path for both direct execution and module import
if __name__ == "__main__":
    # When running directly, add project root to sys.path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np

from utils.calculate_TMR_score.metrics import (
    calculate_frechet_distance,
    calculate_activation_statistics_normalized,
)
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy
from utils.calculate_TMR_score.tmr_paths import ensure_tmr_on_path

ensure_tmr_on_path()
from TMR.src.model.tmr import get_sim_matrix


def calculate_tmr_metrics(
    tmr_forward,
    texts_gt,
    motions_guofeats_pred,
    motions_guofeats_gt=None,
    calculate_retrieval=False,
    calculate_fid=False
):
    """
    Calculate TMR metrics for motion generation evaluation.
    
    Args:
        tmr_forward (callable): Function to obtain latent representations from inputs
        texts_gt (list): List of ground truth text descriptions
        motions_guofeats_pred (list or np.ndarray): Predicted motion features in guofeats format
        motions_guofeats_gt (list or np.ndarray): Ground truth motion features in guofeats format
        calculate_retrieval (bool): Whether to calculate retrieval metrics (R@1, R@3). Default: False
    
    Returns:
        dict: Dictionary containing metrics:
            - 'm2t_score': Motion-to-text TMR score (always included)
            - 'm2m_score': Motion-to-motion TMR score (always included)
            - 'fid': FID+ score (always included)
            - 'm2t_top_1': Motion-to-text R@1 (only if calculate_retrieval=True)
            - 'm2t_top_3': Motion-to-text R@3 (only if calculate_retrieval=True)
    
    Example:
        >>> metrics = calculate_tmr_metrics(
        ...     fps=20.0,
        ...     device='cuda',
        ...     texts_gt=texts,
        ...     motions_guofeats_gt=gt_motions,
        ...     motions_guofeats_pred=pred_motions,
        ...     calculate_retrieval=True
        ... )
        >>> print(f"TMR M2T Score: {metrics['m2t_score']:.3f}")
        >>> print(f"FID+: {metrics['fid']:.2f}")
    """
    if len(texts_gt) == 0 or len(motions_guofeats_pred) == 0:
        raise ValueError("texts_gt and motions_guofeats_pred must be non-empty.")

    if not isinstance(texts_gt[0], str):
        raise TypeError(
            "texts_gt must be a list of strings. "
            "This often means positional arguments were passed in the wrong order. "
            "Use keyword arguments: texts_gt=..., motions_guofeats_pred=..., motions_guofeats_gt=..."
        )

    if isinstance(motions_guofeats_pred[0], str):
        raise TypeError("motions_guofeats_pred must be motion features, not strings.")

    if motions_guofeats_gt is not None and len(motions_guofeats_gt) > 0 and isinstance(motions_guofeats_gt[0], str):
        raise TypeError("motions_guofeats_gt must be motion features, not strings.")

    metrics = {}
    
    # Calculate latent representations
    text_latents_gt = tmr_forward(texts_gt)
    motion_latents_pred = tmr_forward(motions_guofeats_pred)
    
    # Calculate m2t similarity matrix
    sim_matrix_m2t = get_sim_matrix(motion_latents_pred, text_latents_gt).numpy()
    
    # Calculate TMR m2t scores (normalized to [0, 1])
    m2t_score_lst = []
    for idx in range(len(motion_latents_pred)):
        m2t_score_lst.append((sim_matrix_m2t[idx, idx] + 1) / 2)
    metrics['m2t_score'] = np.mean(m2t_score_lst)
    
    # Calculate m2m score only if ground truth motions are provided
    if motions_guofeats_gt is not None:
        motion_latents_gt = tmr_forward(motions_guofeats_gt)
        sim_matrix_m2m = get_sim_matrix(motion_latents_pred, motion_latents_gt).numpy()
        
        m2m_score_lst = []
        for idx in range(len(motion_latents_pred)):
            m2m_score_lst.append((sim_matrix_m2m[idx, idx] + 1) / 2)
        metrics['m2m_score'] = np.mean(m2m_score_lst)
    
    # Calculate retrieval metrics if requested
    if calculate_retrieval:
        m2t_top_1_lst = []
        m2t_top_3_lst = []
        
        for idx in range(len(motion_latents_pred)):
            asort_m2t = np.argsort(sim_matrix_m2t[idx])[::-1]
            m2t_top_1_lst.append(1 * (idx in asort_m2t[:1]))
            m2t_top_3_lst.append(1 * (idx in asort_m2t[:3]))
        
        metrics['m2t_top_1'] = np.mean(m2t_top_1_lst)
        metrics['m2t_top_3'] = np.mean(m2t_top_3_lst)
    
    if calculate_fid:
        # Calculate FID+ metric
        gt_mu, gt_cov = calculate_activation_statistics_normalized(
            motion_latents_gt.numpy()
        )
        pred_mu, pred_cov = calculate_activation_statistics_normalized(
            motion_latents_pred.numpy()
        )
        
        metrics['fid'] = calculate_frechet_distance(
            gt_mu.astype(float),
            gt_cov.astype(float),
            pred_mu.astype(float),
            pred_cov.astype(float),
        )
    
    return metrics


def print_metrics(metrics, name="Model"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of metrics from calculate_tmr_metrics
        name (str): Name of the model/experiment to display
    """
    print(f"\n{'='*50}")
    print(f"Metrics for: {name}")
    print(f"{'='*50}")
    print(f"M2T Score:  {metrics['m2t_score']:.4f}")
    print(f"M2M Score:  {metrics['m2m_score']:.4f}")
    if 'fid' in metrics:
        print(f"FID+:       {metrics['fid']:.2f}")
    
    if 'm2t_top_1' in metrics:
        print(f"R@1:        {100 * metrics['m2t_top_1']:.2f}%")
        print(f"R@3:        {100 * metrics['m2t_top_3']:.2f}%")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    from utils.calculate_TMR_score.tmr_eval_wrapper import calculate_tmr_metrics, print_metrics
    device = 'cuda'  # or 'cpu'
    # Load TMR model
    tmr_forward = load_tmr_model_easy(device)
    texts = ['a man kicks something or someone with his left leg.']
    gt_motion = np.load('dataset/HumanML3D/new_joint_vecs/000000.npy')
    gt_motions = [gt_motion]
    pred_motions = [gt_motion]
    metrics = calculate_tmr_metrics(
        tmr_forward=tmr_forward,
        texts_gt=texts,
        motions_guofeats_gt=gt_motions,
        motions_guofeats_pred=pred_motions,
        calculate_retrieval=True  # Optional.
    )

    print_metrics(metrics, name="My Model")
