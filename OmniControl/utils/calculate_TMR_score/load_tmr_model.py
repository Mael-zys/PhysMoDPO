import torch
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.calculate_TMR_score.tmr_paths import ensure_tmr_on_path, get_tmr_path, resolve_tmr_path

ensure_tmr_on_path()
from TMR.src.data.collate import collate_x_dict
from TMR.src.config import read_config
from TMR.src.load import load_model_from_cfg
from hydra.utils import instantiate


def _resolve_path_from_run_dir(path: str, run_dir: str) -> str:
    if not isinstance(path, str) or path == "":
        return path

    path = os.path.expanduser(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path

    cur_dir = os.path.abspath(run_dir)
    while True:
        candidate = os.path.join(cur_dir, path)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur_dir)
        if parent == cur_dir:
            break
        cur_dir = parent
    return path


def _resolve_cfg_paths(cfg, run_dir: str) -> None:
    key_chains = [
        ("data", "motion_loader", "base_dir"),
        ("data", "motion_loader", "normalizer", "base_dir"),
        ("data", "text_to_token_emb", "path"),
        ("data", "text_to_sent_emb", "path"),
    ]

    for keys in key_chains:
        try:
            value = cfg
            for key in keys:
                value = value[key]
        except Exception:
            continue

        if not isinstance(value, str):
            continue

        resolved = _resolve_path_from_run_dir(value, run_dir)
        if resolved != value:
            target = cfg
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = resolved


def load_tmr_model_easy(device="cpu", run_dir=None, ckpt_name="last"):
    run_dir = resolve_tmr_path(run_dir) or get_tmr_path("models", "tmr_humanml3d_guoh3dfeats")
    ckpt_name = ckpt_name
    cfg = read_config(run_dir)
    _resolve_cfg_paths(cfg, run_dir)

    print("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    print("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, preload=False, device=device)

    def easy_forward(motions_or_texts):
        if isinstance(motions_or_texts[0], str):
            texts = motions_or_texts
            x_dict = collate_x_dict(text_model(texts))
        else:
            motions = motions_or_texts
            motions = [
                normalizer(torch.from_numpy(motion).to(torch.float)).to(device)
                for motion in motions
            ]
            x_dict = collate_x_dict(
                [
                    {
                        "x": motion,
                        "length": len(motion),
                    }
                    for motion in motions
                ]
            )

        # Support both PyTorch 1.7.1 (with no_grad) and 1.9+ (with inference_mode).
        inference_context = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
        with inference_context():
            latents = model.encode(x_dict, sample_mean=True).cpu()
        return latents

    return easy_forward
