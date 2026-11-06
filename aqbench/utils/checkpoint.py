import os
import torch


def get_best_checkpoint(exp_name, ckpt_dir="checkpoints"):
    path = os.path.join(ckpt_dir, exp_name, "best_model.pt")

    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, exp_name, "stage1.pt")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    return path


def load_checkpoint(model, exp_name, device):

    ckpt_path = get_best_checkpoint(exp_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state"] 
    model.load_state_dict(state_dict)

    print(f"Loaded checkpoint from {ckpt_path}")
    return model, ckpt