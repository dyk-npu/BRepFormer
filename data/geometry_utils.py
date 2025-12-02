import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def get_random_rotation():
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)

def rotate_uvgrid(inp, rotation):
    Rmat = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
    orig_size = inp[..., :3].shape
    inp[..., :3] = torch.mm(inp[..., :3].reshape(-1, 3), Rmat).view(orig_size)
    inp[..., 3:6] = torch.mm(inp[..., 3:6].reshape(-1, 3), Rmat).view(orig_size)
    return inp