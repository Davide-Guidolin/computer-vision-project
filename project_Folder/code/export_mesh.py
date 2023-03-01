# Extract mesh
from kornia.utils.grid import create_meshgrid3d
import vren
import mcubes
import trimesh

import torch
import time
import os
import numpy as np
from ngp_pl.models.networks import NGP
from ngp_pl.models.rendering import render
from ngp_pl.metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from ngp_pl.datasets import dataset_dict
from ngp_pl.datasets.ray_utils import get_rays
from ngp_pl.utils import load_ckpt
from ngp_pl.train import depth2img
import imageio

import argparse
parser = argparse.ArgumentParser()
args_list = []

dataset_name = "colmap" #@param ['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv']
parser.add_argument('--dataset_name', type=str)
args_list.append('--dataset_name')
args_list.append(dataset_name)

scale = 0.5  #@param {type:"number"}
parser.add_argument('--scale', type=float)
args_list.append('--scale')
args_list.append(str(scale))

exp_name = "fruit_2" #@param {type:"string"}
parser.add_argument('--exp_name', type=str)
args_list.append('--exp_name')
args_list.append(exp_name)

args = parser.parse_args(args_list)

hparams = args

ckpt_path = "epoch=99_slim.ckpt" #@param {type:"string"}
model = NGP(scale=hparams.scale).cuda()
load_ckpt(model, f'/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}/{ckpt_path}')

xyz = create_meshgrid3d(model.grid_size, model.grid_size, model.grid_size, False, dtype=torch.int32).reshape(-1, 3)
# _density_bitfield = model.density_bitfield
# density_bitfield = torch.zeros(model.cascades*model.grid_size**3//8, 8, dtype=torch.bool)
# for i in range(8):
#     density_bitfield[:, i] = _density_bitfield & torch.tensor([2**i], device='cuda')
# density_bitfield = density_bitfield.reshape(model.cascades, model.grid_size**3).cpu()
indices = vren.morton3D(xyz.cuda()).long()


### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 512 # controls the resolution, set this number small here because we're only finding
        # good ranges here, not yet for mesh reconstruction; we can set this number high
        # when it comes to final reconstruction.
xmin, xmax = -0.3, 0.7 # left/right range
ymin, ymax = -0.2, 0.8 # forward/backward range
zmin, zmax = -0.5, 0.5 # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 20. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
############################################################################################

x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
z = np.linspace(zmin, zmax, N)

xyz = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()

chunk_size = 1024*16
with torch.no_grad():
    B = xyz.shape[0]
    out_chunks = []
    for i in range(0, B, chunk_size):
        xyz_chunk = xyz[i:i+chunk_size]
        out_chunks += [model.density(xyz_chunk)]
    sigma = torch.cat(out_chunks, 0)

sigma = sigma.cpu().numpy().astype(np.float32)
sigma = sigma.reshape(N, N, N)
# The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)

mesh_folder = f'/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}/mesh'
if not os.path.exists(mesh_folder):
    os.mkdir(mesh_folder)

mcubes.export_mesh(vertices, triangles, f"/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}/mesh/mesh.dae")