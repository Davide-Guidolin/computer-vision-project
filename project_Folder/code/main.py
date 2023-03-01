from ngp_pl.train import *

import argparse
parser = argparse.ArgumentParser()
args_list = []

root_dir = "/media/dataset/fruit_2" #@param {type: "string"}
parser.add_argument('--root_dir', type=str)
args_list.append('--root_dir')
args_list.append(root_dir)

dataset_name = "colmap" #@param ['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv']
parser.add_argument('--dataset_name', type=str)
args_list.append('--dataset_name')
args_list.append(dataset_name)

split = "trainvaltest" #@param ['train', 'trainval', 'trainvaltest']
parser.add_argument('--split', type=str)
args_list.append('--split')
args_list.append(split)

downsample = 0.5 #@param {type:"number"}
parser.add_argument('--downsample', type=float)
args_list.append('--downsample')
args_list.append(str(downsample))

#@markdown ---
#@markdown ### Model parameters
scale = 0.5  #@param {type:"number"}
parser.add_argument('--scale', type=float)
args_list.append('--scale')
args_list.append(str(scale))

parser.add_argument('--use_exposure', action='store_true', default=False)
use_exposure = False #@param ["False", "True"] {type:"raw"}
if use_exposure:
    args_list.append('--use_exposure')

#@markdown ---
#@markdown ### Loss parameters
distortion_loss_w = 0 #@param {type:"number"}
parser.add_argument('--distortion_loss_w', type=float)
args_list.append('--distortion_loss_w')
args_list.append(str(distortion_loss_w))

#@markdown ---
#@markdown ### Training options
batch_size = 8192 #@param {type:"integer"}
parser.add_argument('--batch_size', type=int)
args_list.append('--batch_size')
args_list.append(str(batch_size))

ray_sampling_strategy = "all_images" #@param ['all_images', 'same_image']
parser.add_argument('--ray_sampling_strategy', type=str)
args_list.append('--ray_sampling_strategy')
args_list.append(ray_sampling_strategy)

num_epochs = 100 #@param {type:"integer"}
parser.add_argument('--num_epochs', type=int)
args_list.append('--num_epochs')
args_list.append(str(num_epochs))

num_gpus = 1 #@param {type:"integer"}
parser.add_argument('--num_gpus', type=int)
args_list.append('--num_gpus')
args_list.append(str(num_gpus))

lr = 1e-2 #@param {type:"number"}
parser.add_argument('--lr', type=float)
args_list.append('--lr')
args_list.append(str(lr))

#@markdown ---
#@markdown ### Experimental training options
parser.add_argument('--optimize_ext', action='store_true', default=False)
optimize_ext = False #@param ["False", "True"] {type:"raw"}
if optimize_ext:
    args_list.append('--optimize_ext')

parser.add_argument('--random_bg', action='store_true', default=False)
random_bg = False #@param ["False", "True"] {type:"raw"}
if random_bg:
    args_list.append('--random_bg')


#@markdown ---
#@markdown ### Validation options
parser.add_argument('--eval_lpips', action='store_true', default=False)
eval_lpips = False #@param ["False", "True"] {type:"raw"}
if eval_lpips:
    args_list.append('--eval_lpips')

parser.add_argument('--val_only', action='store_true', default=False)
val_only = False #@param ["False", "True"] {type:"raw"}
if val_only:
    args_list.append('--val_only')

parser.add_argument('--no_save_test', action='store_true', default=False)
no_save_test = False #@param ["False", "True"] {type:"raw"}
if no_save_test:
    args_list.append('--no_save_test')

#@markdown ---
#@markdown ### Misc
exp_name = "fruit_2" #@param {type:"string"}
parser.add_argument('--exp_name', type=str)
args_list.append('--exp_name')
args_list.append(exp_name)

def none_string(val):
    if val == 'None':
        return None
    return val

ckpt_path = "None" #@param {type:"string"}
parser.add_argument('--ckpt_path', type=none_string)
args_list.append('--ckpt_path')
args_list.append(ckpt_path)

weight_path = "None" #@param {type:"string"}
parser.add_argument('--weight_path', type=none_string)
args_list.append('--weight_path')
args_list.append(weight_path)

args = parser.parse_args(args_list)

hparams = args
if hparams.val_only and (not hparams.ckpt_path):
    raise ValueError('You need to provide a @ckpt_path for validation!')
system = NeRFSystem(hparams)

ckpt_cb = ModelCheckpoint(dirpath=f'/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}',
                            filename='{epoch:d}',
                            save_weights_only=True,
                            every_n_epochs=hparams.num_epochs,
                            save_on_train_epoch_end=True,
                            save_top_k=-1)
callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

logger = TensorBoardLogger(save_dir=f"/media/checkpoint/logs/{hparams.dataset_name}",
                            name=hparams.exp_name,
                            default_hp_metric=False)

trainer = Trainer(max_epochs=hparams.num_epochs,
                    check_val_every_n_epoch=hparams.num_epochs,
                    callbacks=callbacks,
                    logger=logger,
                    enable_model_summary=False,
                    accelerator='gpu',
                    devices=hparams.num_gpus,
                    strategy=DDPPlugin(find_unused_parameters=False)
                            if hparams.num_gpus>1 else None,
                    num_sanity_val_steps=-1 if hparams.val_only else 0,
                    precision=16)

trainer.fit(system, ckpt_path=hparams.ckpt_path)

if not hparams.val_only: # save slimmed ckpt for the last epoch
    ckpt_ = \
        slim_ckpt(f'/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                    save_poses=hparams.optimize_ext)
    torch.save(ckpt_, f'/media/checkpoint/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

if (not hparams.no_save_test) and \
    hparams.dataset_name=='nsvf' and \
    'Synthetic' in hparams.root_dir: # save video
    imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
    imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                    [imageio.imread(img) for img in imgs[::2]],
                    fps=30, macro_block_size=1)
    imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                    [imageio.imread(img) for img in imgs[1::2]],
                    fps=30, macro_block_size=1)