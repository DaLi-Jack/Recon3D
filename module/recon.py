import logging
import os
import torch
import argparse
import sys
sys.path.append("../")                        # add root path to PYTHONPATH
sys.path.append(os.getcwd())                  # add current path to PYTHONPATH
sys.path.append("./shap-e/")                  # add shap-e to PYTHONPATH
import numpy as np
import random

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh


# fix seed
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class Shap_e():
    def __init__(self, args, device):
        cache_dir = args.shap_e_cache_dir
        self.xm = load_model('transmitter', device=device, cache_dir=cache_dir)
        self.model = load_model('image300M', device=device, cache_dir=cache_dir)
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        self.batch_size = 1
        self.guidance_scale = 3.0

    def run(self, shape_img_path, save_shape_path, det_text):
        # fix seed for every object, for keep gen model orientation
        seed_torch(seed=0)

        # load image
        shape_img = load_image(shape_img_path)

        # sample latents
        latents = sample_latents(
            batch_size=self.batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(images=[shape_img] * self.batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        # decode latents
        t = decode_latent_mesh(self.xm, latents[0]).tri_mesh()
        with open(os.path.join(save_shape_path, f'{det_text}.ply'), 'wb') as f:
            t.write_ply(f)
        with open(os.path.join(save_shape_path, f'{det_text}.obj'), 'w') as f:
            t.write_obj(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shap_e_cache_dir", type=str, default='shap-e/shap_e/examples/shap_e_model_cache', help="path to shap-e cache dir")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recon = Shap_e(args, device)
    shape_img_path = '/home/nijunfeng/mycode/recon3D/output/test_det/000738_rgb_003787/inpaint/sofa 0.99_inpaint.png'
    save_shape_path = '/home/nijunfeng/mycode/recon3D/output/test_det/000738_rgb_003787/shape'
    os.makedirs(save_shape_path, exist_ok=True)

    recon.run(shape_img_path, save_shape_path, 'sofa')

    print('done')