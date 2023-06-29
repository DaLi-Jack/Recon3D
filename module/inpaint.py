import logging
import os
import torch
import argparse
import sys
sys.path.append("../")                        # add root path to PYTHONPATH
sys.path.append(os.getcwd())                  # add current path to PYTHONPATH
import numpy as np
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline

def img_resize(img, size):
    img = img.resize(size, Image.ANTIALIAS)
    return img


class DiffusionInpaint():
    def __init__(self, device):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.to(device)
        self.pipe = pipe

    def run(self, prompt, item, save_inpaint_path):
        image_path = item['sam_vis']
        mask_path = item['diffuse_mask']
        det_text = item['name']
        vis_img = Image.open(image_path)
        vis_img = img_resize(vis_img, (512, 512))
        mask_img = Image.open(mask_path)
        mask_img = img_resize(mask_img, (512, 512))
        image = self.pipe(prompt=prompt, image=vis_img, mask_image=mask_img).images[0]
        # image = pipe(image=image, mask_image=mask_image).images[0]
        image.save(os.path.join(save_inpaint_path, det_text+'_inpaint.png'))
        item['inpaint_res'] = os.path.join(save_inpaint_path, det_text+'_inpaint.png')
        

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inpaint = DiffusionInpaint(device)

    prompt = 'a table'
    save_sam_path = '/home/nijunfeng/mycode/recon3D/output/test_det/000738_rgb_003787/sam'
    det_text = 'table 0.83'
    save_inpaint_path = '/home/zyw/data/repo_common/Recon3D/output/demo/0004/inpaint/'
    os.makedirs(save_inpaint_path, exist_ok=True)
    item = {
        'sam_vis': '/home/zyw/data/repo_common/Recon3D/output/demo/0004/sam/table 0.83_vis.png',
        'diffuse_mask': '/home/zyw/data/repo_common/Recon3D/output/demo/0004/sam/table 0.83_diffuse_mask.png',
        'name': det_text
    }
    inpaint.run(prompt, item, save_inpaint_path)

    print('done')

