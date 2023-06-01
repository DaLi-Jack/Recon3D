# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
sys.path.append("./omni3d/")                  # add omni3d to PYTHONPATH
sys.path.append("./segment-anything/")        # add segment-anything to PYTHONPATH
sys.path.append("./shap-e/")                  # add shap-e to PYTHONPATH
import numpy as np
from collections import OrderedDict
import torch
import cv2
import matplotlib.pyplot as plt
import json
from PIL import Image


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

import copy

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn import util, vis
from utils import *

from segment_anything import sam_model_registry, SamPredictor

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh

from diffusers import StableDiffusionInpaintPipeline

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    device = "cuda"
    cfg = setup(args)
    model = build_model(cfg)

    use_inpainting = args.inpaint
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    # for detection omni3d
    with torch.no_grad():
        detection(args, cfg, model)

    print('detection over!')

    # for sam
    # setup sam
    sam_checkpoint = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # for shap-e
    # setup shap-e
    xm = load_model('transmitter', device=device, cache_dir='shap-e/shap_e/examples/shap_e_model_cache')
    model = load_model('image300M', device=device, cache_dir='shap-e/shap_e/examples/shap_e_model_cache')
    diffusion = diffusion_from_config(load_config('diffusion'))
    batch_size = 1
    guidance_scale = 3.0

    # for inpainting
    # setup stable diffusion inpainting
    if use_inpainting:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.to(device)


    print('model setup over!')

    # load detection results
    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')
    for img_path in list_of_ims:
        im_name = util.file_parts(img_path)[1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        save_root_path = os.path.join(cfg.OUTPUT_DIR, im_name)
        if not os.path.exists(save_root_path):
            raise ValueError('No detection results for {}'.format(im_name))

        save_sam_path = os.path.join(save_root_path, 'sam')
        os.makedirs(save_sam_path, exist_ok=True)
        save_inpaint_path = os.path.join(save_root_path, 'inpaint')
        os.makedirs(save_inpaint_path, exist_ok=True)
        save_shape_path = os.path.join(save_root_path, 'shape')
        os.makedirs(save_shape_path, exist_ok=True)

        predictor.set_image(image)

        detection_results = json.load(open(os.path.join(save_root_path, 'det', im_name+'_detection_results.json'), 'r'))
        for det_text in detection_results:
            obj_dic = detection_results[det_text]

            # get bbox 2d for sam prompt
            input_box = np.array(obj_dic['bbox2D'])
            obj_vis_masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(obj_vis_masks[0], plt.gca())
            show_box(input_box, plt.gca())
            plt.axis('off')
            plt.savefig(os.path.join(save_sam_path, det_text+'_sam.png'))
            # plt.show()

            obj_vis_masks = obj_vis_masks.transpose(1, 2, 0)  # H x W x 1

            # get crop bbox
            crop_bbox = get_crop_bbox(obj_vis_masks)
            [x_square_begin, y_square_begin, x_square_end, y_square_end] = crop_bbox

            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_box(crop_bbox, plt.gca())
            plt.savefig(os.path.join(save_sam_path, det_text+'_crop_bbox.png'))
            # plt.show()

            # get vis object image, use PIL
            img = Image.open(img_path)
            img_np = np.array(img)
            img_np = img_np[:, :, :3]               # remove alpha channel(if have)
            height, width, _ = img_np.shape

            save_vis_img = np.ones((height, width, 3), dtype=np.uint8) * 225
            save_vis_img[obj_vis_masks[:, :, 0] == 1] = img_np[obj_vis_masks[:, :, 0] == 1]
            save_vis_img = save_vis_img[y_square_begin:y_square_end, x_square_begin:x_square_end, :]            # crop square image
            save_vis_img = Image.fromarray(save_vis_img)
            save_vis_img.save(os.path.join(save_sam_path, det_text+'_vis.png'))
            # save_vis_img.show()


            if use_inpainting:
                # stable diffusion inpainting
                # get diffuser mask, now set the bbox 2d as diffuser mask
                bbox_mask = np.zeros((height, width, 1), dtype=np.uint8)
                bbox_mask[int(input_box[1]):int(input_box[3]), int(input_box[0]):int(input_box[2])] = 1

                diffuser_mask = bbox_mask - obj_vis_masks
                crop_diffuse_mask = diffuser_mask[y_square_begin:y_square_end, x_square_begin:x_square_end, :]            # setting crop, must be square
                crop_diffuse_mask = crop_diffuse_mask[:, :, 0]
                crop_diffuse_mask = Image.fromarray(crop_diffuse_mask * 255)
                crop_diffuse_mask.save(os.path.join(save_inpaint_path, det_text+'_diffuse_mask.png'))


                category = det_text.split(' ')[0]
                prompt = f"a {category}"
                vis_img = Image.open(os.path.join(save_sam_path, det_text+'_vis.png'))
                vis_img = img_resize(vis_img, (512, 512))
                mask_img = Image.open(os.path.join(save_inpaint_path, det_text+'_diffuse_mask.png'))
                mask_img = img_resize(mask_img, (512, 512))
                image = pipe(prompt=prompt, image=vis_img, mask_image=mask_img).images[0]
                # image = pipe(image=image, mask_image=mask_image).images[0]
                image.save(os.path.join(save_inpaint_path, det_text+'_inpaint.png'))


            # shap-e gen model
            # fix seed, for keep gen model orientation
            seed_torch(seed=0)
            if use_inpainting:
                shape_img = load_image(os.path.join(save_inpaint_path, det_text+'_inpaint.png'))
            else:
                shape_img = load_image(os.path.join(save_sam_path, det_text+'_vis.png'))
            latents = sample_latents(
                batch_size=batch_size,
                model=model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(images=[shape_img] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )


            t = decode_latent_mesh(xm, latents[0]).tri_mesh()
            with open(os.path.join(save_shape_path, f'{det_text}.ply'), 'wb') as f:
                t.write_ply(f)
            with open(os.path.join(save_shape_path, f'{det_text}.obj'), 'w') as f:
                t.write_obj(f)


        # fusion scene
        scene_mesh = []
        for det_text in detection_results:
            obj_dic = detection_results[det_text]
            mesh_path = os.path.join(save_shape_path, f'{det_text}.ply')

            mesh = get_mesh_scene(mesh_path, obj_dic)
            scene_mesh.append(mesh)
        
        scene_mesh = trimesh.util.concatenate(scene_mesh)
        scene_mesh.export(os.path.join(save_root_path, 'scene.ply'))

        print('Finish {}'.format(im_name))




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    parser.add_argument("--inpaint", default=False, action="store_true", help="Whether to inpaint the image")
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )