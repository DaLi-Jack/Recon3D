import os
import argparse
import sys
sys.path.append("./omni3d/")                  # add omni3d to PYTHONPATH
import torch
import json
import numpy as np
import cv2

from module.det import Omni3DDet
from module.segment import SAM
from module.inpaint import DiffusionInpaint
from module.recon import Shap_e
from module.fusion import fusion_scene

from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg

from cubercnn.config import get_cfg_defaults
from cubercnn import util, vis


def read_args():

    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # for omni3d
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

    # for sam
    parser.add_argument("--sam_checkpoint", type=str, default='segment-anything/checkpoints/sam_vit_h_4b8939.pth', help="path to sam checkpoint file")
    parser.add_argument("--model_type", type=str, default='vit_h', help="sam model type")

    # for shap-e
    parser.add_argument("--shap_e_cache_dir", type=str, default='shap-e/shap_e/examples/shap_e_model_cache', help="path to shap-e cache dir")

    args = parser.parse_args()

    return args


def read_cfg(args):
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


if __name__ == "__main__":

    args = read_args()
    cfg = read_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_inpainting = args.inpaint
    output_dir = cfg.OUTPUT_DIR


    # setup model
    # det
    omni3d_det = Omni3DDet(args, cfg)
    # sam
    sam = SAM(args, device)
    # inpaint
    if use_inpainting:
        inpaint = DiffusionInpaint(device)
    # recon
    recon = Shap_e(args, device)

    # 1. det
    omni3d_det.run()
    print('detection over!')

    # for every image
    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')
    for img_path in list_of_ims:
        im_name = util.file_parts(img_path)[1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam.set_image(image)

        save_root_path = os.path.join(output_dir, im_name)
        if not os.path.exists(save_root_path):
            raise ValueError('No detection results for {}'.format(im_name))
        
        # load detection results
        detection_results = json.load(open(os.path.join(save_root_path, 'det', im_name+'_detection_results.json'), 'r'))
        
        save_sam_path = os.path.join(save_root_path, 'sam')
        os.makedirs(save_sam_path, exist_ok=True)
        if use_inpainting:
            save_inpaint_path = os.path.join(save_root_path, 'inpaint')
            os.makedirs(save_inpaint_path, exist_ok=True)
        save_shape_path = os.path.join(save_root_path, 'shape')
        os.makedirs(save_shape_path, exist_ok=True)

        for det_text in detection_results:
            obj_dic = detection_results[det_text]

            # 2. sam
            input_box = np.array(obj_dic['bbox2D'])
            sam.run(input_box, det_text, save_sam_path, img_path, use_inpainting=True)

            # 3. inpaint
            if use_inpainting:
                category = det_text.split(' ')[0]
                prompt = f"a {category}"
                inpaint.run(prompt, save_sam_path, det_text, save_inpaint_path)

            # 4. recon
            if use_inpainting:
                shape_img_path = os.path.join(save_inpaint_path, det_text+'_inpaint.png')
            else:
                shape_img_path = os.path.join(save_sam_path, det_text+'_vis.png')
            recon.run(shape_img_path, save_shape_path, det_text)

        
        # 5. fusion scene mesh
        fusion_scene(detection_results, save_root_path, save_shape_path)

        print('Finish {}'.format(im_name))






