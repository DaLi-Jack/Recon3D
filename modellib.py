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
    parser.add_argument("--config-file", default="cubercnn://omni3d/cubercnn_DLA34_FPN.yaml", metavar="FILE", help="path to config file")
    # parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    parser.add_argument("--inpaint", default=True, action="store_true", help="Whether to inpaint the image")
    
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
    cfg.MODEL.WEIGHTS = 'cubercnn://omni3d/cubercnn_DLA34_FPN.pth'
    cfg.OUTPUT_DIR = 'output/demo'
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


class Robot:
    def __init__(self):
        args = read_args()
        # cfg.MODEL.WEIGHTS = 'cubercnn://omni3d/cubercnn_DLA34_FPN.pth'
        # cfg.OUTPUT_DIR = 'output/demo'
        cfg = read_cfg(args)
        self.output_dir = cfg.OUTPUT_DIR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # detection model
        self.det = Omni3DDet(args, cfg)
        # segmentation model
        self.seg = SAM(args, device)
        # inpaint model 
        self.use_inpaint = True
        if self.use_inpaint:
            self.inpaint = DiffusionInpaint(device)
        # recon 
        self.recon = Shap_e(args, device)
        self.items = {}
        
        
        
    def set(self, image_path):
        #init args
        self.items['test_image'] = image_path 
        self.input_folder = os.path.dirname(image_path)
        self.image_name = image_path.split('/')[-1].split('.')[0]
        self.save_root_path = os.path.join(self.output_dir, self.image_name)
        os.makedirs(self.save_root_path, exist_ok=True)
        #init sam
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.seg.set_image(image)
        
        self.det_file = os.path.join(self.save_root_path, 'det', self.image_name+'_detection_results.json')
        self.save_sam_path = os.path.join(self.save_root_path, 'sam')
        self.save_inpaint_path = os.path.join(self.save_root_path, 'inpaint')
        self.save_shape_path = os.path.join(self.save_root_path, 'shape')
        os.makedirs(self.save_sam_path, exist_ok=True)
        os.makedirs(self.save_inpaint_path, exist_ok=True)
        os.makedirs(self.save_shape_path, exist_ok=True)
        

        
    def get_detection_result(self):
        if os.path.exists(self.det_file):
            with open(self.det_file, "r") as f:
                detection_results = json.load(f)
            return True, detection_results
        else:
            return False, {}
        
        
    def run_det(self):
        self.det.args.input_folder = self.input_folder
        self.det.run(self.items['test_image'])
        self.det_file = os.path.join(self.save_root_path, 'det', self.image_name+'_detection_results.json')
        det_res = os.path.join(self.save_root_path, 'det', f'{self.image_name}_3Dboxes.jpg')
        self.items['det_res'] = det_res  
        flag, detection_results = self.get_detection_result()
        items = {}
        if flag:
            for det_text, obj_dic in detection_results.items():
                item = {}
                for k, v in obj_dic.items():
                    item[k] = v
                item['name'] = det_text
                items[det_text] = item
                
        self.items['items'] = items
        print("run detection!")
    
    def run_seg(self, det_text, obj_dic):
        self.save_sam_path = os.path.join(self.save_root_path, 'sam')
        os.makedirs(self.save_sam_path, exist_ok=True)
        input_box = np.array(obj_dic['bbox2D'])
        self.seg.run(input_box, det_text, self.save_sam_path, use_inpainting=True)
        self.items['items'][det_text]['sam_res'] = os.path.join(self.save_root_path, 'sam', det_text + '_sam.png')
        self.items['items'][det_text]['mask'] = os.path.join(self.save_root_path, 'sam', det_text + '_mask.png')
        self.items['items'][det_text]['diffuse_mask'] = os.path.join(self.save_root_path, 'sam', det_text + '_diffuse_mask.png')
        self.items['items'][det_text]['sam_vis'] = os.path.join(self.save_root_path, 'sam', det_text + '_vis.png')
        print("run sam!")
        return True

    def run_inpaint(self, det_text, prompt=None):
        obj_dic = self.items['items'][det_text]
        self.save_inpaint_path = os.path.join(self.save_root_path, 'inpaint')
        os.makedirs(self.save_inpaint_path, exist_ok=True)
        category = det_text.split(' ')[0]
        if prompt is None:
            prompt = f"a {category}"
        self.inpaint.run(prompt, obj_dic, self.save_inpaint_path)
        # self.items['items'][det_text]['inpaint_res'] = self.save_inpaint_path
        print("run inpaint")
        return True 
    
    def run_recon(self, det_text):
        det_obj = self.items['items'][det_text]
        self.save_shape_path = os.path.join(self.save_root_path, 'shape')
        os.makedirs(self.save_shape_path, exist_ok=True)
        if self.use_inpaint:
            inpaint_image = det_obj['inpaint_res']
            self.shape_img_path = inpaint_image
        else: 
            sam_image = det_obj['sam_vis']
            self.shape_img_path = sam_image
        # else:
        #     print("recon error")
        #     return False
        self.recon.run(self.shape_img_path, self.save_shape_path, det_text)
        self.items['items'][det_text]['mesh']  = os.path.join(self.save_root_path, 'shape', det_text + '.ply')
        print("run recon")
        return True 
        
    def run_fusion(self):
        fusion_scene(self.items['items'], self.save_root_path, self.save_shape_path)
if __name__ == "__main__":
    robot = Robot()
    robot.set(image_path='./test_img/real_img/0005.png') 
    robot.run_det()
    flag, detection_results = robot.get_detection_result()
    for det_text, obj_dic in detection_results.items():
        robot.run_seg(det_text, obj_dic)
        robot.run_inpaint(det_text)
        robot.run_recon(det_text)
    robot.run_fusion()
    