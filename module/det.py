import logging
import os
import torch
import argparse
import sys
sys.path.append("../")                        # add root path to PYTHONPATH
sys.path.append(os.getcwd())                  # add current path to PYTHONPATH
sys.path.append("./omni3d/")                  # add omni3d to PYTHONPATH
import numpy as np

from pytorch3d.io import IO
import trimesh
import copy
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# for omni3d
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis


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


class Omni3DDet():
    def __init__(self, args, cfg):

        logger = logging.getLogger("detectron2")

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )

        self.args = args
        self.cfg = cfg
        self.model = model


    def run(self, image_path):
        with torch.no_grad():
            list_of_ims = util.list_files(os.path.join(self.args.input_folder, ''), '*')

            self.model.eval()

            thres = self.args.threshold

            output_dir = self.cfg.OUTPUT_DIR
            min_size = self.cfg.INPUT.MIN_SIZE_TEST
            max_size = self.cfg.INPUT.MAX_SIZE_TEST
            augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

            util.mkdir_if_missing(output_dir)

            category_path = os.path.join(util.file_parts(self.args.config_file)[0], 'category_meta.json')
                
            # store locally if needed
            if category_path.startswith(util.CubeRCNNHandler.PREFIX):
                category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

            metadata = util.load_json(category_path)
            cats = metadata['thing_classes']
            
            for path in list_of_ims:
                if path != image_path:
                    continue
                im_name = util.file_parts(path)[1]
                im = util.imread(path)
                img_draw_2d = copy.deepcopy(im)

                if im is None:
                    continue

                save_root_path = os.path.join(output_dir, im_name)
                os.makedirs(save_root_path, exist_ok=True)
                save_det_path = os.path.join(save_root_path, 'det')
                os.makedirs(save_det_path, exist_ok=True)
                
                image_shape = im.shape[:2]  # h, w

                h, w = image_shape
                f_ndc = 4
                f = f_ndc * h / 2

                K = np.array([
                    [f, 0.0, w/2], 
                    [0.0, f, h/2], 
                    [0.0, 0.0, 1.0]
                ])

                aug_input = T.AugInput(im)
                _ = augmentations(aug_input)
                image = aug_input.image

                batched = [{
                    'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
                    'height': image_shape[0], 'width': image_shape[1], 'K': K
                }]

                dets = self.model(batched)[0]['instances']
                n_det = len(dets)

                meshes = []
                meshes_text = []

                save_bbox3d = {}

                if n_det > 0:
                    for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx, bbox_2d) in enumerate(zip(
                            dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                            dets.pred_pose, dets.scores, dets.pred_classes, dets.pred_boxes
                        )):

                        # skip
                        if score < thres:
                            continue
                        
                        cat = cats[cat_idx]

                        mesh_text = '{} {:.2f}'.format(cat, score)
                        meshes_text.append(mesh_text)

                        # vis bbox 2d
                        bbox_2d = bbox_2d.data.cpu().numpy()
                        color = [c for c in util.get_color(idx)]
                        vis.draw_2d_box(img_draw_2d, [bbox_2d[0], bbox_2d[1], bbox_2d[2]-bbox_2d[0], bbox_2d[3]-bbox_2d[1]], color=color, thickness=max(2, int(np.round(3*img_draw_2d.shape[0]/1250))))
                        vis.draw_text(img_draw_2d, '{}'.format(mesh_text), [bbox_2d[0], bbox_2d[1]], scale=0.50*img_draw_2d.shape[0]/500, bg_color=color)

                        # vis bbox 3d
                        bbox3D = center_cam.tolist() + dimensions.tolist()
                        color = [c/255.0 for c in util.get_color(idx)]          # normalize to [0, 1]
                        box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                        meshes.append(box_mesh)

                        # save bbox3d
                        save_bbox3d[mesh_text] = {'bbox2D':bbox_2d.tolist(), 'bbox3D': bbox3D, 'pose': pose.tolist()}

                
                print('File: {} with {} dets'.format(im_name, len(meshes)))

                if len(meshes) > 0:
                    # vis bbox_3d
                    im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
                    
                    if self.args.display:
                        im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
                        vis.imshow(im_concat)

                    util.imwrite(im_drawn_rgb, os.path.join(save_det_path, im_name+'_3Dboxes.jpg'))
                    util.imwrite(im_topdown, os.path.join(save_det_path, im_name+'_novel.jpg'))

                    # vis bbox_2d
                    util.imwrite(img_draw_2d, os.path.join(save_det_path, im_name+'_faster2Dboxes.jpg'))

                else:
                    util.imwrite(img_draw_2d, os.path.join(save_det_path, im_name+'_faster2Dboxes.jpg'))
                    util.imwrite(im, os.path.join(save_det_path, im_name+'_3Dboxes.jpg'))

                # save 3d bbox
                util.save_json(os.path.join(save_det_path, im_name+'_detection_results.json'), save_bbox3d)

                # save 3d bbox ply
                for i, mesh in enumerate(meshes):
                    IO().save_mesh(mesh, os.path.join(save_det_path, im_name+'_bbox3D_{}.ply'.format(i)))
                
                tri_meshes = []
                for i in range(len(meshes)):
                    tri_meshes.append(trimesh.load(os.path.join(save_det_path, im_name+'_bbox3D_{}.ply'.format(i))))
                    os.remove(os.path.join(save_det_path, im_name+'_bbox3D_{}.ply'.format(i)))

                # concatentate meshes
                if len(tri_meshes):
                    scene_mesh = trimesh.util.concatenate(tri_meshes)
                    scene_mesh.export(os.path.join(save_det_path, im_name+'_scene_bbox3D.ply'))



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

    cfg = read_cfg(args)

    omni3d_det = Omni3DDet(args, cfg)
    omni3d_det.run()
    print('done')
