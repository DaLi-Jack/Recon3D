import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def img_resize(img, size):
    img = img.resize(size, Image.ANTIALIAS)
    return img


# for detection omni3d
import sys
sys.path.append('./omni3d')                 # for 3d detection, omni3d is needed
from pytorch3d.io import IO
import trimesh
import copy
from detectron2.data import transforms as T
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis


def detection(args, cfg, model):

    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')

    model.eval()

    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    
    for path in list_of_ims:

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

        dets = model(batched)[0]['instances']
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
            
            if args.display:
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
        scene_mesh = trimesh.util.concatenate(tri_meshes)
        scene_mesh.export(os.path.join(save_det_path, im_name+'_scene_bbox3D.ply'))



# for sam
def get_crop_bbox(obj_vis_masks):
    # calculate crop bbox
    height, width, _ = obj_vis_masks.shape
    vis_masks_idx = np.argwhere(obj_vis_masks == 1)
    # print(vis_masks_idx.shape)
    px = vis_masks_idx[:, 0]
    py = vis_masks_idx[:, 1]
    xmin, xmax = int(np.min(py)), int(np.max(py))       # W
    ymin, ymax = int(np.min(px)), int(np.max(px))       # H
    full_bbox_2d = [xmin, ymin, xmax, ymax]
    x_center, y_center = (xmin + xmax) // 2, (ymin + ymax) // 2

    square_length = max(xmax - xmin, ymax - ymin) + 100
    x_square_begin, y_square_begin = max(0, x_center - square_length // 2), max(0, y_center - square_length // 2)
    [x_square_end, y_square_end] = [min(width, x_square_begin + square_length), min(height, y_square_begin + square_length)]
    crop_bbox = [x_square_begin, y_square_begin, x_square_end, y_square_end]


    return crop_bbox


# for fusion scene
def get_mesh_scene(mesh_path, obj_dic):
    x_rot_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    x_rot_mat = np.eye(4)
    x_rot_mat[0:3, 0:3] = x_rot_90

    y_rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    y_rot_mat = np.eye(4)
    y_rot_mat[0:3, 0:3] = y_rot_180

    
    mesh = trimesh.load(mesh_path)
    mesh.apply_transform(x_rot_mat)
    mesh.apply_transform(y_rot_mat)

    bbox3D = np.array(obj_dic['bbox3D'])
    translation = np.array([bbox3D[0], bbox3D[1], bbox3D[2]])   # [tx, ty, tz]
    gt_length = np.array([bbox3D[3], bbox3D[4], bbox3D[5]])     # [lx, ly, lz]

    # scale mesh according to gt bbox size
    mesh_bbox = mesh.bounding_box.extents                                           # [lx, ly, lz]
    scale = gt_length / mesh_bbox
    mesh.apply_scale(scale)

    R = np.array(obj_dic['pose'])
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = translation

    mesh.apply_transform(mat)

    return mesh

