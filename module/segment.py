import logging
import os
import torch
import argparse
import sys
sys.path.append("../")                        # add root path to PYTHONPATH
sys.path.append(os.getcwd())                  # add current path to PYTHONPATH
sys.path.append("./segment-anything/")        # add segment-anything to PYTHONPATH
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from copy import deepcopy
import json

from segment_anything import sam_model_registry, SamPredictor


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



class SAM():
    def __init__(self, args, device):
        sam_checkpoint = args.sam_checkpoint
        model_type = args.model_type
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)


    def set_image(self, image):
        self.image = image
        self.predictor.set_image(image)


    def run(self, input_box, det_text, save_sam_path, use_inpainting):
        obj_vis_masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # save object visable masks
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        show_mask(obj_vis_masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.savefig(os.path.join(save_sam_path, det_text+'_sam.png'))
        # plt.show()

        obj_vis_masks = obj_vis_masks.transpose(1, 2, 0)  # H x W x 1

        # get crop bbox
        crop_bbox = get_crop_bbox(obj_vis_masks)
        [x_square_begin, y_square_begin, x_square_end, y_square_end] = crop_bbox

        # save crop bbox in image
        plt.figure(figsize=(10,10))
        plt.imshow(self.image)
        show_box(crop_bbox, plt.gca())
        plt.savefig(os.path.join(save_sam_path, det_text+'_crop_bbox.png'))
        # plt.show()

        # get vis object image, use PIL
        # img = Image.open(img_path)
        img = deepcopy(self.image)
        img_np = np.array(img)
        img_np = img_np[:, :, :3]               # remove alpha channel(if have)
        height, width, _ = img_np.shape

        # save vis object image
        save_vis_img = np.ones((height, width, 3), dtype=np.uint8) * 225
        save_vis_img[obj_vis_masks[:, :, 0] == 1] = img_np[obj_vis_masks[:, :, 0] == 1]
        save_vis_img = save_vis_img[y_square_begin:y_square_end, x_square_begin:x_square_end, :]            # crop square image
        save_vis_img = Image.fromarray(save_vis_img)
        save_vis_img.save(os.path.join(save_sam_path, det_text+'_vis.png'))
        cv2.imwrite(os.path.join(save_sam_path, det_text+'_mask.png'), obj_vis_masks*255)
        # save_vis_img.show()

        if use_inpainting:
            # get inpainting mask, now set the bbox 2d as diffuser mask temporarily
            bbox_mask = np.zeros((height, width, 1), dtype=np.uint8)
            bbox_mask[int(input_box[1]):int(input_box[3]), int(input_box[0]):int(input_box[2])] = 1

            diffuser_mask = bbox_mask - obj_vis_masks
            crop_diffuse_mask = diffuser_mask[y_square_begin:y_square_end, x_square_begin:x_square_end, :]            # setting crop, must be square
            crop_diffuse_mask = crop_diffuse_mask[:, :, 0]
            crop_diffuse_mask = Image.fromarray(crop_diffuse_mask * 255)
            crop_diffuse_mask.save(os.path.join(save_sam_path, det_text+'_diffuse_mask.png'))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sam_checkpoint", type=str, default='segment-anything/checkpoints/sam_vit_h_4b8939.pth', help="path to sam checkpoint file")
    parser.add_argument("--model_type", type=str, default='vit_h', help="sam model type")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = SAM(args, device)

    # load image
    img_path = 'test_img/real_img/000738_rgb_003787.jpeg'
    im_name = '000738_rgb_003787'
    output_dir = '/home/nijunfeng/mycode/recon3D/output/test_det'
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam.set_image(image)

    save_root_path = os.path.join(output_dir, im_name)
    if not os.path.exists(save_root_path):
        raise ValueError('No detection results for {}'.format(im_name))
    save_sam_path = os.path.join(save_root_path, 'sam')
    os.makedirs(save_sam_path, exist_ok=True)
    detection_results = json.load(open(os.path.join(save_root_path, 'det', im_name+'_detection_results.json'), 'r'))
    for det_text in detection_results:
        obj_dic = detection_results[det_text]
        input_box = np.array(obj_dic['bbox2D'])
        sam.run(input_box, det_text, save_sam_path, img_path, use_inpainting=True)

    print('done')