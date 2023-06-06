import trimesh
import os
import numpy as np


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
    gt_length = np.array([bbox3D[5], bbox3D[4], bbox3D[3]])     # [lx, ly, lz]

    # scale mesh according to gt bbox size
    mesh_bbox = mesh.bounding_box.extents                       # [lx, ly, lz]
    scale = gt_length / mesh_bbox
    mesh.apply_scale(scale)

    R = np.array(obj_dic['pose'])
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = translation

    mesh.apply_transform(mat)

    return mesh


def fusion_scene(detection_results, save_root_path, save_shape_path):
    scene_mesh = []
    for det_text in detection_results:
        obj_dic = detection_results[det_text]
        mesh_path = os.path.join(save_shape_path, f'{det_text}.ply')

        # mesh = get_mesh_scene_inst(mesh_path, obj_dic)
        mesh = get_mesh_scene(mesh_path, obj_dic)
        scene_mesh.append(mesh)
    
    scene_mesh = trimesh.util.concatenate(scene_mesh)
    scene_mesh.export(os.path.join(save_root_path, 'scene.ply'))
