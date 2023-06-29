import sys
import os 
from PyQt5.QtWidgets import *  
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# from PyQt5.QtWidgets import QApplication
from pyvistaqt import QtInteractor,MultiPlotter
from PySide6.Qt3DCore import (Qt3DCore)
from PySide6.Qt3DExtras import (Qt3DExtras)

from glob import glob 
import open3d as o3d
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import trimesh 
import numpy as np
import pyvista as pv
import plyfile
import json 
from draw_canvas import Qpaint_canvas, Image_canvas, Brush_canvas, Position_canvas
from IPython import embed
from modellib import Robot

from module.fusion import remove_xy_rotation, get_mesh_scene
class MeshObj:
    def __init__(self, item, plotter):
        self.rot_angle = 0
        self.translate = np.array([0., 0., 0.])
        self.item = item
        self.name = self.item['name']
        self.mesh_file = self.item['mesh']
        self.plotter = plotter
        bbox3D = np.array(item['bbox3D'])
        self.init_translation = np.array([bbox3D[0], bbox3D[1], bbox3D[2]])
        self.gt_length = np.array([bbox3D[5], bbox3D[4], bbox3D[3]]) 
        
        self.size = self.gt_length[0] * self.gt_length[1] * self.gt_length[2]
        self.init_pose = self.make_transformat_from_R(np.array(item['pose']))
        self.camera_pose = np.eye(4)
        
        # self.plane = pv.Plane(center=self.init_translation, direction=[0, 1, 0], i_size=10, j_size=10)
        # self.plane.translate([0, -self.gt_length[1], 0], inplace=True)
        self.show_plane = False
        self.highlight = False

        
    def update_meshfile(self, file):
        self.mesh_file = file
    
    def get_mesh_shape(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        x_length = xmax - xmin 
        y_length = ymax - ymin 
        z_length = zmax - zmin 
        return np.array([x_length, y_length, z_length])
    
    def show_mesh_bbox(self):
        bbox = pv.Box(self.mesh.bounds)
        lines = bbox.lines
        for i in range(0, lines.shape[0], 2):
            print(lines, lines[i:i+2, :])
            self.plotter.add_lines(lines[i:i+2, :], color='red', width=3) 
            
        # return box 
        
    
    def scale_mesh(self):
        scale = self.gt_length / self.get_mesh_shape()
        self.mesh.scale(scale, inplace=True)
        
    def init_mesh(self):
        self.mesh = pv.read(self.mesh_file)
        self.mesh.translate(-np.array(self.mesh.center), inplace=True)
        self.mesh.rotate_x(-90, inplace=True)
        self.mesh.rotate_y(180, inplace=True)
        

    def make_transformat_from_R(self, R):
        mat = np.eye(4)
        mat[0:3, 0:3] = R
        return mat
        
    # def get_local_axis(self, base_vector, R, translation):
    #     vector = np.array([base_vector]).transpose(1, 0)
    #     x_rot_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    #     x_rot_mat = np.eye(4)
    #     x_rot_mat[0:3, 0:3] = x_rot_90

    #     y_rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    #     y_rot_mat = np.eye(4)
    #     y_rot_mat[0:3, 0:3] = y_rot_180
        
    #     mat = np.eye(4)
    #     mat[0:3, 0:3] = R
    #     # mat[0:3, 3] = translation
                
    #     vector = np.dot(x_rot_mat, vector)
    #     vector = np.dot(y_rot_mat, vector)
    #     vector = np.dot(mat, vector).transpose(1, 0)[0][:3]
    #     axis = pv.Line(pointa=[0, 0, 0], pointb=vector)
    #     return axis

    # def get_axis_vector(self, axis):
    #     if axis == 'x':
    #         return np.array(self.x_axis.points)[1]
    #     elif axis == "y":
    #         return np.array(self.y_axis.points)[1]
    #     elif axis == 'z':
    #         return np.array(self.z_axis.points)[1]
    #     else:
    #         return [0, 0, 1]
        
    # def update_axis(self, rotate_vector, angle):
    #     self.x_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
    #     self.y_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
    #     self.z_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
        
    # def rotate_x(self, angle):
    #     rotate_vector = self.get_axis_vector("x")
    #     self.mesh.rotate_vector(rotate_vector, angle, point=self.mesh.center, inplace=True)
    #     self.update_axis(rotate_vector, angle)


    def get_axis_vector(self, axis):
        if axis == 'x':
            return np.array(self.x_axis.points)[1]
        elif axis == "y":
            return np.array(self.y_axis.points)[1]
        elif axis == 'z':
            return np.array(self.z_axis.points)[1]
        else:
            return [0, 0, 1]
        
    def get_axis_vector(self, axis):
        if axis == 'x':
            return np.array(self.x_axis.points)[1]
        elif axis == "y":
            return np.array(self.y_axis.points)[1]
        elif axis == 'z':
            return np.array(self.z_axis.points)[1]
        else:
            return [0, 0, 1]
        
    # def rotate_y(self, angle):
    #     rotate_vector = self.get_axis_vector("y")
    #     self.mesh.rotate_vector(rotate_vector, angle, point=self.mesh.center, inplace=True)
    #     self.update_axis(rotate_vector, angle)
        
            
    def rotate_z(self, angle):
        self.rot_angle += 90
        self.update_mesh()
        
    def translate_x(self, distance):
        self.translate += np.array([distance, 0, 0])
        self.update_mesh()

        
        
    def translate_y(self, distance):
        self.translate += np.array([0, distance, 0])
        self.update_mesh()

        
        
    def translate_z(self, distance):
        self.translate += np.array([0, 0, distance])
        self.update_mesh()

        
        
    def update_mesh(self):
        # self.plotter.clear()
        self.init_mesh()
        self.mesh.rotate_y(self.rot_angle, inplace=True)
        self.scale_mesh()
        self.mesh.transform(self.init_pose, inplace=True)
        
        self.mesh.translate(self.init_translation, inplace=True)
        self.mesh.transform(self.camera_pose, inplace=True)
        # print(self.gt_length, self.translate, self.mesh.center)
        self.mesh.translate(self.translate, inplace=True)
        
        self.plotter.enable_surface_picking()
        if self.highlight:
            self.show_mesh_bbox()
            # self.plotter.add_mesh(self.mesh, color='red', name=self.name)
        
        self.plotter.add_mesh(self.mesh, rgb=True, name=self.name)
        # if self.show_plane:
        #     self.plotter.add_floor(face='-y', i_resolution=10, j_resolution=10, pad=0.2)
            # self.plotter.add_bounding_box(line_width=5, color='black')
            # self.plotter.add_mesh(self.plane, name=self.name+'_plane')

def get_items(path):
    det_path = os.path.join(path, 'det', '0004_detection_results.json')
    image_name = '0004'
    with open(det_path, "r") as f:
        det_results = json.load(f)
    sam_path = os.path.join(path, 'sam')
    shape_path = os.path.join(path, 'shape')
    det_res = os.path.join(path, 'det', f'{image_name}_3Dboxes.jpg')
    items = {}
    for det_text, obj_dic in det_results.items():
        item = {}
        for k,v in obj_dic.items():
            item[k] = v
        item['sam_res'] = os.path.join(path, 'sam', det_text + '_sam.png')
        item['sam_vis'] = os.path.join(path, 'sam', det_text + '_vis.png')
        
        item['inpaint_res'] = os.path.join(path, 'inpaint', det_text + '_inpaint.png')
        item['mask'] = os.path.join(path, 'sam', det_text + '_mask.png')
        item['mesh'] = os.path.join(path, 'shape', det_text + '.ply')
        item['mesh_world'] = os.path.join(path, 'shape', det_text + '_world.ply')
        item['diffuse_mask'] = os.path.join(path, 'sam', det_text + '_diffuse_mask.png')
        
        
        item['name'] = det_text
        items[det_text] = item
    result = {
        'test_image': './test_img/real_img/0004.png',
        'det_res': det_res,
        'items': items,
        'scene': os.path.join(path, "scene.ply"),
    }
    return result
    
# class viewer 
items = get_items("./output/demo/0004/")
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.robot = Robot()
        layout = self.create_layout()
        self.init_layout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setLayout(layout)
        self.update_layout()
        
    def init_layout(self):
        self.items_combo.clear()
        self.items = self.robot.items
        
        # self.robot.items = items
        # self.robot.set(image_path = items['test_image'])
        # self.items = self.robot.items
        # self.set_plotter_items(self.items['items'])
        
        
    def update_layout(self):
        print('update')
        if 'test_image' not in self.items.keys():
            return 
        self.canvas.update_bg(self.items['test_image'])
        if 'items' in self.items.keys() and len(list(self.items['items'].keys())) == 0:
            return
        self.detection_canvas.update_bg(self.items['det_res'])
        self.sam_prompt.update_canvas(background=self.items['test_image'])
        current_text = self.items_combo.currentText()
        # self.robot.run_fusion()
        max_size = 0
        max_item = None 
        for name, item in self.items['items'].items():
            if self.items_combo.findText(name) == -1: 
                self.items_combo.addItem(name)
            if current_text == name:
                if 'mask' in item.keys():
                    self.sam_widget.update_bg(item['sam_res'])
                    # self.sam_widget.update_canvas(background=self.items['test_image'], mask=item['mask'])
                if 'diffuse_mask' in item.keys():
                    self.inpaint_widget.update_canvas(background=item['sam_vis'])
                    self.inpaint_widget.update()
                if 'inpaint_res' in item.keys():
                    self.inpaint_output_widget.update_bg(item['inpaint_res'])
                if 'mesh_obj' in item.keys():
                    item['mesh_obj'].highlight = True 
            else:
                if 'mesh_obj' in item.keys():
                    item['mesh_obj'].highlight = False
            if 'mesh_obj' in item.keys():
                item['mesh_obj'].update_mesh()

        
    def create_layout(self):
        layout = QHBoxLayout()
        # inputlayout  
        inputlayout = QVBoxLayout()
        self.canvas = Image_canvas(size=(500, 500))
        self.detection_canvas = Image_canvas(size=(500, 500))
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.upload_upload)
        input_bt_layout = QHBoxLayout()
        
        run_bt = QPushButton("Run Recon")
        run_bt.clicked.connect(self.on_run_all)
        # run_bt
        input_bt_layout.addWidget(upload_button)
        input_bt_layout.addWidget(run_bt)
        inputlayout.addLayout(input_bt_layout, 10)  
        inputlayout.addWidget(self.canvas, 45)
        inputlayout.addWidget(self.detection_canvas, 45)
        inputlayout_empty = QHBoxLayout()
        
        inputlayout_all = QVBoxLayout()
        inputlayout_all.addLayout(inputlayout)
        inputlayout_all.addLayout(inputlayout_empty, 10)
        layout.addLayout(inputlayout_all)
        # process part
        process_layout = QVBoxLayout()
        # upper layout
        self.items_combo = QComboBox()
        self.items_combo.currentIndexChanged.connect(self.onComboChanged) 
        process_layout.addWidget(self.items_combo)
        # lower layout 
        lower_layout = QHBoxLayout()
        process_layout.addLayout(lower_layout)
        # process seg layout 
        seg_layout = QVBoxLayout()
        self.sam_prompt = Position_canvas(size=(500, 500))
        # self.sam_widget = Qpaint_canvas(size=(500, 500))
        self.sam_widget = Image_canvas(size=(500, 500))
        # self.sam_widget.update_canvas(background=self.items['test_image'])
        seg_bt_layout = QHBoxLayout()
        seg_run_bt = QPushButton("rerun segmentation")
        seg_run_bt.clicked.connect(self.on_run_seg)
        seg_save_bt = QPushButton("Save")
        seg_save_bt.clicked.connect(self.on_save_seg)
        # add text input         
        seg_bt_layout.addWidget(seg_run_bt)
        # seg_bt_layout.addWidget(seg_save_bt)
        seg_layout.addWidget(self.sam_prompt, 45)
        seg_layout.addWidget(self.sam_widget, 45)
        seg_layout.addLayout(seg_bt_layout, 10)
        lower_layout.addLayout(seg_layout, 30)
        # process inpaint layout 
        inpaint_layout = QVBoxLayout()
        self.inpaint_widget = Brush_canvas(size=(500, 500))
        self.inpaint_output_widget = Image_canvas(size=(500, 500))
        inpaint_bt_layout = QHBoxLayout()
        inpaint_run_bt = QPushButton("rerun inpainting")
        inpaint_save_bt = QPushButton("Save inpaint")
        inpaint_save_bt.clicked.connect(self.on_save_inpaint)
        inpaint_run_bt.clicked.connect(self.on_run_inpaint)
        # inpaint_text_layout = QHBoxLayout()
        self.inpaint_prompt_text = QLineEdit()
        # self.inpaint_prompt_text.setLine('please input inpaint text')

        inpaint_bt_layout.addWidget(self.inpaint_prompt_text)
        inpaint_bt_layout.addWidget(inpaint_run_bt)
        inpaint_bt_layout.addWidget(inpaint_save_bt)
        inpaint_layout.addWidget(self.inpaint_widget, 45)
        inpaint_layout.addWidget(self.inpaint_output_widget, 45)
        inpaint_layout.addLayout(inpaint_bt_layout, 10)
        lower_layout.addLayout(inpaint_layout, 30)
        # process 3d layout 
        layout_3d = QVBoxLayout()
        move_control_layout = QGridLayout()
        movex_up_bt = QPushButton('x-up', clicked=self.on_xup)
        movex_down_bt = QPushButton('x-down', clicked=self.on_xdown)
        movey_up_bt = QPushButton('y-up', clicked=self.on_yup)
        movey_down_bt = QPushButton('y-down', clicked=self.on_ydown)
        movez_up_bt = QPushButton('z-up', clicked=self.on_zup)
        movez_down_bt = QPushButton('z-down', clicked=self.on_zdown)
        rotate_up_bt = QPushButton("R+", clicked=self.on_rot_up)
        rotate_down_bt = QPushButton("R-", clicked=self.on_rot_down)
        
        x_delta = QSlider(Qt.Horizontal)
        x_delta.setRange(0, 30)
        x_delta.setValue(1)
        x_delta.setTickInterval(1)
        x_delta.setTickPosition(QSlider.TicksAbove)
        
        y_delta = QSlider(Qt.Horizontal)
        y_delta.setRange(0, 30)
        y_delta.setValue(1)
        y_delta.setTickInterval(1)
        y_delta.setTickPosition(QSlider.TicksAbove)
        
        z_delta = QSlider(Qt.Horizontal)
        z_delta.setRange(0, 30)
        z_delta.setValue(1)
        z_delta.setTickInterval(1)
        z_delta.setTickPosition(QSlider.TicksAbove)

        self.x_delta = x_delta
        self.y_delta = y_delta
        self.z_delta = z_delta
        move_control_layout.addWidget(movex_up_bt, 0, 0)
        move_control_layout.addWidget(movex_down_bt, 0, 1)
        move_control_layout.addWidget(x_delta, 0, 2)
        move_control_layout.addWidget(movey_up_bt, 1, 0)
        move_control_layout.addWidget(movey_down_bt, 1, 1)
        move_control_layout.addWidget(y_delta, 1, 2)
        move_control_layout.addWidget(movez_up_bt, 2, 0)
        move_control_layout.addWidget(movez_down_bt, 2, 1)
        move_control_layout.addWidget(z_delta, 2, 2)
        
        move_control_layout.addWidget(rotate_up_bt, 3, 0)
        move_control_layout.addWidget(rotate_down_bt, 3, 1)
        
        save_3d_bt = QPushButton("Save")
        save_3d_bt.clicked.connect(self.on_save_3d)
        move_control_layout.addWidget(save_3d_bt, 4, 0)
        move_control_layout.setVerticalSpacing(10)
        move_control_layout.setHorizontalSpacing(10)
        self.init_ploter()
        layout_3d.addWidget(self.plotter, 90)
        layout_3d.addLayout(move_control_layout, 10)
        lower_layout.addLayout(layout_3d, 40)
        
        layout.addLayout(process_layout)
        
        return layout
        
    def on_run_all(self):
        self.items_combo.clear()
        self.on_run_det()
        for name, item in self.items['items'].items():
            self.on_run_seg(name)
            # self.on_run_inpaint(name)
            self.on_run_recon(name)
        self.set_plotter_items(self.items['items'])
        self.update_layout()
        return
    
    def on_run_det(self):
        self.robot.run_det()
        self.update_layout()

    def on_run_seg(self, name=None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        item = self.items['items'][current_text]
        points = self.sam_prompt.get_positions()
        if len(points):
            self.robot.run_seg(current_text, item, True, self.sam_prompt.get_positions())
        else:
            self.robot.run_seg(current_text, item)
        self.update_layout()
        
    def on_save_seg(self, name=None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        if current_text:
            item = self.items['items'][current_text]
            if 'sam_vis' not in item.keys():
                return 
            if 'diffuse_mask' not in item.keys():
                return 
            # self.sam_widget.save(item)
            self.update_layout()
    
    def on_run_inpaint(self, name=None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        item = self.items['items'][current_text]
        prompt = self.inpaint_prompt_text.text()
        if prompt == '':
            prompt = current_text.split(' ')[0]
        self.robot.run_inpaint(current_text, prompt)
        self.on_run_recon(current_text)
        self.update_layout()
        
    def on_clear_inpaint(self, name=None):
        self.inpaint_widget.clear()
        
    def on_save_inpaint(self, name= None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        item = self.items['items'][current_text]
        self.inpaint_widget.save(item)
        self.inpaint_widget.clear()
        self.update_layout()
        
        
        
        
    def on_run_recon(self, name):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        self.robot.run_recon(current_text)
        self.update_layout()
        
    def on_save_3d(self):
        self.plotter.export_obj('./scene.obj')

    
    def on_rot_up(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.rotate_z(90)

    def on_rot_down(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.rotate_z(-90)
        
        
    def on_xup(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_x(self.x_delta.value()/10)
            
    def on_xdown(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_x(-self.x_delta.value()/10)
         
    def on_yup(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_y(self.y_delta.value()/10)


    def on_ydown(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_y(-self.y_delta.value()/10)

    def on_zup(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_z(self.z_delta.value()/10)

    def on_zdown(self):
        current_text = self.items_combo.currentText()
        if current_text:
            current_item = self.items['items'][current_text]['mesh_obj']
            current_item.translate_z(-self.z_delta.value()/10)
        
    def init_ploter(self):
        # create the frame
        self.frame = QFrame()
        self.plotter = QtInteractor(self.frame)
        self.plotter.add_axes(
                color='black',
                labels_off=False)
        # light = pv.Light(position=(0, 3, 0), focal_point=(0, 0, 0), color='white', cone_angle=90,
        #                  exponent=10, intensity=3)
        # light.show_actor()
        # self.plotter.add_light(light)
        self.plotter.set_background('white')
        self.plotter.set_viewup([0, 1, 0])
        return

    def set_plotter_items(self, items):
        self.plotter.clear()
        max_size = 0
        camera_pose = np.eye(4)
        mesh_center = np.zeros([0, 0, 0])
        max_item = None 
        for name, item in items.items():
            if 'mesh' in item.keys():
                mesh_obj = MeshObj(item, self.plotter)
                if mesh_obj.size > max_size:
                    camera_pose = mesh_obj.init_pose
                    mesh_center = -mesh_obj.init_translation
                    max_item = mesh_obj 
                    max_size = mesh_obj.size
                item['mesh_obj'] = mesh_obj
        if max_item:
            max_item.show_plane = True
        for name, item in items.items():
            if 'mesh' in item.keys():
                mesh_obj = item['mesh_obj']       
                mesh_obj.camera_pose = np.linalg.inv(camera_pose)
                mesh_obj.camera_pose[:3, -1:] = np.array([mesh_center]).transpose(1, 0)
                mesh_obj.update_mesh()
        self.plotter.add_floor(face='-y', i_resolution=10, j_resolution=10, pad=0.2)

    # add sam process to fix sam result 
    def onComboChanged(self):
        self.sam_prompt.clear()
        self.update_layout()

    def imagelist_select(self, item):
        self.now_item = item
        
    def upload_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Image", "", 
                                            "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_path:
            self.robot.items = {}
            self.items = self.robot.items
            
            self.init_layout()
            self.canvas.update_bg(file_path)
            self.items['test_image'] = file_path
            self.robot.set(image_path = file_path)
            
            # # init with data
            # self.items = items
            # self.robot.items = items
            # self.set_plotter_items(self.items['items'])
            # self.update_layout()


        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()  
    
    sys.exit(app.exec_())