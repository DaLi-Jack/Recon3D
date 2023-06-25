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
from draw_canvas import Qpaint_canvas, get_maskcontours, Image_canvas, Brush_canvas
from IPython import embed
from modellib import Robot

from module.fusion import remove_xy_rotation, get_mesh_scene
class MeshObj:
    def __init__(self, item, plotter):
        self.item = item
        self.name = self.item['name']
        self.mesh = pv.read(item['mesh_world'])
        self.plotter = plotter
        bbox3D = np.array(item['bbox3D'])
        self.translation = np.array([bbox3D[0], bbox3D[1], bbox3D[2]])
        self.gt_length = np.array([bbox3D[5], bbox3D[4], bbox3D[3]]) 
        self.size = self.gt_length[0] * self.gt_length[1] * self.gt_length[2]
        R = np.array(item['pose'])
        R = remove_xy_rotation(R)
        self.x_axis = self.get_local_axis([1, 0, 0, 1], R, self.translation)
        self.y_axis = self.get_local_axis([0, 1, 0, 1], R, self.translation)
        self.z_axis = self.get_local_axis([0, 0, 1, 1], R, self.translation)
        self.plotter.add_mesh(self.mesh,rgb=True, name=item['name'])
        self.plane = pv.Plane(center=self.translation, direction=self.get_axis_vector('z'), i_size=10, j_size=10)
        self.plane.translate([0, -self.gt_length[1]/2, 0], inplace=True)
        # plotter.add_mesh(self.x_axis, name=f"{item['name']}_x")
        # plotter.add_mesh(self.y_axis, name=f"{item['name']}_y")
        # plotter.add_mesh(self.z_axis, name=f"{item['name']}_z")
        
    def show_plane(self):
        self.plotter.add_mesh(self.plane, name=self.name+'_plane')
        
    def to_world(self):
        x_rot_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        x_rot_mat = np.eye(4)
        x_rot_mat[0:3, 0:3] = x_rot_90

        y_rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        y_rot_mat = np.eye(4)
        y_rot_mat[0:3, 0:3] = y_rot_180
        self.mesh.transform(x_rot_mat)
        self.mesh.transform(y_rot_mat)
        mesh_bbox = self.mesh.dimensions[:3] 
        scale = self.gt_length / mesh_bbox
        
        
    def highlight(self):
        self.plotter.add_mesh(self.mesh, color='red', name=self.item['name'])
        
    def nohighlight(self):
        self.plotter.add_mesh(self.mesh,rgb=True, name=self.item['name'])
        
        
    def get_local_axis(self, base_vector, R, translation):
        vector = np.array([base_vector]).transpose(1, 0)
        x_rot_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        x_rot_mat = np.eye(4)
        x_rot_mat[0:3, 0:3] = x_rot_90

        y_rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        y_rot_mat = np.eye(4)
        y_rot_mat[0:3, 0:3] = y_rot_180
        
        mat = np.eye(4)
        mat[0:3, 0:3] = R
        # mat[0:3, 3] = translation
                
        vector = np.dot(x_rot_mat, vector)
        vector = np.dot(y_rot_mat, vector)
        vector = np.dot(mat, vector).transpose(1, 0)[0][:3]
        axis = pv.Line(pointa=[0, 0, 0], pointb=vector)
        return axis
    
    # def get_bbox(self, R, translation):
        
    

    def get_axis_vector(self, axis):
        if axis == 'x':
            return np.array(self.x_axis.points)[1]
        elif axis == "y":
            return np.array(self.y_axis.points)[1]
        elif axis == 'z':
            return np.array(self.z_axis.points)[1]
        else:
            return [0, 0, 1]
        
    def update_axis(self, rotate_vector, angle):
        self.x_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
        self.y_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
        self.z_axis.rotate_vector(rotate_vector, angle, point=[0, 0, 0], inplace=True)
        
    def rotate_x(self, angle):
        rotate_vector = self.get_axis_vector("x")
        self.mesh.rotate_vector(rotate_vector, angle, point=self.mesh.center, inplace=True)
        self.update_axis(rotate_vector, angle)
        
        
    def rotate_y(self, angle):
        rotate_vector = self.get_axis_vector("y")
        self.mesh.rotate_vector(rotate_vector, angle, point=self.mesh.center, inplace=True)
        self.update_axis(rotate_vector, angle)
        
            
    def rotate_z(self, angle):
        rotate_vector = self.get_axis_vector("z")
        self.mesh.rotate_vector(rotate_vector, angle, point=self.mesh.center, inplace=True)
        self.update_axis(rotate_vector, angle)
        
    def translate_x(self, distance):
        des = self.get_axis_vector("x")*distance
        self.mesh.translate(des, inplace=True)
        
    def translate_y(self, distance):
        des = self.get_axis_vector("y")*distance
        self.mesh.translate(des, inplace=True)
    def translate_z(self, distance):
        des = self.get_axis_vector("z")*distance
        self.mesh.translate(des, inplace=True)

def get_items(path):
    det_path = os.path.join(path, 'det', '0005_detection_results.json')
    image_name = '0005'
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
        'test_image': './test_img/real_img/0005.png',
        'det_res': det_res,
        'items': items,
        'scene': os.path.join(path, "scene.ply"),
    }
    return result
    
# class viewer 
items = get_items("./output/demo/0005/")
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.robot = Robot()
        layout = self.create_layout()
        self.init_layout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setLayout(layout)
        # self.update_layout()
        
    def init_layout(self):
        self.items_combo.clear()
        # self.robot.items = items
        self.robot.set(image_path = items['test_image'])
        
        self.items = self.robot.items
        
        
    def update_layout(self):
        # self.items = items
        print('update')
        self.canvas.update_bg(self.items['test_image'])
        if 'items' in self.items.keys() and len(list(self.items['items'].keys())) == 0:
            return
        self.detection_canvas.update_bg(self.items['det_res'])
        self.set_plotter_items(self.items['items'])
        self.sam_prompt.update_canvas(background=self.items['test_image'])
        current_text = self.items_combo.currentText()
        self.robot.run_fusion()
        max_size = 0
        max_item = None 
        for name, item in self.items['items'].items():
            if self.items_combo.findText(name) == -1: 
                self.items_combo.addItem(name)
            if current_text == name:
                
                if 'mask' in item.keys():
                    self.sam_widget.update_canvas(background=self.items['test_image'], mask=item['mask'])
                if 'diffuse_mask' in item.keys():
                    self.inpaint_widget.update_canvas(background=item['sam_vis'])
                    self.inpaint_widget.update()
                if 'inpaint_res' in item.keys():
                    self.inpaint_output_widget.update_bg(item['inpaint_res'])
                if 'mesh_obj' in item.keys():
                    item['mesh_obj'].highlight()
            if 'mesh_obj' in item.keys():
                if item['mesh_obj'].size > max_size:
                    max_size = item['mesh_obj'].size
                    max_item = item['mesh_obj']
        if max_item:
            max_item.show_plane()
                
        # for name, item in self.items['items'].items():
        #     self.items_combo.addItem(name)
        
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
        self.sam_prompt = Qpaint_canvas(size=(500, 500))
        self.sam_widget = Qpaint_canvas(size=(500, 500))
        # self.sam_widget.update_canvas(background=self.items['test_image'])
        seg_bt_layout = QHBoxLayout()
        seg_run_bt = QPushButton("rerun segmentation")
        seg_run_bt.clicked.connect(self.on_run_seg)
        seg_save_bt = QPushButton("Save")
        seg_save_bt.clicked.connect(self.on_save_seg)
        # add text input         
        seg_bt_layout.addWidget(seg_run_bt)
        seg_bt_layout.addWidget(seg_save_bt)
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
        inpaint_save_bt = QPushButton("Save")
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
        self.on_run_det()
        for name, item in self.items['items'].items():
            self.on_run_seg(name)
            self.on_run_inpaint(name)
            self.on_run_recon(name)
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
            self.sam_widget.save(item)
            self.update_layout()
    
    def on_run_inpaint(self, name=None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        item = self.items['items'][current_text]
        prompt = self.inpaint_prompt_text.text()
        self.robot.run_inpaint(current_text, prompt)
        self.on_run_recon(current_text)
        self.update_layout()
        
    def on_save_inpaint(self, name=None):
        if name:
            current_text = name
        else:
            current_text = self.items_combo.currentText()
        item = self.items['items'][current_text]
        if 'diffuse_mask' not in item.keys():
            return 
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
        self.plotter.add_axes(labels_off=False)
        self.plotter.set_background('white')
        self.plotter.view_vector([1, -1, 1], [0, -1, 0])
        return
    
    def get_up_vector(self, R, translation):
        vector = np.array([[0, 0, 1, 1]]).transpose(1, 0)
        x_rot_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        x_rot_mat = np.eye(4)
        x_rot_mat[0:3, 0:3] = x_rot_90

        y_rot_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        y_rot_mat = np.eye(4)
        y_rot_mat[0:3, 0:3] = y_rot_180
        
        vector = np.dot(x_rot_mat, vector)
        vector = np.dot(y_rot_mat, vector)


        mat = np.eye(4)
        mat[0:3, 0:3] = R
        mat[0:3, 3] = translation
        
        vector = np.dot(mat, vector).transpose(1, 0)[0][:3]
        return vector

    def set_plotter_items(self, items):
        for name, item in items.items():
            # if 'mesh_obj' in item.keys():
            #     continue
            if 'mesh_world' in item.keys():
                mesh_obj = MeshObj(item, self.plotter)
                item['mesh_obj'] = mesh_obj
        
    def update_mesh(self, file):
        mesh = pv.read(file)
        self.plotter.clear()
        self.plotter.add_mesh(mesh, rgb=True)
        self.plotter.reset_camera()
    

    # add sam process to fix sam result 
    def onComboChanged(self):
        select_text = self.items_combo.currentText()
        # item = self.items['items'][text]
        # item_mesh_obj = item['mesh_obj']
        self.update_layout()
        for text, item in self.items['items'].items():
            if text == select_text:
                self.sam_prompt.update_canvas(background=self.items['test_image'])
                if 'mask' in item.keys():
                    self.sam_widget.update_canvas(background=self.items['test_image'], mask=item['mask'])
                if 'diffuse_mask' in item.keys():
                    self.inpaint_widget.update_canvas(background=item['sam_vis'])
                    self.inpaint_widget.update()
                if 'inpaint_res' in item.keys():
                    self.inpaint_output_widget.update_bg(item['inpaint_res'])
                if 'mesh_obj' in item.keys():
                    item['mesh_obj'].highlight()
            else:
                if 'mesh_obj' in item.keys():
                    item['mesh_obj'].nohighlight()
        
            
    def imagelist_select(self, item):
        self.now_item = item
        
    def upload_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Image", "", 
                                            "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.init_layout()
            self.canvas.update_bg(file_path)
            self.items['test_image'] = file_path
            self.robot.set(image_path = file_path)
            self.items = self.robot.items


        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()  
    
    sys.exit(app.exec_())