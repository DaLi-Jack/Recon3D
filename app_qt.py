import sys
import os 
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from pyvistaqt import QtInteractor, MainWindow
from PySide6.Qt3DCore import (Qt3DCore)
from PySide6.Qt3DExtras import (Qt3DExtras)

from glob import glob 
import open3d as o3d
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import plyfile


def init_open_view():
    open3d_view = o3d.visualization.VisualizerWithKeyCallback()
    open3d_view.create_window()

    # Load 3D model
    ply_path = "./output/demo/0005/shape/cabinet 0.33.ply"
    model = o3d.io.read_triangle_mesh(ply_path)

    # Add 3D model to view
    open3d_view.add_geometry(model)

    # Set up callback
    def change_view(vis):
        cam = vis.get_view_control()
        cam.rotate(10.0, 0.0)    # Rotate view by 10 degrees
    open3d_view.register_key_callback(65362, change_view)  # 65362 is key code for UP_ARROW

    # Set up view options
    open3d_view.get_render_option().point_size = 1.0
    open3d_view.set_background_color((1, 1, 1))  # Set background to white

    # Update view 
    open3d_view.update_renderer()  
    return open3d_view
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # add image show
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        # add mesh show 
        self.init_ploter()
        
        # Add image upload button
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.upload_upload)
        run_bt = QPushButton("Run")
        
        # Add image list widget (lower part)
        image_list = QListWidget()
        image_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)   
        image_list.setIconSize(QSize(500, 500))
        self.set_image_list(image_list, "./output/demo/0005/det/")
        image_list.itemClicked.connect(self.imagelist_select)
        # Set up layout
        self.upper_layout = QHBoxLayout()
        upper_layout = self.upper_layout
        upper_layout.addWidget(self.canvas)  # Add matplotlib canvas
        upper_layout.addWidget(self.plotter)
        upper_layout.addWidget(image_list)
        # image_list.setFixedHeight(upper_width*0.3)
        
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(run_bt)
        lower_layout.addWidget(upload_button)     # Add upload button
        
        layout = QVBoxLayout()
        layout.addLayout(upper_layout)
        layout.addLayout(lower_layout)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        
    def init_ploter(self):
        # create the frame
        self.frame = QFrame()
        vlayout = QVBoxLayout()
        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        # sphere = pv.Sphere()
        # self.plotter.add_mesh(sphere, show_edges=True)
        # self.plotter.reset_camera()
        # self.plotter.show()
        return 
        
    def update_mesh(self, file):
        mesh = pv.read(file)
        self.plotter.clear()
        self.plotter.add_mesh(mesh, rgb=True)
        self.plotter.reset_camera()
        

    def set_image_list(self, widget, path):
        widget.clear()
        images = glob(os.path.join(path, '*.jpg'))
        for image in images:
            item = QListWidgetItem()
            icon = QIcon(QPixmap(image))
            item.setIcon(icon)
            widget.addItem(item)
            
    def imagelist_select(self, item):
        self.now_item = item
        
        
    def update_fig(self, image_path):
        if image_path:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            image = mpimg.imread(image_path)
            ax.imshow(image)
            self.canvas.draw()
        
    def upload_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Upload Image", "", 
                                            "Image Files (*.png *.jpg *.bmp)")
        self.fig.clear()
        self.update_fig(file_path)    

        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()  
    
    sys.exit(app.exec_())