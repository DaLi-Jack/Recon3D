import sys
from PyQt5.QtWidgets import *  
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import PyQt5
import json
import cv2 
from IPython import embed
from PIL import Image
import numpy as np

class Position_canvas(QWidget):
    def __init__(self, size, background=None):
        super().__init__()

        self.painter = None
        self.background = None
        self.height, self.width = size
        self.ratio = 1
        self.setFixedSize(self.height, self.width)
        self.polygon = []  
        if background is not None:
            self.background = QImage(background)
            self.ratio_bg()
        self.temp_position = None
        self.temp_selected = -1
        self.point_size = 10 


    def set_point_size(self, point_size):
        self.point_size = point_size
        
    def clear(self):
        self.polygon = []
        
    def get_positions(self):
        points = np.array([[point.x(), point.y()]for point in self.polygon])*self.ratio
        return points
        
    def get_annotation(self):
        return self.polygon
    
    def update_canvas(self, background=None):
        if background is not None:
            self.background = QImage(background)
            self.ratio_bg()
        self.update()
    
    def ratio_bg(self):
        orig_w = self.background.width()
        orig_h = self.background.height()
        self.ratio = max(orig_w / self.rect().width(), orig_h / self.rect().height())
        self.background = self.background.scaled(orig_w/ self.ratio, orig_h/self.ratio)
        # self.polygon = [QPoint(p.x()/self.ratio, p.y()/self.ratio) for p in self.polygon]
        
    def mousePressEvent(self, event):
        if self.in_polygon(event):
            self.temp_selected, self.temp_position = self.in_polygon(event)
        self.update()
        
    def in_polygon(self, event):
        for index, point in enumerate(self.polygon):
            half = int(self.point_size / 2 )
            rect = QRect(int(point.x() - half), 
                         int(point.y() - half), 
                         self.point_size, self.point_size)
                         
            if rect.contains(event.pos()):
                return index, point
        return -1, event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            pos = self.safe_pos(event)
            self.temp_position.setX(pos.x())
            self.temp_position.setY(pos.y())
            self.update()  
            
    def safe_pos(self, event):
        if self.background:
            position = QPoint(min(max(0, event.pos().x()), self.background.width()), 
                              min(max(0, event.pos().y()), self.background.height())) 
        else:
            position = QPoint(min(max(0, event.pos().x()), self.width), 
                              min(max(0, event.pos().y()), self.height))
        return position
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if len(self.polygon) == 0 or self.temp_selected == -1:
                self.polygon.append(self.safe_pos(event))  
            self.temp_position = None
            self.temp_selected = -1
            self.update()  
        if event.button() == Qt.RightButton:
            if len(self.polygon) > 0 and self.temp_selected > -1:
                self.polygon.pop(self.temp_selected)
            self.temp_position = None
            self.temp_selected = -1
            self.update()
    # 绘制界面

    def paintEvent(self, event):    
        painter = QPainter(self)
        self.painter = painter
        if self.background is not None:
            painter.drawImage(0, 0, self.background)  
        else:
            painter.fillRect(self.rect(), Qt.white)
        if self.temp_position is not None:
            self.draw_square(self.temp_position, self.point_size)  
        if self.polygon:
            for index, point in enumerate(self.polygon):
                if index == self.temp_selected:
                    point_size = self.point_size*1.5
                else:
                    point_size = self.point_size
                self.draw_square(point, point_size)
        painter.end()

    def draw_square(self, point, size):
        half = int(size / 2)
        rect = QRect(int(point.x() - half), int(point.y() - half), size, size)
        self.painter.fillRect(rect, Qt.green)
    
    # def get_bbox(self):
    #     if self.background:
    #         win_width = self.background.width()
    #         win_height = self.background.height()
    #     else:
    #         win_width = self.width
    #         win_height = self.height 
    #     data = np.array([[point.x(),point.y()] for point in self.polygon])
    #     x_min = np.min(data[:, 0])
    #     x_max = np.max(data[:, 0])

    #     y_min = np.min(data[:, 1])
    #     y_max = np.max(data[:, 1])        
    #     width, height = x_max - x_min, y_max - y_min 
    #     pad_w = width *0.1 
    #     pad_h = height *0.1 
    #     x_min = max(0, x_min - pad_w)
    #     x_max = min(x_max + pad_w, win_width)
    #     y_min = max(0, y_min - pad_h)
    #     y_max = min(y_max + pad_h, win_height)
    #     width, height = x_max - x_min, y_max - y_min 
    #     return x_min, y_min, width, height
    
    # def save(self, item):
    #     save_image_path = item['sam_vis'].replace('vis.png', 'vis_fix.png')
    #     item['sam_vis'] = save_image_path
    #     save_mask_path = item['diffuse_mask'].replace('diffuse_mask.png', 'diffuse_mask_fix.png')
    #     item['diffuse_mask'] = save_mask_path
    #     if self.mask and len(self.polygon) > 2 and self.background:
    #         x, y, w, h = self.get_bbox()
    #         region = QRect(x, y, w, h)
    #         cropped_mask = self.mask.copy(region)
    #         rgbMask = cropped_mask.convertToFormat(QImage.Format_Grayscale8)
    #         rgbMask.save(save_mask_path)
    #         if self.background:
    #             full_mask = self.mask.copy(QRect(0, 0, self.background.width(), self.background.height())).convertToFormat(QImage.Format_Grayscale8)
    #         else:
    #             full_mask = self.mask.copy(QRect(0, 0, self.width, self.height)).convertToFormat(QImage.Format_Grayscale8)
                
    #         # full_mask.save(item['mask'])
    #         cropped_bg = self.background.copy(region)
    #         cropped_bg.save(save_image_path)
            
    #         image = cv2.imread(save_image_path)
    #         mask_image = np.ones_like(image)*255
    #         mask = cv2.imread(save_mask_path)
    #         mask_image[mask[:, :, :] == [255, 255, 255]] = image[mask[:, :, :] == [255, 255, 255]]
    #         cv2.imwrite(save_image_path, mask_image)
    #         cv2.imwrite(save_mask_path, np.logical_not(mask.astype(np.bool))*255)
            

class Brush_canvas(QWidget):
    def __init__(self, size, background=None):
        super().__init__()
        self.background = background
        if self.background is not None:
            self.background = QImage(background)
 
        
        self.brush_size = 40               # 笔刷大小
        self.brush_color = QColor(255, 255, 255) # 笔刷颜色
        # self.brush_color.setAlpha(100)
        self.brush = QBrush(self.brush_color)
        
        
        self.opacity = 0.5                # 蒙层透明度
        self.setWindowTitle('Inpaint Tool')
        # self.resize(600, 400)
        self.draw_mask = False
        self.current_pos = None 
        self.height, self.width = size
        self.setFixedSize(self.height, self.width)

        if self.background:
            self.ratio_bg() 
        self.mask = QImage(self.size(), QImage.Format_ARGB32)
        self.mask.fill(Qt.transparent)   
    def update_canvas(self, background):
        self.background = QImage(background)
        self.ratio_bg()
        self.current_pos = None
        self.draw_mask = False 
        self.update()
        
    def clear(self):
        self.mask = QImage(self.size(), QImage.Format_ARGB32)
        self.mask.fill(Qt.transparent)  
        
    def save_mask(self, path=None):
        if self.background:
            if path:
                self.background.save(path)       # 保存当前 mask
            else:
                self.background.save('mask.png')
            
    def ratio_bg(self):
        orig_w = self.background.width()
        orig_h = self.background.height()
        self.ratio = max(orig_w / self.rect().width(), orig_h / self.rect().height())
        self.background = self.background.scaled(orig_w/ self.ratio, orig_h/self.ratio)

    def paint_mask(self):
        if self.background:
            painter = QPainter(self.mask)
        else:
            painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.brush)
        half = int(self.brush_size / 2)
        painter.drawEllipse(self.current_pos.x()-half, self.current_pos.y()-half, self.brush_size, self.brush_size)
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.background is not None:
            painter.drawImage(0, 0, self.background)
        else:
            painter.fillRect(self.rect(), Qt.white)
        if self.current_pos:
            self.paint_mask()    
            painter.drawImage(0, 0, self.mask)
        painter.end()
        self.save()
        
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.current_pos = self.safe_pos(event)
            
    def mouseReleaseEvent(self, event):
        self.current_pos = None
           
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.current_pos = self.safe_pos(event)
            self.draw_mask = True
            self.update()
            
        self.last_pos = event.pos()
        self.update()  # 更新界面
        
    def mouseReleaseEvent(self, event):
        self.last_pos = None
        
    def safe_pos(self, event):
        if self.background:
            position = QPoint(min(max(0, event.pos().x()), self.background.size().width()), 
                              min(max(0, event.pos().y()), self.background.size().height())) 
        else:
            position = QPoint(min(max(0, event.pos().x()), self.width), 
                              min(max(0, event.pos().y()), self.height))
        return position
    
    def save(self, item=None):
        if self.background:
            mask = self.mask.copy(QRect(0, 0, self.background.width(), self.background.height())).convertToFormat(QImage.Format_Grayscale8)
        else:
            mask = self.mask.copy(QRect(0, 0, self.width, self.height)).convertToFormat(QImage.Format_Grayscale8)
        mask.save('mask.png')
        # item = {
        #     'diffuse_mask': '/home/zyw/data/repo_common/Recon3D/output/demo/0005/sam/sofa 0.96_diffuse_mask_fix.png',
        # }
        if item:
            diffuse_mask = cv2.imread(item['diffuse_mask'])
            inpaint_mask = cv2.imread('mask.png')
            inpaint_mask = cv2.resize(inpaint_mask, diffuse_mask.shape[:2][::-1])
            # white= np.ones_like(inpaint_mask)*255
            # inpaint_mask[inpaint_mask[:, :, :]==self.brush_color] = white[inpaint_mask[:, :, :]==self.brush_color]
            cv2.imwrite(item['diffuse_mask'], inpaint_mask)
        
class Image_canvas(QWidget):
    def __init__(self, size, image=None):
        super().__init__()

        self.painter = None
        self.background = None
        if image is not None:
            self.background = QImage(image)
        
        self.height, self.width = size
        self.setFixedSize(self.height, self.width)
        if self.background:
            self.ratio_bg()
            self.update()
        
    def ratio_bg(self):
        orig_w = self.background.width()
        orig_h = self.background.height()
        self.ratio = max(orig_w / self.rect().width(), orig_h / self.rect().height())
        self.background = self.background.scaled(orig_w/ self.ratio, orig_h/self.ratio)
        
    def paintEvent(self, event):
        # 在窗口绘制图像
        self.painter = QPainter(self)
        if self.background is not None:
            self.painter.drawImage(0, 0, self.background)
        else:
            self.painter.fillRect(self.rect(), Qt.white)
        self.painter.end()
        
    def update_bg(self, image):
        if image:
            self.background = QImage(image)
            self.ratio_bg()
            self.update()
        
class Qpaint_canvas(QWidget):
    def __init__(self, size, background=None, mask=None):
        super().__init__()

        self.painter = None
        self.background = None
        self.height, self.width = size
        self.ratio = 1
        self.setFixedSize(self.height, self.width)
        self.polygon = []  
        contours = []
        self.mask_path = mask 
        
        # 显示背景图
        if mask is not None:
            contours = get_maskcontours(mask)
            
        if background is not None:
            self.background = QImage(background)
            self.ratio_bg()
        self.temp_position = None
        self.temp_selected = -1
        self.point_size = 10 
        # 当前标注的多边形
        if len(contours):
            self.polygon = [QPoint(p[0]/self.ratio, p[1]/self.ratio) for p in contours]
        self.mask = QImage(self.size(), QImage.Format_ARGB32)
        self.mask_brush = QBrush(QColor(0, 255, 0, 128))
        self.mask.fill(Qt.transparent)    
        
        # self.brush_size = 20               # 笔刷大小
        # self.brush_color = QColor(255, 0, 0) # 笔刷颜色
        # self.brush_color.setAlpha(100)
        # self.brush = QBrush(self.brush_color)


    def set_point_size(self, point_size):
        self.point_size = point_size
        
    def clear(self):
        self.polygon = []
        
    def get_annotation(self):
        return self.polygon
    
    def update_canvas(self, background=None, mask=None):
        if mask is not None:
            contours = get_maskcontours(mask)
        if background is not None:
            # self.background = QPixmap(background)
            self.background = QImage(background)
            self.ratio_bg()
        if mask and len(contours):
            self.polygon = [QPoint(p[0]/self.ratio, p[1]/self.ratio) for p in contours]
        self.update()
    
    def ratio_bg(self):
        orig_w = self.background.width()
        orig_h = self.background.height()
        self.ratio = max(orig_w / self.rect().width(), orig_h / self.rect().height())
        self.background = self.background.scaled(orig_w/ self.ratio, orig_h/self.ratio)
        self.polygon = [QPoint(p.x()/self.ratio, p.y()/self.ratio) for p in self.polygon]
        # self.mask = QImage(self.size(), QImage.Format_ARGB32)
        # self.mask.fill(Qt.transparent)    
        
    def mousePressEvent(self, event):
        if self.in_polygon(event):
            self.temp_selected, self.temp_position = self.in_polygon(event)
        self.update()
        
    def in_polygon(self, event):
        for index, point in enumerate(self.polygon):
            half = int(self.point_size / 2 )
            rect = QRect(int(point.x() - half), 
                         int(point.y() - half), 
                         self.point_size, self.point_size)
                         
            if rect.contains(event.pos()):
                return index, point
        return -1, event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            pos = self.safe_pos(event)
            self.temp_position.setX(pos.x())
            self.temp_position.setY(pos.y())
            self.update()  
            
    def add_polygon(self, event):
        min_dist = 1000000  # 初始最小距离
        min_idx = -2            # 初始最近点索引 
        if len(self.polygon)==0:
            self.polygon.append(self.safe_pos(event))
            return
        pos = self.safe_pos(event)
        for i, point in enumerate(self.polygon):
        # 计算点与鼠标位置的距离
            dist = (point.x() - pos.x())**2 + (point.y() - pos.y())**2
            
            # 如果距离更小,更新最小距离和最近点索引
            if dist < min_dist: 
                min_dist = dist
                min_idx = i 
        if min_idx > -1:
            self.polygon.insert(min_idx, pos)
            
    def safe_pos(self, event):
        if self.background:
            position = QPoint(min(max(0, event.pos().x()), self.background.width()), 
                              min(max(0, event.pos().y()), self.background.height())) 
        else:
            position = QPoint(min(max(0, event.pos().x()), self.width), 
                              min(max(0, event.pos().y()), self.height))
        return position
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if len(self.polygon) == 0 or self.temp_selected == -1:
                # self.polygon.append(event.pos())  
                self.add_polygon(event)      
            self.temp_position = None
            self.temp_selected = -1
            self.update()  
        if event.button() == Qt.RightButton:
            if len(self.polygon) > 0 and self.temp_selected > -1:
                self.polygon.pop(self.temp_selected)
            self.temp_position = None
            self.temp_selected = -1
            self.update()
    # 绘制界面

    def paintEvent(self, event):    
        painter = QPainter(self)
        self.painter = painter
        if self.background is not None:
            painter.drawImage(0, 0, self.background)  
        else:
            painter.fillRect(self.rect(), Qt.white)
        if self.temp_position is not None:
            self.draw_square(self.temp_position, self.point_size)  
        if self.polygon:
            for index, point in enumerate(self.polygon):
                if index == self.temp_selected:
                    point_size = self.point_size*1.5
                else:
                    point_size = self.point_size
                self.draw_square(point, point_size)
            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine)) 
            painter.setBrush(self.mask_brush)
            painter.drawPolygon(*self.polygon) 
            self.mask = QImage(self.size(), QImage.Format_ARGB32)
            self.mask.fill(Qt.transparent)   
            mask_painter = QPainter(self.mask)
            brush = QBrush(QColor(255, 255, 255, 128))
            mask_painter.setBrush(brush)
            mask_painter.drawPolygon(*self.polygon)
            mask_painter.end()
        painter.drawImage(0, 0, self.mask)
        painter.end()

    def draw_square(self, point, size):
        half = int(size / 2)
        rect = QRect(int(point.x() - half), int(point.y() - half), size, size)
        self.painter.fillRect(rect, Qt.green)
    
    def get_bbox(self):
        if self.background:
            win_width = self.background.width()
            win_height = self.background.height()
        else:
            win_width = self.width
            win_height = self.height 
        data = np.array([[point.x(),point.y()] for point in self.polygon])
        x_min = np.min(data[:, 0])
        x_max = np.max(data[:, 0])

        y_min = np.min(data[:, 1])
        y_max = np.max(data[:, 1])        
        width, height = x_max - x_min, y_max - y_min 
        pad_w = width *0.1 
        pad_h = height *0.1 
        x_min = max(0, x_min - pad_w)
        x_max = min(x_max + pad_w, win_width)
        y_min = max(0, y_min - pad_h)
        y_max = min(y_max + pad_h, win_height)
        width, height = x_max - x_min, y_max - y_min 
        return x_min, y_min, width, height
    
    def save(self, item):
        save_image_path = item['sam_vis'].replace('vis.png', 'vis_fix.png')
        item['sam_vis'] = save_image_path
        save_mask_path = item['diffuse_mask'].replace('diffuse_mask.png', 'diffuse_mask_fix.png')
        item['diffuse_mask'] = save_mask_path
        if self.mask and len(self.polygon) > 2 and self.background:
            x, y, w, h = self.get_bbox()
            region = QRect(x, y, w, h)
            cropped_mask = self.mask.copy(region)
            rgbMask = cropped_mask.convertToFormat(QImage.Format_Grayscale8)
            rgbMask.save(save_mask_path)
            if self.background:
                full_mask = self.mask.copy(QRect(0, 0, self.background.width(), self.background.height())).convertToFormat(QImage.Format_Grayscale8)
            else:
                full_mask = self.mask.copy(QRect(0, 0, self.width, self.height)).convertToFormat(QImage.Format_Grayscale8)
                
            # full_mask.save(item['mask'])
            cropped_bg = self.background.copy(region)
            cropped_bg.save(save_image_path)
            
            image = cv2.imread(save_image_path)
            mask_image = np.ones_like(image)*255
            mask = cv2.imread(save_mask_path)
            mask_image[mask[:, :, :] == [255, 255, 255]] = image[mask[:, :, :] == [255, 255, 255]]
            cv2.imwrite(save_image_path, mask_image)
            cv2.imwrite(save_mask_path, np.logical_not(mask.astype(np.bool))*255)
            
                
        
def get_maskcontours(path, size=20):
    mask = Image.open(path)#.astype(bool).astype(np.uint8)
    # mask.show()
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # 二值化
    contours, _ = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0 
    max_idx = 0

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)  
        
        if area > max_area:
            max_area = area
            max_idx = idx 
    largest_contour = contours[max_idx]
    num_points = len(largest_contour)
    draw_contour_index = np.random.choice([i for i in range(num_points)], size, replace=False)
    draw_contour = largest_contour[draw_contour_index, 0, :]
    return largest_contour[:, 0, :]
    # return draw_contour
    
    
class MainWindow(QWidget): 
    def __init__(self):
        super().__init__()  
        
        window = Qpaint_canvas(size=(500, 500), background='./test_img/real_img/0005.png', contours=[])
        
        layout = QVBoxLayout()
        layout.addWidget(window) 
        
        self.setLayout(layout)
        
        # self.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Position_canvas(size=(500, 500), background='./test_img/real_img/0005.png')
    window.show()  
    sys.exit(app.exec_()) 