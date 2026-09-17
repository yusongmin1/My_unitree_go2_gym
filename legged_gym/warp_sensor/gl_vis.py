import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class UpdateSignals(QObject):
    update_points = pyqtSignal(object)
    update_mesh = pyqtSignal(object)

class ObjectInfo:
    def __init__(self, name, obj_type):
        self.name = name
        self.type = obj_type  # 'point_cloud' or 'mesh'
        self.visible = True
        self.color = [1.0, 0.0, 0.0] if obj_type == 'point_cloud' else [0.0, 1.0, 0.0]
        self.size = 3.0 if obj_type == 'point_cloud' else 1.0
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.points = []
        self.meshes = []
        self.objects_info = {}  # Store information of all objects
        self.current_point_cloud_id = 0
        self.current_mesh_id = 0
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.scale = 1.0
        self.lastPos = QPoint()

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(self.xRot, 1.0, 0.0, 0.0)
        glRotatef(self.yRot, 0.0, 1.0, 0.0)
        glRotatef(self.zRot, 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)

        self.drawGrid()
        self.drawAxes()
        
        # Draw point clouds
        for i, points in enumerate(self.points):
            obj_info = self.objects_info.get(f'point_cloud_{i}')
            if obj_info and obj_info.visible:
                glPointSize(obj_info.size)
                glPushMatrix()
                glTranslatef(*obj_info.position)
                glRotatef(obj_info.rotation[0], 1, 0, 0)
                glRotatef(obj_info.rotation[1], 0, 1, 0)
                glRotatef(obj_info.rotation[2], 0, 0, 1)
                
                glBegin(GL_POINTS)
                glColor3f(*obj_info.color)
                for point in points:
                    glVertex3f(*point)
                glEnd()
                glPopMatrix()

        # Draw meshes
        for i, mesh in enumerate(self.meshes):
            obj_info = self.objects_info.get(f'mesh_{i}')
            if obj_info and obj_info.visible:
                glPushMatrix()
                glTranslatef(*obj_info.position)
                glRotatef(obj_info.rotation[0], 1, 0, 0)
                glRotatef(obj_info.rotation[1], 0, 1, 0)
                glRotatef(obj_info.rotation[2], 0, 0, 1)
                
                glBegin(GL_TRIANGLES)
                glColor3f(*obj_info.color)
                for triangle in mesh:
                    for vertex in triangle:
                        glVertex3f(*vertex)
                glEnd()
                glPopMatrix()

    def drawGrid(self):
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        
        for i in np.arange(-0.5, 0.51, 0.1):
            glVertex3f(i, -0.5, 0)
            glVertex3f(i, 0.5, 0)
            glVertex3f(-0.5, i, 0)
            glVertex3f(0.5, i, 0)
        
        glEnd()

    def drawAxes(self):
        glBegin(GL_LINES)
        
        # x-axis
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(10.0, 0.0, 0.0)
        
        # y-axis
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 10.0, 0.0)
        
        # z-axis
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 10.0)
        
        glEnd()

    def drawPoints(self):
        if self.points:
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            for point in self.points:
                glVertex3f(*point)
            glEnd()

    def drawMeshes(self):
        if self.meshes:
            glBegin(GL_TRIANGLES)
            glColor3f(0.0, 1.0, 0.0)
            for mesh in self.meshes:
                for vertex in mesh:
                    glVertex3f(*vertex)
            glEnd()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.yRot += dx
            self.xRot += dy
        elif event.buttons() & Qt.RightButton:
            self.zRot += dx

        self.update()
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        self.update()

class ControlPanel(QWidget):
    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Object list
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.on_item_selected)
        layout.addWidget(QLabel("Objects:"))
        layout.addWidget(self.object_list)

        # Control group
        self.controls_group = QGroupBox("Object Controls")
        controls_layout = QVBoxLayout()

        # Visibility control
        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.stateChanged.connect(self.on_visible_changed)
        controls_layout.addWidget(self.visible_checkbox)

        # Color selection
        color_btn = QPushButton("Change Color")
        color_btn.clicked.connect(self.on_color_click)
        controls_layout.addWidget(color_btn)

        # Point size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinner = QDoubleSpinBox()
        self.size_spinner.setRange(0.1, 20.0)
        self.size_spinner.setSingleStep(0.1)
        self.size_spinner.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_spinner)
        controls_layout.addLayout(size_layout)

        # Position control
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout()
        self.pos_spinners = []
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            pos_layout.addWidget(QLabel(label), i, 0)
            spinner = QDoubleSpinBox()
            spinner.setRange(-10, 10)
            spinner.setSingleStep(0.1)
            spinner.valueChanged.connect(self.on_position_changed)
            self.pos_spinners.append(spinner)
            pos_layout.addWidget(spinner, i, 1)
        pos_group.setLayout(pos_layout)
        controls_layout.addWidget(pos_group)

        # Rotation control
        rot_group = QGroupBox("Rotation")
        rot_layout = QGridLayout()
        self.rot_spinners = []
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            rot_layout.addWidget(QLabel(label), i, 0)
            spinner = QDoubleSpinBox()
            spinner.setRange(-360, 360)
            spinner.setSingleStep(5)
            spinner.valueChanged.connect(self.on_rotation_changed)
            self.rot_spinners.append(spinner)
            rot_layout.addWidget(spinner, i, 1)
        rot_group.setLayout(rot_layout)
        controls_layout.addWidget(rot_group)

        self.controls_group.setLayout(controls_layout)
        layout.addWidget(self.controls_group)
        
        # Add stretch
        layout.addStretch()
        
        self.setLayout(layout)
        self.current_item = None

    def add_object(self, obj_id, obj_info):
        item = QListWidgetItem(obj_info.name)
        item.setData(Qt.UserRole, obj_id)
        self.object_list.addItem(item)

    def on_item_selected(self, item):
        self.current_item = item
        obj_id = item.data(Qt.UserRole)
        obj_info = self.gl_widget.objects_info[obj_id]
        
        # Update control panel
        self.visible_checkbox.setChecked(obj_info.visible)
        self.size_spinner.setValue(obj_info.size)
        
        for i, value in enumerate(obj_info.position):
            self.pos_spinners[i].setValue(value)
        
        for i, value in enumerate(obj_info.rotation):
            self.rot_spinners[i].setValue(value)

    def on_visible_changed(self, state):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            self.gl_widget.objects_info[obj_id].visible = bool(state)
            self.gl_widget.update()

    def on_color_click(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            current_color = self.gl_widget.objects_info[obj_id].color
            color = QColorDialog.getColor(QColor.fromRgbF(*current_color))
            
            if color.isValid():
                self.gl_widget.objects_info[obj_id].color = [color.redF(), color.greenF(), color.blueF()]
                self.gl_widget.update()

    def on_size_changed(self, value):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            self.gl_widget.objects_info[obj_id].size = value
            self.gl_widget.update()

    def on_position_changed(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            position = [spinner.value() for spinner in self.pos_spinners]
            self.gl_widget.objects_info[obj_id].position = position
            self.gl_widget.update()

    def on_rotation_changed(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            rotation = [spinner.value() for spinner in self.rot_spinners]
            self.gl_widget.objects_info[obj_id].rotation = rotation
            self.gl_widget.update()

class Visualizer(QMainWindow):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.setWindowTitle("GL Visualizer")
        self.resize(800, 600)
        
        # Create main window widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create horizontal layout
        layout = QHBoxLayout(main_widget)
        
        # Create and add GL widget
        self.glWidget = GLWidget()
        
        # Create and add control panel
        self.control_panel = ControlPanel(self.glWidget)
        
        # Set fixed width for control panel
        self.control_panel.setFixedWidth(250)
        
        # Add control panel and GL widget to layout
        layout.addWidget(self.control_panel)
        layout.addWidget(self.glWidget, stretch=1)
        
        # Create signals
        self.signals = UpdateSignals()
        self.signals.update_points.connect(self._update_points)
        self.signals.update_mesh.connect(self._update_mesh)

    def _update_points(self, points):
        point_cloud_id = f'point_cloud_{self.glWidget.current_point_cloud_id}'
        self.glWidget.points.append(points)
        self.glWidget.objects_info[point_cloud_id] = ObjectInfo(
            f"Point Cloud {self.glWidget.current_point_cloud_id}", 
            'point_cloud'
        )
        self.control_panel.add_object(point_cloud_id, self.glWidget.objects_info[point_cloud_id])
        self.glWidget.current_point_cloud_id += 1
        self.glWidget.update()

    def _update_mesh(self, vertices):
        mesh_id = f'mesh_{self.glWidget.current_mesh_id}'
        self.glWidget.meshes.append(vertices)
        self.glWidget.objects_info[mesh_id] = ObjectInfo(
            f"Mesh {self.glWidget.current_mesh_id}", 
            'mesh'
        )
        self.control_panel.add_object(mesh_id, self.glWidget.objects_info[mesh_id])
        self.glWidget.current_mesh_id += 1
        self.glWidget.update()

    def add_points(self, points):
        if isinstance(points, np.ndarray):
            points = points.tolist()
        self.signals.update_points.emit(points)

    def add_mesh(self, vertices):
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        self.signals.update_mesh.emit(vertices)
    
    def add_trimesh(self, vertices, faces):
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        if isinstance(faces, np.ndarray):
            faces = faces.tolist()
        mesh = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] for face in faces]
        self.add_mesh(mesh)

class VisualizerWrapper:
    _instance = None
    _app = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if QApplication.instance() is None:
                cls._app = QApplication(sys.argv)
            cls._instance = Visualizer()
            cls._instance.show()
        return cls._instance

    @classmethod
    def run(cls):
        if cls._app is not None:
            cls._app.exec_()

def create_visualizer():
    """Create and return a visualizer instance"""
    return VisualizerWrapper.get_instance()

if __name__ == '__main__':
    # Create visualizer
    vis = create_visualizer()

    # Add some random point clouds
    points1 = np.random.rand(1000, 3) * 0.5 - 0.25
    vis.add_points(points1)

    points2 = np.random.rand(1000, 3) * 0.5 + 0.25
    vis.add_points(points2)

    # Add a simple triangle mesh
    triangle = [
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]
    ]
    vis.add_mesh(triangle)

    VisualizerWrapper.run()