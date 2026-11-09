深度学习目标检测框架使用YOLOv8来训练——_变压器红外测温过热点检测图像数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d36a1b5d2f8f41c88f53db8d08b9ea14.png)
变压器红外测温过热点检测图像数据集，总共包含600张图片，其中200多张为包含过热点的图像，标注为voc格式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6a7884e416ea44c788ac6acea97635d3.png)
使用YOLOv8来训练变压器红外测温过热点检测图像数据集。以下是详细的步骤：

1. **环境准备**：
   - 安装必要的库。
   - 下载并组织数据集。

2. **数据预处理**：
   - 将VOC XML格式的标注转换为YOLOv8所需的格式。
   - 确认数据集的划分和标签格式。

3. **模型定义与训练**：
   - 使用YOLOv8进行目标检测。

4. **评估与可视化**：
   - 评估模型性能。
   - 可视化结果。
在这里插入图片描述

### 环境准备

首先，我们需要安装必要的库。您可以使用以下命令来设置环境：

```bash
pip install ultralytics lxml opencv-python-headless
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/76c4954394dc4ff584503185926fd8af.png)

### 数据预处理![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3150c76d2a24c868d8eb1206b2b78ca.png)


我们需要将VOC XML格式的标注文件转换为YOLOv8所需的TXT格式。以下是数据转换脚本 `convert_voc_to_yolo.py`:

```python
[<title="Convert VOC XML Annotations to YOLO TXT Format">]
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Define paths
base_path = Path('datasets/transformer_ir')
annotations_dir = base_path / 'Annotations'
images_dir = base_path / 'JPEGImages'
output_labels_dir = base_path / 'labels'

# Create output directory if it doesn't exist
os.makedirs(output_labels_dir, exist_ok=True)

# Class names and their corresponding IDs
class_names = {
    'hotspot': 0
}

def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    label_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert bounding box to YOLO format (center_x, center_y, width, height)
        center_x = (xmin + xmax) / 2.0 / image_width
        center_y = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        class_id = class_names[class_name]
        label_line = f"{class_id} {center_x} {center_y} {width} {height}\n"
        label_lines.append(label_line)

    return label_lines

for annotation_file in annotations_dir.glob('*.xml'):
    image_name = annotation_file.stem + '.jpg'  # Assuming images are in JPEG format
    label_file = output_labels_dir / (annotation_file.stem + '.txt')
    
    label_lines = convert_annotation(annotation_file)
    
    with open(label_file, 'w') as f:
        f.writelines(label_lines)

print("Conversion completed.")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0d3e81ff6b074ab79849f2b2b3ce687d.png)

### 创建YAML配置文件

确保您有一个正确的YAML配置文件 `transformer_ir.yaml` 来描述数据集。以下是示例配置文件：

```yaml
[<title="YOLOv8 Configuration File for Transformer IR Hotspot Detection">]
train: ../datasets/transformer_ir/train/images
val: ../datasets/transformer_ir/val/images

nc: 1
names: ['hotspot']
```

将上述内容保存为 `datasets/transformer_ir/transformer_ir.yaml`。

### 划分数据集

假设您的数据集已经分为训练集、验证集和测试集，并且每个集合都有对应的图像和标签文件。如果还没有划分数据集，可以使用以下脚本来随机划分数据集：

```python
[<title="Split Dataset into Train, Val, Test Sets">]
import os
import random
from pathlib import Path

# Define paths
base_path = Path('datasets/transformer_ir')
images_dir = base_path / 'JPEGImages'
annotations_dir = base_path / 'Annotations'
train_images_dir = base_path / 'train/images'
train_labels_dir = base_path / 'train/labels'
val_images_dir = base_path / 'val/images'
val_labels_dir = base_path / 'val/labels'
test_images_dir = base_path / 'test/images'
test_labels_dir = base_path / 'test/labels'

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# List all image files
image_files = list(images_dir.glob('*.jpg'))  # Adjust extension if necessary

# Shuffle the image files
random.shuffle(image_files)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate split indices
num_images = len(image_files)
train_split = int(num_images * train_ratio)
val_split = int(num_images * (train_ratio + val_ratio))

# Split images and labels
train_images = image_files[:train_split]
val_images = image_files[train_split:val_split]
test_images = image_files[val_split:]

def copy_files(source_images, source_labels, dest_images_dir, dest_labels_dir):
    for img_file in source_images:
        label_file = annotations_dir / (img_file.stem + '.txt')
        if label_file.exists():
            os.symlink(img_file, dest_images_dir / img_file.name)
            os.symlink(label_file, dest_labels_dir / label_file.name)

copy_files(train_images, train_labels_dir, train_images_dir, train_labels_dir)
copy_files(val_images, val_labels_dir, val_images_dir, val_labels_dir)
copy_files(test_images, test_labels_dir, test_images_dir, test_labels_dir)

print("Dataset splitting completed.")
```

### 模型定义与训练

我们将使用YOLOv8进行目标检测。以下是训练脚本 `train_detection.py`:

```python
[<title="Training Script for Transformer IR Hotspot Detection using YOLOv8">]
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='../datasets/transformer_ir/transformer_ir.yaml',
    epochs=50,
    imgsz=513,
    batch=16,
    project='../runs/train',
    name='transformer_ir_hotspot_detection'
)

# Evaluate the model
metrics = model.val()
results = model.export(format='onnx')  # export the trained model to ONNX format
```

### 评估与可视化

使用YOLOv8自带的评估脚本来评估目标检测模型。

```python
[<title="Evaluation Script for Transformer IR Hotspot Detection using YOLOv8">]
from ultralytics import YOLO

# Load the best model
best_model = YOLO('../runs/train/transformer_ir_hotspot_detection/weights/best.pt')

# Evaluate the model on the validation dataset
metrics = best_model.val(data='../datasets/transformer_ir/transformer_ir.yaml', conf=0.5, iou=0.45)
print(metrics)
```

### 用户界面

我们将使用 PyQt5 创建一个简单的 GUI 来加载和运行模型进行实时预测。以下是用户界面脚本 `ui.py`:

```python
[<title="PyQt5 Main Window for Transformer IR Hotspot Detection">]
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

# Load model
detection_model = YOLO('../runs/train/transformer_ir_hotspot_detection/weights/best.pt')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("变压器红外测温过热点检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_image_button = QPushButton("加载图像", self)
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        self.start_prediction_button = QPushButton("开始预测", self)
        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.start_prediction_button)

        self.stop_prediction_button = QPushButton("停止预测", self)
        self.stop_prediction_button.clicked.connect(self.stop_prediction)
        self.layout.addWidget(self.stop_prediction_button)

        self.central_widget.setLayout(self.layout)

        self.image_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.display_image(file_name)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def start_prediction(self):
        if self.image_path is not None and not self.timer.isActive():
            self.timer.start(30)  # Update frame every 30 ms

    def stop_prediction(self):
        if self.timer.isActive():
            self.timer.stop()
            self.image_label.clear()

    def update_frame(self):
        original_image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Detection
        results = detection_model.predict(image_rgb, size=513, conf=0.5, iou=0.45)[0]

        for box in results.boxes.cpu().numpy():
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Map class ID to name
            class_names = ['过热点']
            class_name = class_names[cls]

            # Draw bounding box
            cv2.rectangle(image_rgb, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

            # Put text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_rgb, f'{class_name} ({conf:.2f})', (r[0], r[1] - 10), font, 0.9, (0, 255, 0), 2)

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

请确保将路径替换为您实际的路径。

### 使用说明

1. **配置路径**：
   - 确保 `datasets/transformer_ir` 目录结构正确，并且包含 `train`、`val` 和 `test` 子目录。
   - 确保 `runs/train/transformer_ir_hotspot_detection/weights/best.pt` 是训练好的 YOLOv8 模型权重路径。

2. **运行脚本**：
   - 在终端中运行 `convert_voc_to_yolo.py` 脚本来将VOC XML格式的标注文件转换为YOLOv8所需的TXT格式。
   - 如果尚未划分数据集，在终端中运行 `split_dataset.py` 脚本来划分数据集。
   - 在终端中运行 `check_dataset.py` 脚本来检查数据集的有效性。
   - 在终端中运行 `train_detection.py` 脚本来训练目标检测模型。
   - 在终端中运行 `evaluate_detection.py` 来评估目标检测模型性能。
   - 在终端中运行 `ui.py` 来启动 GUI 应用程序。

3. **注意事项**：
   - 确保所有必要的工具箱已安装，特别是 `ultralytics` 和 `PyQt5`。
   - 根据需要调整参数，如 `epochs` 和 `batch_size`。

### 示例

假设您的数据文件夹结构如下：
```
datasets/
└── transformer_ir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

并且每个数据集中包含相应的图像和标签文件。运行 `ui.py` 后，您可以点击按钮来加载图像并进行变压器红外测温过热点检测。

