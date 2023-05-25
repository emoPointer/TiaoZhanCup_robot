# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import math
import os
import platform
import sys
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtCore import QTimer, pyqtSignal, QRectF, QUrl, QRect, QDateTime, QDate
from PyQt5.QtGui import QImage, QPixmap, QTransform, QPainter, QFont, QPalette, QBrush
from PyQt5.QtMultimedia import QMediaPlayer, QVideoFrame, QAbstractVideoSurface, QAbstractVideoBuffer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QHBoxLayout, QGridLayout, QLabel, \
    QSpacerItem, QSizePolicy, QVBoxLayout, QLineEdit, QTextEdit, QFrame, QPushButton, QProgressBar
import sys
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
global mains
global pointList
class VideoSurface(QAbstractVideoSurface):
    showImageSignal = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super(VideoSurface, self).__init__(parent)

    def supportedPixelFormats(self, type):
        return [QVideoFrame.Format_RGB32, QVideoFrame.Format_RGB32]

    def present(self, frame):
        """获取帧并发送信号"""
        if frame.isValid():
            cloneFrame = QVideoFrame(frame)
            cloneFrame.map(QAbstractVideoBuffer.ReadOnly)
            img = QImage(cloneFrame.bits(), cloneFrame.width(), cloneFrame.height(),
                         QVideoFrame.imageFormatFromPixelFormat(cloneFrame.pixelFormat()))
            cloneFrame.unmap()
            self.showImageSignal.emit(img)
            return True
        return False


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        # 当前帧QImage
        self.__image = None
        # 旋转的度数
        self.__degree = 0
        # 缩放后的宽度
        self.__scaleWidth = None
        # 缩放后的高度
        self.__scaleHeight = None
        # 缩放后的位置
        self.__posX = 0
        self.__posY = 0
        # 垂直翻转标志位
        self.__verticalFlipFlag = False
        # 水平翻转标志位
        self.__horizontalFlipFlag = False

    def resizeEvent(self, event):
        self.calculateRectAfterResize()
        if self.__degree in (-90, 90, -270, 270):
            # 交换两个变量
            self.__scaleWidth, self.__scaleHeight = self.__scaleHeight, self.__scaleWidth
        self.update()
        super(VideoWidget, self).resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self.__image:
            rect = QRectF(self.__posX, self.__posY, self.__image.width(), self.__image.height())
            painter.drawImage(rect, self.__image)
        else:
            # 这里可以做视频加载动画
            pass
        painter.end()

    def calculateRectAfterResize(self):
        """调整大小后计算宽高和位置"""
        if self.__image:
            # 当前显示窗口宽高比
            widgetRatio = self.width() / self.height()
            srcRatio = self.__image.width() / self.__image.height()
            if widgetRatio >= srcRatio:
                self.__scaleWidth = srcRatio * self.height()
                self.__scaleHeight = self.height()
                self.__posX = (self.width() - self.__scaleWidth) / 2
                self.__posY = 0
            else:
                self.__scaleWidth = self.width()
                self.__scaleHeight = self.width() / srcRatio
                self.__posX = 0
                self.__posY = (self.height() - self.__scaleHeight) / 2

    def calculateRectAfterTransform(self):
        """setTransform后计算宽高和位置"""
        if self.__image:
            # 当前显示窗口宽高比
            widgetRatio = self.width() / self.height()
            srcRatio = self.__image.height() / self.__image.width()
            if widgetRatio >= srcRatio:
                self.__scaleWidth = self.height()
                self.__scaleHeight = self.height() * srcRatio
                self.__posX = (self.width() - self.__scaleHeight) / 2
                self.__posY = 0
            else:
                self.__scaleWidth = self.width() / srcRatio
                self.__scaleHeight = self.width()
                self.__posX = 0
                self.__posY = (self.height() - self.__scaleWidth) / 2

    def showImageSlot(self, img):
        """槽函数，接收图片，进行缩放和变换"""
        if self.__image is None:
            # 如果是初次接收图片，需要根据当前窗口大小计算宽高和位置
            self.__image = img
            self.calculateRectAfterResize()
            if self.__degree in (-90, 90, -270, 270):
                # 交换两个变量
                self.__scaleWidth, self.__scaleHeight = self.__scaleHeight, self.__scaleWidth
        self.__image = img
        # 缩放
        self.doScale()
        # 旋转
        self.doRotate()
        # 翻转
        self.doFlip()
        self.update()

    def doScale(self):
        """进行缩放操作"""
        if self.__scaleWidth and self.__scaleHeight:
            self.__image = self.__image.scaled(self.__scaleWidth, self.__scaleHeight)

    def doRotate(self):
        """进行旋转操作"""
        if self.__degree != 0:
            matrix = QTransform()
            matrix.rotate(self.__degree)
            self.__image = self.__image.transformed(matrix, Qt.FastTransformation)

    def doFlip(self):
        """进行翻转操作"""
        if self.__verticalFlipFlag:
            # 垂直翻转
            self.__image = self.__image.mirrored()
        elif self.__horizontalFlipFlag:
            # 水平翻转
            self.__image = self.__image.mirrored(True, False)

    def setFlip(self, direction):
        if direction == horizontalFlip:
            # 如果是水平垂直翻转，对应的标志位取反
            self.__horizontalFlipFlag = not self.__horizontalFlipFlag
        elif direction == verticalFlip:
            self.__verticalFlipFlag = not self.__verticalFlipFlag
        self.update()

    def setRotate(self, direction):
        """设置旋转角度并计算一些值"""
        if direction == rotateToLeft:
            # 如果是左右旋转，需要修改度数
            self.__degree -= 90
        elif direction == rotateToRight:
            self.__degree += 90
        # 如果旋转度数达到了+-360，归零
        if self.__degree == 360 or self.__degree == -360:
            self.__degree = 0
        self.calculateRectAfterTransform()
        if self.__degree in (0, -180, 180):
            self.__scaleWidth, self.__scaleHeight = self.__scaleHeight, self.__scaleWidth
        self.update()


horizontalFlip = 0
verticalFlip = 1
rotateToLeft = 0
rotateToRight = 1


class Ui_Form(object):

    def setupUi(self, Form):
        Form.setObjectName("Class")
        Form.resize(1280, 600)
        self.timer_camera = QTimer()
        # self.showVideo=QtWidgets.QWidget
        # self.showDetail=QtWidgets.QWidget
        # self.showVideo.setGeometry(0,0,768,600)
        # self.showDetail.setGeometry(768,0,256,600)
        # self.pushButton = QtWidgets.QPushButton(Form)
        # self.pushButton.setGeometry(QtCore.QRect(130, 200, 75, 23))
        # self.pushButton.setObjectName("pushButton")
        # self.label = QtWidgets.QLabel(Form)
        # self.label.setGeometry(QtCore.QRect(50, 90, 291, 61))
        # self.label.setObjectName("label")
        # self.retranslateUi(Form)
        # QtCore.QMetaObject.connectSlotsByName(Form)
        # self.pushButton.clicked.connect(self.slot_btn_clicked)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "PushButton"))
        self.label.setText(_translate("Form", "TextLabel"))

    def slot_btn_clicked(self):
        self.label.setText("hello world!!!")


class mainwindow:
    def __init__(self):
        self.sumOfRubbishs = [0, 0, 0, 0]
        self.sumOfRubbish = 0

    def getNum(self,label,isFull=False):
        label,percent=self.cuts(label)
        if label=="battery":
            self.sumOfRubbishs[1]=percent
            print(self.sumOfRubbishs)
        if label=="can" or label=="bottle":
            self.sumOfRubbishs[0]=percent
        if label=="red_carrot" or label=="white_carrot" or label=="potato":
            self.sumOfRubbishs[2]=percent
        Time = QDateTime.currentDateTime()  # 获取现在的时间
        Timeplay = Time.toString('hh:mm:ss')  # 设置显示时间的格式
        if label== "can" or label== "bottle":
            self.textStream.insertText("可回收垃圾×1\t" + Timeplay + "\n")
        if label=="battery":
            self.textStream.insertText("有害垃圾×1\t" + Timeplay + "\n")
        if label=="red_carrot" or label=="white_carrot" or label=="potato":
            self.textStream.insertText("厨余垃圾×1\t" + Timeplay + "\n")
    def showLabels(self):
        print(self.sumOfRubbishs)
        style = QFont()
        style.setPointSize(16)
        self.showResName.setText("可回收垃圾：" + str(self.sumOfRubbishs[0]))
        self.showResName.setFont(style)
        self.showUnResName.setText("有害垃圾：  " + str(self.sumOfRubbishs[1]))
        self.showUnResName.setFont(style)
        self.showFoodName.setText("厨余垃圾：  " + str(self.sumOfRubbishs[2]))
        self.showFoodName.setFont(style)
        self.showOtherName.setText("其它垃圾：  " + str(self.sumOfRubbishs[3]))
        self.showOtherName.setFont(style)
        self.sumOfRubbish=self.sumOfRubbishs[0]+self.sumOfRubbishs[1]+self.sumOfRubbishs[2]+self.sumOfRubbishs[3]
        self.showRubbishSum.setText("垃圾总数：  " + str(self.sumOfRubbish) + "(" + self.stringIsFull + ")")
        self.showRubbishSum.setFont(style)
        self.showProcesser.setValue(int(self.sumOfRubbish / 2))
    def start(self):
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        MainWindow.setWindowTitle("垃圾分类")
        ui = Ui_Form()
        ui.setupUi(MainWindow)
        player = QMediaPlayer()
        videoSurface = VideoSurface()
        player.setVideoOutput(videoSurface)
        showVideo = VideoWidget()

        showVideo.resize(500, 350)

        showVideo.setStyleSheet('''
                    QPushButton {
                        color: blue;
                        background-color: rgba(0,0,0,0.5)
                    }
                ''')
        MainWindow.setCentralWidget(showVideo)
        mainLayout = QGridLayout()
        mainLayout.addWidget(showVideo, 1, 1)

        videoSurface.showImageSignal.connect(showVideo.showImageSlot)
        player.setMedia(QMediaContent(QUrl.fromLocalFile(r'C:\Users\dell\Desktop\video1.avi')))
        player.play()

        showTimeDetailBack = QGridLayout()
        showTimeDetail = QWidget()
        showBackground = QGridLayout()
        showSecondBackground = QGridLayout()
        showDetailBack = QHBoxLayout()
        self.showDetail = QTextEdit()


        self.showProcesser=QProgressBar()
        showEm = QLabel()
        style = QFont()

        style.setPointSize(16)
        # 定义信息
        full = False
        percent = 30
        if (full):
            self.stringIsFull = "满载"
        else:
            self.stringIsFull = "未满载"
        self.showResName = QLabel()
        self.showUnResName = QLabel()
        self.showFoodName = QLabel()
        self.showOtherName = QLabel()
        self.showResName.setText("可回收垃圾：" + str(self.sumOfRubbishs[0]))
        self.showResName.setFont(style)
        self.showUnResName.setText("有害垃圾：  " + str(self.sumOfRubbishs[1]))
        self.showUnResName.setFont(style)
        self.showFoodName.setText("厨余垃圾：  " + str(self.sumOfRubbishs[2]))
        self.showFoodName.setFont(style)
        self.showOtherName.setText("其它垃圾：  " + str(self.sumOfRubbishs[3]))
        self.showOtherName.setFont(style)
        showSecondBackground.addWidget(self.showResName, 1, 1, 1, 1)
        showSecondBackground.addWidget(self.showUnResName, 1, 2, 1, 1)
        showSecondBackground.addWidget(self.showFoodName, 2, 1, 1, 1)
        showSecondBackground.addWidget(self.showOtherName, 2, 2, 1, 1)

        self.showRubbishSum = QLabel()



        self.showRubbishSum.setText("垃圾总数：  " + str(self.sumOfRubbish) + "(" + self.stringIsFull + ")")
        self.showRubbishSum.setFont(style)
        fonta = QFont("微软雅黑 Light", 30)
        fontb = QFont("微软雅黑", 10)
        self.timeLabel = QLabel()
        self.timeLabel.setFont(fonta)
        self.statusShowTime()
        dataLabel = QLabel()
        dataLabel.setFont(fontb)
        dataLabel.setText(QDate.currentDate().toString("yyyy年MM月dd日") + " " + QDate.currentDate().toString("dddd"))
        dataLabel.setStyleSheet("color:#2F4F4F")
        showTimeDetailBack.addWidget(self.timeLabel, 1, 1, 1, 1)
        showTimeDetailBack.addWidget(dataLabel, 2, 1, 1, 1)
        showIMG = QLabel()
        pathIMG = QImage()
        pathIMG.load(r'C:\Users\dell\Desktop\recycle.jpg')
        resIMG = pathIMG.scaled(100, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        rec = QPixmap(QPixmap.fromImage(resIMG))
        showIMG.setPixmap(rec)
        showTimeDetailBack.addWidget(showIMG, 1, 2, 2, 1)
        showTimeDetail.setLayout(showTimeDetailBack)
        self.showDetail.setFixedSize(480, 200)
        self.showDetail.setFocusPolicy(QtCore.Qt.NoFocus)
        self.showDetail.setStyleSheet("QTextEdit{border:2px solid cornflowerblue;border-radius:10px;border-top-left-radius:10px"
                                 ";border-top-right-radius:10px;border-bottom-left-radius:10px;border-bottom-right"
                                 "-radius:10px;}")
        self.showDetail.scroll(1,1)
        # 定义信息区

        stringTable = "垃圾桶实时情况数据：\n"
        stringFull = "满载检测：" + self.stringIsFull + "\n"
        stringStream = "剩余流量：" + str(percent) + "%\n"
        stringItem = "------物品栏------\n"
        textString = [stringTable, stringFull, stringStream, stringItem]
        self.textStream = self.showDetail.textCursor()
        Time = QDateTime.currentDateTime()  # 获取现在的时间
        for i in range(4):
            self.textStream.insertText(textString[i])
        self.showProcesser.setFixedSize(480,25)
        self.showProcesser.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; background-color: #FFFFFF; text-align: center;}QProgressBar::chunk {background:QLinearGradient(x1:0,y1:0,x2:2,y2:0,stop:0 #666699,stop:1  #DB7093); }")
        font = QFont()
        font.setBold(True)
        font.setWeight(30)
        self.showProcesser.setFormat('垃圾占比:%p%'.format(self.showProcesser.value()-self.showProcesser.minimum()))
        self.showProcesser.setFont(font)

        self.showProcesser.setValue(int(self.sumOfRubbish/2))
        self.showDetail.setFont(fontb)
        showDetailBack.addWidget(self.showDetail)
        showBackground.addWidget(showTimeDetail, 1, 1)
        showBackground.addWidget(self.showRubbishSum, 2, 1)
        showBackground.addLayout(showSecondBackground,3, 1)
        showBackground.addWidget(self.showProcesser,4,1)
        showBackground.addLayout(showDetailBack, 5, 1)
        showBackground.setVerticalSpacing(20)
        showBackground.setAlignment(QtCore.Qt.AlignTop)
        VSpacer = QSpacerItem(20, 250, QSizePolicy.Fixed, QSizePolicy.Minimum)
        showBackground.addItem(VSpacer)
        mainLayout.addLayout(showBackground, 1, 2)

        big = QWidget()
        big.setLayout(mainLayout)
        big.setStyleSheet("QWidget{background:white}")

        MainWindow.setCentralWidget(big)
        MainWindow.setFixedSize(1280, 600)

        MainWindow.show()
        opt = parse_opt()
        main(opt)
        sys.exit(app.exec_())

    def statusShowTime(self):
        self.Timer = QTimer()  # 自定义QTimer类
        self.Timer.timeout.connect(self.updateTime)  # 与updateTime函数连接
        self.Timer.start(100)  # 每0.1s运行一次

    def updateTime(self):
        time = QDateTime.currentDateTime()  # 获取现在的时间
        timeplay = time.toString('hh:mm:ss')  # 设置显示时间的格式
        self.showLabels()
        self.timeLabel.setText(timeplay)  # 设置timeLabel控件显示的内容
    def cuts(self,label):
        paralist=label.split(" ")
        cnt=0
        name=""
        percent=0.0
        for i in paralist:
            if cnt==0:
                cnt=1
                name=i
            else:
                percent=int(i)
        return name,percent
class RubbishClass:
    name=""
    percent=0.0
    def __init__(self,label):
        self.name,self.percent=self.cuts(label)
        print("NAME:"+self.name+" "+"PERCENT:"+str(self.percent))
        print("TRANSFER:"+str(self.transfer()))
    def cuts(self,label):
        paralist=label.split(" ")
        cnt=0
        name=""
        percent=0.0
        for i in paralist:
            if cnt==0:
                cnt=1
                name=i
            else:
                percent=float(i)
        return name,percent
    def transfer(self):
        if self.name=="battery":
            return 0x00
        if self.name=="bottle":
            return 0x01
        if self.name=="potato":
            return 0x02
        if self.name=="carrot":
            return 0x03
        return 0x04
class shit:
    def __init__(self):
        self.pl=[]
    def set(self,pi):
        self.pl=pi
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    print(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                count=1
                for *xyxy, conf, cls in reversed(det):
                    xywh1 = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    pointList.pl=xywh1
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {count:.0f}')
                        count+=1

                        #RubbishClass(label)
                        #mains.getNum(label,True)
                        #mains.stringIsFull="已满载"

                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                if not pointList.pl==[]:
                    w=640
                    h=480
                    x_, y_, w_, h_ = pointList.pl[0], pointList.pl[1], pointList.pl[2], pointList.pl[3]
                    x1 = w * x_ - 0.5 * w * w_
                    x2 = w * x_ + 0.5 * w * w_
                    y1 = h * y_ - 0.5 * h * h_
                    y2 = h * y_ + 0.5 * h * h_
                    print(x1,y1,x2,y2)

                    image=im0[int(y1):int(y1+math.fabs(y1-y2)),int(x1):int(x1+math.fabs(x1-x2))]
                    cv2.imwrite('1.png', image)
                    cv2.imshow("114514",image)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best_114514.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'video.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))



if __name__ == "__main__":
    pointList=shit()
    opt = parse_opt()
    main(opt)



