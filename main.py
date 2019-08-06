import sys
import mainwindow
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.Qt import QTimer, QTime
from PyQt5.QtGui import QPixmap, QImage, qRgb
from PyQt5.QtWidgets import QFileDialog
import cv2
from tf_ssd import tf_classifier
from df_yolo import df_classifier

gray_color_table = [qRgb(i, i, i) for i in range(256)]
NUM_CLASSES = 1

def convert_ndarray_to_qimg(arr):
    if arr is None:
        return QImage()
    qim = None
    if arr.dtype is not np.uint8:
        arr = arr.astype(np.uint8)
    if arr.dtype == np.uint8:
        if len(arr.shape) == 2:
            qim = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
        elif len(arr.shape) == 3:
            if arr.shape[2] == 3:
                qim = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
    return qim.copy()

class guiApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):

	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.video_path = '2019-05-06-10-45-16-tv_30_tnf.avi'
		self.video_capturer = cv2.VideoCapture(self.video_path)
		self.video_status = 'Video is not playing'
		self.width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.out = cv2.VideoWriter(self.video_path[:-4] + '_label.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (self.width, self.height))
		self.video_frame = None
		self.video_src = 'camera'
		self.proc_image_width =960
		self.proc_image_height = 540
		self.proc_qimage = QImage()
		self.proc_qpixmap = QPixmap()
		self.tmr = QTimer(self)
		self.tmr.setInterval(40)
		self.tmr.timeout.connect(self.timeout_slot)
		self.tmr.start()
		self.time = QTime()
		self.time.start()
		self.fps_period = None
		self.last_timestamp = self.time.elapsed()
		self.enable_detector = False
		self.enable_detector_yolo = False
		self.enable_playing = False
		self.enable_record = False
		self.cnn_model_path = 'output_models/frozen_inference_graph.pb'
		self.cfg_yolo_path = 'output_modelsgoit/yolov2.cfg'
		self.weights_yolo_path = 'output_models/yolov2_last.weights'
		self.ssd = None
		self.yolo = None

		self.camera_id = 0
		self.camera_capturer = cv2.VideoCapture(self.camera_id)
		self.camera_status = 'Stream is non active'
		self.camera_frame = None

		self.setWindowTitle('SKL CNN Classifier')
		self.checkBox_enable_detection.toggled.connect(self.checkbox_enable_detection_toggled)
		self.checkBox_enable_playing.toggled.connect(self.checkbox_enable_playing_toggled)
		self.checkBox_enable_camera.toggled.connect(self.checkbox_enable_camera_toggled)
		self.checkBox_enable_detection_yolo.toggled.connect(self.checkBox_enable_detection_yolo_toggled)
		self.checkBox_enable_record.toggled.connect(self.checkBox_enable_record_toggled)
		self.lineEdit_path_video.editingFinished.connect(self.lineedit_path_video_editing_finished)
		self.lineEdit_path_graph.editingFinished.connect(self.lineedit_path_graph_editing_finished)
		self.lineEdit_path_cfg_yolo.editingFinished.connect(self.lineedit_path_cfg_yolo_editing_finished)
		self.lineEdit_path_weight_yolo.editingFinished.connect(self.lineedit_path_weight_yolo_editing_finished)
		self.pushButton_load_video.clicked.connect(self.pushbutton_load_video_clicked)
		self.pushButton_load_graph.clicked.connect(self.pushbutton_load_graph_clicked)
		self.pushButton_load_cdg_yolo.clicked.connect(self.pushButton_load_cdg_yolo_clicked)
		self.pushButton_load_weights_yolo.clicked.connect(self.pushButton_load_weights_yolo_clicked)

		self.lineEdit_path_video.setText(self.video_path)
		self.lineEdit_path_graph.setText(self.cnn_model_path)
		self.lineEdit_path_cfg_yolo.setText(self.cfg_yolo_path)
		self.lineEdit_path_weight_yolo.setText(self.weights_yolo_path)
		self.checkBox_enable_camera.setChecked(True)

		pass

	def timeout_slot(self):
		if self.video_src == 'camera':
			self.refresh_camera_frame()
			self.proc_frame = self.camera_frame
		if self.video_src == 'video':
			self.refresh_video_frame()
			self.proc_frame = self.video_frame
		if self.proc_frame is not None:

			if self.enable_detector:
				cropimg = self.proc_frame[0:self.proc_frame.shape[0], int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2):
																	  int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), :]
				ssd_image = self.ssd.forwardpass(cropimg)
				self.proc_frame[0:self.proc_frame.shape[0],
				int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2):int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), :] = ssd_image
				cv2.rectangle(self.proc_frame, (int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2), 0),
							  (int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), self.proc_frame.shape[0]),
							  (255, 0, 0), 2)
			elif self.enable_detector_yolo:

				cropimg = self.proc_frame[0:self.proc_frame.shape[0],
						  int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2):int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), :]
				yolo_image = self.yolo.detect_by_yolo(cropimg)
				self.proc_frame[0:self.proc_frame.shape[0],
				int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2):int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), :] = yolo_image
				cv2.rectangle(self.proc_frame, (int((self.proc_frame.shape[1] - self.proc_frame.shape[0]) / 2), 0),
							  (int((self.proc_frame.shape[1] + self.proc_frame.shape[0]) / 2), self.proc_frame.shape[0]), (255, 0, 0), 2)

			if self.enable_record:
				self.out.write(cv2.cvtColor(self.proc_frame, cv2.COLOR_RGB2BGR))
			self.proc_frame = cv2.resize(self.proc_frame, (self.proc_image_width, self.proc_image_height),
										 interpolation=cv2.INTER_CUBIC)
			self.proc_qimage = convert_ndarray_to_qimg(self.proc_frame)
			self.proc_qpixmap = QPixmap.fromImage(self.proc_qimage)
			if self.proc_qpixmap is not None:
				self.label_image.setPixmap(self.proc_qpixmap)

		cur_time = self.time.elapsed()
		self.fps_period = cur_time - self.last_timestamp
		self.last_timestamp = cur_time
		self.label_processing_time.setText(str(self.fps_period))
		self.label_fps.setText(str(int(1000.0/self.fps_period)))
		pass

	def refresh_camera_frame(self):
		ret, self.camera_frame = self.camera_capturer.read()
		if self.camera_frame is not None:
			self.camera_frame = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
			self.camera_status = 'Capturing in progress'
		else:
			self.camera_status = 'Error'
		pass

	def refresh_camera_stream(self):
		self.camera_capturer.release()
		self.camera_capturer = cv2.VideoCapture(self.camera_id)
		pass

	def refresh_video_frame(self):
		ret, self.video_frame = self.video_capturer.read()
		if self.video_frame is not None:
			self.video_frame = cv2.cvtColor(self.video_frame, cv2.COLOR_BGR2RGB)
			self.video_status = 'Playing in progress'
		else:
			self.video_status = 'Error'
		self.label_video_status.setText(self.video_status)
		pass

	def refresh_video_stream(self):
		self.video_capturer.release()
		self.video_capturer = cv2.VideoCapture(self.video_path)

	def lineedit_path_video_editing_finished(self):
		self.video_path = self.lineEdit_path_video.text()
		self.width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.out = cv2.VideoWriter(self.video_path[:-4] + '_label.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0,
								   (self.width, self.height))
		self.refresh_video_stream()
		pass

	def lineedit_path_graph_editing_finished(self):
		self.cnn_model_path = self.lineEdit_path_graph.text()
		pass

	def lineedit_path_cfg_yolo_editing_finished(self):
		self.cfg_yolo_path = self.lineEdit_path_cfg_yolo.text()
		pass

	def lineedit_path_weight_yolo_editing_finished(self):
		self.weights_yolo_path = self.lineEdit_path_weight_yolo.text()
		pass

	def checkbox_enable_detection_toggled(self, checked):
		if self.ssd is not tf_classifier:
			self.ssd = tf_classifier()
			self.ssd.select_path_to_model(self.cnn_model_path)
		self.enable_detector = checked
		pass

	def checkBox_enable_detection_yolo_toggled(self, checked):
		if self.yolo is not df_classifier:
			self.yolo = df_classifier()
			self.yolo.select_path_to_model(self.cfg_yolo_path, self.weights_yolo_path)
		self.enable_detector_yolo = checked
		pass

	def checkbox_enable_camera_toggled(self, checked):
		if checked:
			self.video_src = 'camera'
		else:
			self.video_src = 'none'
		self.switch_video_src(self.video_src)
		pass

	def checkbox_enable_playing_toggled(self, checked):
		if checked:
			self.video_src = 'video'
		else:
			self.video_src = 'none'
		self.switch_video_src(self.video_src)
		pass

	def checkBox_enable_record_toggled(self, checked):
		self.enable_record = checked
		pass

	def switch_video_src(self, new_src):
		if new_src == 'camera':
			self.checkBox_enable_playing.setChecked(False)
			self.checkBox_enable_camera.setChecked(True)
		elif new_src == 'video':
			self.checkBox_enable_playing.setChecked(True)
			self.checkBox_enable_camera.setChecked(False)

		else:
			self.checkBox_enable_playing.setChecked(False)
			self.checkBox_enable_camera.setChecked(False)
		pass

	def pushbutton_load_video_clicked(self):
		fname = QFileDialog.getOpenFileName(self, 'Select video', '')[0]
		self.lineEdit_path_video.setText(fname)
		self.video_path = self.lineEdit_path_video.text()
		self.refresh_video_stream()
		pass

	def pushbutton_load_graph_clicked(self):
		fname = QFileDialog.getOpenFileName(self, 'Select graph SSD', '')[0]
		self.lineEdit_path_graph.setText(fname)
		pass

	def pushButton_load_cdg_yolo_clicked(self):
		fname = QFileDialog.getOpenFileName(self, 'Select cfg Yolo', '')[0]
		self.lineEdit_path_cfg_yolo.setText(fname)
		pass

	def pushButton_load_weights_yolo_clicked(self):
		fname = QFileDialog.getOpenFileName(self, 'Select weights Yolo', '')[0]
		self.lineEdit_path_weight_yolo.setText(fname)
		pass

def main():
	app = QtWidgets.QApplication(sys.argv)
	window = guiApp()
	window.show()
	sys.exit(app.exec_())
	pass

if __name__ == '__main__':
	main()