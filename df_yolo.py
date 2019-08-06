from darkflow.net.build import TFNet
import numpy as np
import cv2

class df_classifier():

    def __init__(self):
        self.PATH_TO_CFG = "/media/aigul/Tom/Aigul/Narrow_field_tensorflow/YOLO/drone_detector/yolov2.cfg"
        self.PATH_TO_WEIGHTS = "/media/aigul/Tom/Aigul/Narrow_field_tensorflow/YOLO/drone_detector/backup/yolov2_last.weights"
        pass

    def select_path_to_model(self, path1, path2):
        self.PATH_TO_CFG = path1
        self.PATH_TO_WEIGHTS = path2
        self.init_df_graph()
        pass

    def init_df_graph(self):
        options = {"model": self.PATH_TO_CFG, "load": self.PATH_TO_WEIGHTS, "threshold": 0.1, "gpu": 1.0}
        self.tfnet = TFNet(options)
        pass

    def detect_by_yolo(self, img):
        results = self.tfnet.return_predict(img)

        newImage = np.copy(img)

        for result in results:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence > 0.5:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 2)
                newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                       (0, 230, 0), 1, cv2.LINE_AA)

        return newImage


