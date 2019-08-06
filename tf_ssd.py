import numpy as np
import sys
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2 as cv
from object_detection.utils import ops as utils_ops

IMAGE_WIDTH = 300  # 227
IMAGE_HEIGHT = 300  # 227
MODEL_NAME = 'output'
MODEL_NAME2 = 'annotations'
NUM_CLASSES = 2

class tf_classifier():

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = (image.shape[1], image.shape[2])  # image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def select_path_to_model(self, path):
        self.PATH_TO_CKPT = path
        self.init_tf_graph()
        pass

    def init_tf_graph(self):
        self.img_x = 227
        self.img_y = 227
        self.col_ch = 3
        sys.path.append("..")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        with self.detection_graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.ops = tf.get_default_graph().get_operations()
            self.all_tensor_names = {output.name for op in self.ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                self.tensor_name = key + ':0'
                if self.tensor_name in self.all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(self.tensor_name)
        pass

    def __init__(self):  # full abs path to .caffemodel
        self.PATH_TO_CKPT = 'output/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'phantom_mavic_label_map.pbtxt'
        pass

    def run_inference_for_single_image(self, image):  # , graph):
        if 'detection_masks' in self.tensor_dict:
            detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
            real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,
                                                                                  image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        image_exp = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        output_dict = self.sess.run(self.tensor_dict, feed_dict={image_tensor: image_exp})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def transform_img(self, img, img_width, img_height, eq):
        if eq:
            if len(img.shape) == 2:
                img = cv.equalizeHist(img)
            elif len(img.shape) == 3:
                img[:, :, 0] = cv.equalizeHist(img[:, :, 0])
                img[:, :, 1] = cv.equalizeHist(img[:, :, 1])
                img[:, :, 2] = cv.equalizeHist(img[:, :, 2])
            else:
                return np.zeros(img_height, img_width, 3)
        img = cv.resize(img, (img_width, img_height), interpolation=cv.INTER_CUBIC)
        img = np.multiply(img, 0.003921)
        return img

    def forwardpass(self, img):

        output_dict = self.run_inference_for_single_image(img)
        vis_util.visualize_boxes_and_labels_on_image_array(img, output_dict['detection_boxes'],
                                                           output_dict['detection_classes'],
                                                           output_dict['detection_scores'], self.category_index,
                                                           instance_masks=output_dict.get('detection_masks'),
                                                           use_normalized_coordinates=True, line_thickness=2)
        return img

    def get_probs(self):
        return self.prediction


