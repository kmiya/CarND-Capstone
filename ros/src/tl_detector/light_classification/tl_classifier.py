from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

"""
Frozen inference graph files
INFO: tensorflow: global step 4997: loss = 1.5760 (5.049 sec / step)
INFO: tensorflow: global step 4998: loss = 1.2266 (5.078 sec / step)
INFO: tensorflow: global step 4999: loss = 1.5215 (5.051 sec / step)
INFO: tensorflow: global step 5000: loss = 1.9217 (5.059 sec / step)
INFO: tensorflow: Stopping Training.
INFO: tensorflow: Finished training! Saving model to disk.
"""
SSD_GRAPH_FILE = 'model/ssd_mobilenet_v1_sim.pb'

"""
The following functions are based on the codes in the repository:
https://github.com/udacity/CarND-Object-Detection-Lab
"""


def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


# def to_image_coords(boxes, height, width):
#     """
#     The original box coordinate output is normalized, i.e [0, 1].

#     This converts it back to the original coordinate based on the image
#     size.
#     """
#     box_coords = np.zeros_like(boxes)
#     box_coords[:, 0] = boxes[:, 0] * height
#     box_coords[:, 1] = boxes[:, 1] * width
#     box_coords[:, 2] = boxes[:, 2] * height
#     box_coords[:, 3] = boxes[:, 3] * width

#     return box_coords


# cmap = ImageColor.colormap
# COLOR_LIST = sorted([c for c in cmap.keys()])


# def draw_boxes(image, boxes, classes, thickness=4):
#     """Draw bounding boxes on the image"""
#     draw = ImageDraw.Draw(image)
#     for i in range(len(boxes)):
#         bot, left, top, right = boxes[i, ...]
#         class_id = int(classes[i])
#         color = COLOR_LIST[class_id]
#         draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
#     return draw


class TLClassifier(object):
    def __init__(self):
        self.detection_graph = load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.frame_count = 0
        self.previous_result = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            """
        # reduce detection frequency to one-third
        if self.frame_count % 3 != 0:
            self.frame_count = (self.frame_count + 1) % 4
            return self.previous_result

        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(cv_image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # # The current box coordinates are normalized to a range between 0 and 1.
            # # This converts the coordinates actual location on the image.
            # width, height, _ = image.shape
            # box_coords = to_image_coords(boxes, height, width)

            result = TrafficLight.UNKNOWN
            for i in range(len(boxes)):
                class_id = int(classes[i])
                TL_ID = 2  # red light id
                if class_id == TL_ID:
                    result = TrafficLight.RED
                    # print("Traffic light found!" + str(self.frame_count))
                    self.frame_count += 1
                    break

                    # Each class with be represented by a differently colored box
            # pil_image = image.copy()
            # pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # pil_image = Image.fromarray(pil_image)
            # draw_boxes(pil_image, box_coords, classes)
            # filename = 'camera/camera' + str(self.image_count) + ".jpg"
            # pil_image.save(filename)
            # self.image_count += 1
        self.previous_result = result
        return result
