import cv2
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

"""
Frozen inference graph files

- Framework
I used TensorFlow v1 Object Detection API to train my model.
[1] https://github.com/tensorflow/models/tree/master/research/object_detection

- Dataset
An annotated dataset used to train is the following:
[2] https://github.com/vatsl/TrafficLight_Detection-TensorFlowAPI

- Pre-trained Model
SSD-Mobilenet_v1 model
[3] http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

- How to Train
To use the API for TensorFlow v1 [1], the following commands are required:
```
$ git clone git@github.com:tensorflow/models.git
$ cd models
$ git checkout f7e99c0
```
Then download the dataset and a config file from [2], and
also download the pre-trained model.
I adopted the SSD-Mobilenet_v1 model as a pre-trained model [3].

Training can be started with the command:
```
python object_detection/train.py \
    --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config \
    --train_dir=data/sim_training_data/sim_data_capture
```

- Training Log
INFO: tensorflow: global step 4997: loss = 1.5760 (5.049 sec / step)
INFO: tensorflow: global step 4998: loss = 1.2266 (5.078 sec / step)
INFO: tensorflow: global step 4999: loss = 1.5215 (5.051 sec / step)
INFO: tensorflow: global step 5000: loss = 1.9217 (5.059 sec / step)
INFO: tensorflow: Stopping Training.
INFO: tensorflow: Finished training! Saving model to disk.

- Extract the Trained Model
```
python object_detection/export_inference_graph.py \
    --pipeline_config_path=config/ssd_mobilenet-traffic-udacity_sim.config \
    --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 \
    --output_directory=frozen_models/frozen_sim_mobile/
```
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
        self.TL_ID = 2  # Red light id
        self.CUT_DOWN_FACTOR = 3  # detection frequency is cut down to one-third

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

            Notice:
            To reduce the computational cost, the detection frequency is cut down to one-third.
            You may have to adjust self.CUT_DOWN_FACTOR.

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Cut down the detection frequency to one-third to reduce the computational cost
        if self.frame_count % self.CUT_DOWN_FACTOR != 0:
            self.frame_count = (self.frame_count + 1) % (self.CUT_DOWN_FACTOR + 1)
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

            result = TrafficLight.UNKNOWN
            for i in range(len(boxes)):
                class_id = int(classes[i])
                if class_id == self.TL_ID:
                    result = TrafficLight.RED
                    self.frame_count += 1
                    break

        self.previous_result = result
        return result
