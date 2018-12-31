# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )


    cap = cv2.VideoCapture()
    cap.open(0)
    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            while True:
                ret, img = cap.read()
                img_resized = cv2.resize(img,(int(416),int(416)))


                t0 = time.time()
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)
                print("Predictions found in {:.2f}s".format(time.time() - t0))

                # img = draw_boxes(filtered_boxes, img, classes, (416, 416), True)
                print (filtered_boxes)
                cv2.imshow("Output", img)


if __name__ == '__main__':
    tf.app.run()
