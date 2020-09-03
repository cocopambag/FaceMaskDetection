import cv2
import time
import imutils

import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import pytorch_inference
from utils.meta import checkRotate 

import os


feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
b = time.time()
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
print("anchors time :",time.time()-b)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(model, image, target_shape, conf_thresh=0.5, iou_thresh=0.4, mode=1 ):

    image = np.array(image)[:, :, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0 
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)


        if class_id == 0:
            color = (0, 255, 0)
        else:
            if mode:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
                
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    
    return (output_info, image)

def run_on_video(model, video_path, output_video_name, conf_thresh):
    try:
        rotate = checkRotate(video_path)
        print(str(rotate))
    except Exception as e:
        rotate = 0
        print("get rotate fail")
        print(e)
        print(" ")
    try:
        cap = cv2.VideoCapture(video_path)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if rotate == 90 or rotate == 270:
            height, width = width, height

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
        print(output_video_name)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened():
            raise ValueError("Video open failed.")
            return
        status = True
        idx = 0
    except Exception as e :
        print(e)
        return {'msg': '[Video] Reading Error'}

    try:
        while status:
            start_stamp = time.time()
            status, img_raw = cap.read()
            
            if (status):
                img_raw = imutils.rotate(img_raw, 360-rotate)
                read_frame_stamp = time.time()
                result = inference(model,
                                img_raw,
                                target_shape=(360, 360),
                                conf_thresh=conf_thresh,
                                iou_thresh=0.4)
                cv2.waitKey(1)
                inference_stamp = time.time()
                writer.write(result[1])
                write_frame_stamp = time.time()
                idx += 1
                if idx%100 == 0:
                    print("%d of %d" % (idx, total_frames))
                    print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                        inference_stamp - read_frame_stamp,
                                                                        write_frame_stamp - inference_stamp))
    except Exception as e:
        print(e)
        return {'msg': '[Video] Detecting error'}
    writer.release()
    return {'msg':'Success'}