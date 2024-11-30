from io import BytesIO
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


def reconstruct(pb_path):
    if not os.path.isfile(pb_path):
        print(f"Erro: {pb_path} não encontrado")
        return None
    print(f"Caminho: {pb_path}")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def detect(image):
    detection_graph = reconstruct("files/ssd_mobilenet_v2_taco_2018_03_29.pb")
    if detection_graph is None:
        return None

    with open("files/annotations.json") as json_file:
        data = json.load(json_file)
    categories = data['categories']
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image_np = np.expand_dims(np.array(image), axis=0)
            boxes, scores, classes, num = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np}
            )

            npim = np.array(image)
            vis_util.draw_bounding_boxes_on_image_array(
                npim,
                np.squeeze(boxes),
                color='yellow',
                thickness=2
            )

            output_image = Image.fromarray(npim)
            byte_io = BytesIO()
            output_image.save(byte_io, format='JPEG')
            byte_io.seek(0)
            image_base64 = base64.b64encode(byte_io.read()).decode('utf-8')
            
            detection = list()
            threshold = 0.5
            
            for i in range(int(num[0])):
                class_id = int(classes[0][i])
                score = scores[0][i]
                print(f"{class_id}: {category_index[class_id]['name']} - {float(score)}")
                if score > threshold:
                    detection.append({ 'category': class_id, 'score': float(score) })
            
            output = { 'image': image_base64, 'detection': detection }

            return output

