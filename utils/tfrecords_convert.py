import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import cv2

from . import tfrecord_creation, tfrecord_type
from . import process
from img_io import read_indexed_png

MAX_NEIGHBOR = 32

def convert_to_tf_example(img_path,
                          gt_path,
                          neighbor_distance_in_percent=0.02,
                          resize=None,
                          dist_map=False,
                          gt_type="label",
                          max_neighbor=MAX_NEIGHBOR):
    
    characteristicLength = 3

    # inject images 
    feature_dict = tfrecord_creation.inject_fn_img(img_path, resize)
    # read and process ground truth image
    if gt_type == "indexed":
        gt, _ = read_indexed_png(gt_path)
    else:
        gt = cv2.imread(gt_path, -1)
    if resize is not None:
        gt = cv2.resize(gt, resize, interpolation=cv2.INTER_NEAREST)
    label = process.remove_small(gt, size=25, relabel=True)
    # ignore images which contains only one object
    unique = np.unique(label)
    assert unique[0] == 0
    if len(unique) <= 2:
        print("Omit an image {} containing only one object".format(img_path))
        return None
    # save label map as uint16 image
    label = label.astype(np.uint16)
    _, label_encoded = cv2.imencode('.png', label)
    label_encoded = label_encoded.tobytes()
    feature_dict['image/label'] = tfrecord_type.bytes_feature(label_encoded)
    feature_dict['image/obj_count'] = tfrecord_type.int64_feature(len(unique)-1)
    # save neighbor relationship
    neighbor_distance = int(label.shape[1] * neighbor_distance_in_percent)
    neighbors = process.get_neighbor_by_distance(label, distance=neighbor_distance, max_neighbor=max_neighbor)
    feature_dict['image/neighbor'] = tfrecord_type.bytes_feature(neighbors.reshape(-1).tostring())
    feature_dict['image/max_neighbor'] = tfrecord_type.int64_feature(neighbors.shape[1])
    # save dist_map as uint8 image, resolution 1/255=0.0039
    if dist_map:
        d_map = process.distance_map(label, normalize=True)
        d_map = (((d_map-d_map.min())/(d_map.max()-d_map.min()))*255).astype(np.uint8)
        _, d_map_encoded = cv2.imencode('.png', d_map)
        d_map_encoded = d_map_encoded.tobytes()
        feature_dict['image/dist_map'] = tfrecord_type.bytes_feature(d_map_encoded)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


def create_tf_record(image_dict, 
                     gt_dict, 
                     output_file,
                     neighbor_distance_in_percent=0.02,
                     resize=None, 
                     dist_map=False,
                     num_shards=5,
                     gt_type="label",
                     max_neighbor=MAX_NEIGHBOR):
    
    """Creates a TFRecord file from examples.

    Args:
    image_dict: dict of image paths
    gt_dict: dict of ground truth paths
    output_file: name of tfrecord files
    num_shards: number of tfrecord shards
    generate_dist_map: generate distance map or not
    max_neighbor: max. number of neighbors saved 
    """
    import contextlib2

    # image_dict = name_dict(image_list)
    # gt_dict = name_dict(gt_list)

    total = len(gt_dict)

    with contextlib2.ExitStack() as tf_record_close_stack:
        
        output_tfrecords = tfrecord_creation.open_sharded_output_tfrecords(
            tf_record_close_stack, output_file, num_shards)
        
        processed_count = 0
        count = 0
        for k, gt_path in gt_dict.items():
            if k in image_dict.keys():
                tf_example = convert_to_tf_example(image_dict[k],
                                                   gt_path,
                                                   neighbor_distance_in_percent=neighbor_distance_in_percent,
                                                   resize=resize,
                                                   dist_map=dist_map,
                                                   gt_type=gt_type,
                                                   max_neighbor=max_neighbor)
                if tf_example is not None:
                    processed_count += 1
                    shard_idx = processed_count % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                
                count += 1

                if count % 10 == 0:
                    print('On image {} of {}, processed images: {}'.format(count, total, processed_count))

                # debug
                # if count == 50:
                #     break
