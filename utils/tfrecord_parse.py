import tensorflow as tf


def extract_fn_base(data_record, image_depth='uint8'):
    feature_dict = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/image': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string)
    }
    sample = tf.parse_single_example(data_record, feature_dict)
    if image_depth == 'uint8':
        sample['image/image'] = tf.image.decode_png(sample['image/image'], dtype=tf.uint8)    
    else:
        sample['image/image'] = tf.image.decode_png(sample['image/image'], dtype=tf.uint16)
    
    return sample


def extract_fn_local_dis(data_record, image_depth='uint8', dist_map=False):

    sample = extract_fn_base(data_record, image_depth)

    feature_dict = {
        'image/label': tf.FixedLenFeature([], tf.string),
        'image/neighbor': tf.FixedLenFeature([], tf.string),
        'image/obj_count': tf.FixedLenFeature([], tf.int64),
        'image/max_neighbor': tf.FixedLenFeature([], tf.int64)
    }

    if dist_map:
        feature_dict['image/dist_map'] = tf.FixedLenFeature([], tf.string)

    sample_add = tf.parse_single_example(data_record, feature_dict)

    sample['image/label'] = tf.image.decode_png(sample_add['image/label'], dtype=tf.uint16)
    sample['image/neighbor'] = tf.reshape(tf.decode_raw(sample_add['image/neighbor'], tf.int32),
                                          tf.stack([sample_add['image/obj_count'], sample_add['image/max_neighbor']]))
    sample['image/obj_count'] = sample_add['image/obj_count']
    sample['image/max_neighbor'] = sample_add['image/max_neighbor']

    if dist_map:
        sample['image/dist_map'] = tf.image.decode_png(sample_add['image/dist_map'])

    return sample

