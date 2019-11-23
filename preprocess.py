import tensorflow as tf
from utils.tfrecord_parse import extract_fn_local_dis


MAX_INSTANCE = 500

def extract_fn(data_record, 
               image_channels,
               image_depth='uint16', 
               dist_map=False):

    sample = extract_fn_local_dis(data_record, image_depth=image_depth, dist_map=dist_map)

    sample['image/image'].set_shape([None, None, image_channels])
     # sample['image/image'] = tf.image.per_image_standardization(sample['image/image']) # moved to Net.py
    sample['image/label'] = tf.cast(sample['image/label'], dtype=tf.int32)
    if dist_map:
        sample['image/dist_map'] = sample['image/dist_map']/255

    # recommend not to use tensorflow resize function !!! not !!!
    # if resize is not None:
    #     img = tf.image.resize_images(img, resize, align_corners=False)
    #     label = tf.image.resize_images(label, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
    #     if dist_map:
    #         d_map = tf.image.resize_images(d_map, resize, align_corners=False)
  
    # pad neighbor list to the same size (to form batches, the same size is required)
    neighbor = sample['image/neighbor']
    padding = [[0, MAX_INSTANCE - tf.shape(neighbor)[0]], [0, 0]]
    sample['image/neighbor'] = tf.pad(neighbor, padding, 'CONSTANT', constant_values=0)

    return sample

if __name__ == "__main__":
    tf.enable_eager_execution()
    dataset = tf.data.TFRecordDataset(['d:/Datasets/DSB2018/tfrecords/stage1_train/DSB2018.record-00000-of-00005'])
    dataset = dataset.map(lambda example: extract_fn(example, [512, 512]))
    iterator = dataset.make_one_shot_iterator()
    img, label, dist, neigbro = iterator.get_next()

    import numpy as np
    l = np.min(label)
    print(l)
