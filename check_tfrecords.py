import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time
import shutil
from preprocess import extract_fn
from utils.img_io import save_indexed_png

dist_map_included = True

# dataset_dir = "./tfrecords/U2OScell/train"
# image_channels = 1
# image_depth = 'uint16'

dataset_dir = "./tfrecords/CVPPP2017/train"
image_channels = 3
image_depth = 'uint8'

test_dir = "./tfrecords_check"

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.mkdir(test_dir)
time.sleep(1)

tfrecords = [os.path.join(dataset_dir, f)
             for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
dataset = tf.data.TFRecordDataset(tfrecords)
dataset = dataset.map(lambda x: extract_fn(x, image_channels=image_channels, image_depth=image_depth, dist_map=dist_map_included))
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(100):
        sample = sess.run(next_element)
        print(sample['image/filename'].decode("utf-8")+": height {}, width {}".format(sample['image/height'], sample['image/width']))
        print("objects in total: {}".format(sample['image/obj_count']))

        cv2.imwrite(os.path.join(test_dir, 'image'+str(i)+'.tif'), sample['image/image'])
        save_indexed_png(os.path.join(test_dir, 'label'+str(i)+'.png'), sample['image/label'].astype(np.uint8))
        if dist_map_included:
            cv2.imwrite(os.path.join(test_dir, 'dist'+str(i)+'.png'), sample['image/dist_map']*255)
        
        # print(sample['image/neighbor'][:,0:10])
