from utils.tfrecords_convert import create_tf_record
import os
import random

image_dir = 'D:/Datasets/CVPPP2017_CodaLab/training_images'
gt_dir = 'D:/Datasets/CVPPP2017_CodaLab/training_truth'
examples = ['A1', 'A2', 'A3', 'A4']
val_ratio = 0.2
output_dir = './tfrecords/CVPPP2017_val'
# val_ratio = 0
# output_dir = './tfrecords/CVPPP2017'

neighbor_distance_in_percent = 0.02
resize = (512, 512)
dist_map = True
gt_type = 'label'
max_neighbor = 32


img_dict = {}
gt_dict = {}
for g in examples:
    for f in os.listdir(os.path.join(image_dir, g)):
        b, _ = os.path.splitext(f)
        img_dict[g+'_'+b] = os.path.join(image_dir, g, f)
    for f in os.listdir(os.path.join(gt_dir, g)):
        b, _ = os.path.splitext(f)
        gt_dict[g+'_'+b] = os.path.join(gt_dir, g, f)

keys = list(img_dict.keys())
random.shuffle(keys)
split = int((1-val_ratio) * len(keys))

if not os.path.exists(os.path.join(output_dir, 'train')):
    os.makedirs(os.path.join(output_dir, 'train'))
if split == len(keys):
    create_tf_record(img_dict,
                     gt_dict,
                     os.path.join(output_dir, 'train', 'train'),
                     neighbor_distance_in_percent=neighbor_distance_in_percent,
                     resize=resize,
                     dist_map=dist_map, 
                     gt_type=gt_type,
                     max_neighbor=max_neighbor)
else:
    img_dict_train = {k: img_dict[k] for k in keys[0:split]}
    gt_dict_train = {k: gt_dict[k] for k in keys[0:split]}

    img_dict_val = {k: img_dict[k] for k in keys[split:]}
    gt_dict_val = {k: gt_dict[k] for k in keys[split:]}

    create_tf_record(img_dict_train,
                    gt_dict_train,
                    os.path.join(output_dir, 'train', 'train'),
                    neighbor_distance_in_percent=neighbor_distance_in_percent,
                    resize=resize,
                    dist_map=dist_map, 
                    gt_type=gt_type,
                    max_neighbor=max_neighbor)

    if not os.path.exists(os.path.join(output_dir, 'val')):
        os.makedirs(os.path.join(output_dir, 'val'))
    create_tf_record(img_dict_val,
                    gt_dict_val,
                    os.path.join(output_dir, 'val', 'val'),
                    neighbor_distance_in_percent=neighbor_distance_in_percent,
                    resize=resize,
                    dist_map=dist_map, 
                    gt_type=gt_type,
                    max_neighbor=max_neighbor)




