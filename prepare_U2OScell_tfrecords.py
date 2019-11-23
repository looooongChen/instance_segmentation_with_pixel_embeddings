from utils.tfrecords_convert import create_tf_record
import os
import random

img_dir = 'd:/Datasets/BBBC006_U2OScell/images'
gt_dir = 'd:/Datasets/BBBC006_U2OScell/ground_truth'
val_ratio = 0.2
output_dir = './tfrecords/U2OScell'

neighbor_distance_in_percent = 0.02
resize = (512, 512)
dist_map = True
gt_type = 'label'
max_neighbor = 32

assert os.path.exists(img_dir)
assert os.path.exists(gt_dir)
if not os.path.exists(os.path.join(output_dir, 'train')):
    os.makedirs(os.path.join(output_dir, 'train'))
if not os.path.exists(os.path.join(output_dir, 'val')):
    os.makedirs(os.path.join(output_dir, 'val'))

img_dict = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))}
gt_dict = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))}

for k in gt_dict.keys():
    if k not in img_dict.keys():
        del gt_dict[k]

keys = list(gt_dict.keys())
random.shuffle(keys)
split = int((1-val_ratio) * len(keys))

img_dict_train = {k: img_dict[k] for k in keys[0:split]}
gt_dict_train = {k: gt_dict[k] for k in keys[0:split]}

img_dict_val = {k: img_dict[k] for k in keys[split:]}
gt_dict_val = {k: gt_dict[k] for k in keys[split:]}

# print(img_dict)

create_tf_record(img_dict_train,
                 gt_dict_train,
                 os.path.join(output_dir, 'train', 'train'),
                 neighbor_distance_in_percent=neighbor_distance_in_percent,
                 resize=resize,
                 dist_map=dist_map, 
                 gt_type=gt_type,
                 max_neighbor=max_neighbor)

create_tf_record(img_dict_val,
                 gt_dict_val,
                 os.path.join(output_dir, 'val', 'val'),
                 neighbor_distance_in_percent=neighbor_distance_in_percent,
                 resize=resize,
                 dist_map=dist_map, 
                 gt_type=gt_type,
                 max_neighbor=max_neighbor)

