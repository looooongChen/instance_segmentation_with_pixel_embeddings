import numpy as np
import cv2
from skimage.measure import regionprops
from skimage import io as ski_io
from scipy.ndimage.morphology import distance_transform_edt

def relabel_map(map):
    map = map.copy()
    new_map = map.copy()
    index = 1
    for u in np.unique(map):
        if u == 0:
            continue
        new_map[map==u] = index
        index += 1
    return new_map

def remove_small(map, size, relabel=True):
    map = map.copy()
    props = regionprops(map)
    for p in props:
        if p.area < size:
            map[map == p.label] = 0
    if relabel:
        map = relabel_map(map)
    return map.astype(np.int32)


# def stack2map(s, close=0, remove_small=0):
#     s = np.squeeze(s)
#     assert len(s.shape) == 3
#     s = s > 0

#     map = np.zeros((s.shape[0], s.shape[1]), dtype=np.int32)

#     for i in range(s.shape[-1]):
#         obj = s[:, :, i]

#         if np.sum(obj) == 0:
#             continue

#         map[obj > 0] = i+1

#     if close != 0 or remove_small != 0:
#         map = process_map(map, close=close, remove_small=remove_small)

#     return map.astype(np.int32)


# def map2stack(map, close=0, remove_small=0):
#     map = np.squeeze(map)
#     assert len(map.shape) == 2

#     if close != 0 or remove_small != 0:
#         map = process_map(map, close=close, remove_small=remove_small)

#     unique = np.unique(map)
#     s = np.zeros((map.shape[0], map.shape[1], len(unique)-1), dtype=bool) \
#         if 0 in unique else np.zeros((map.shape[0], map.shape[1], len(unique)))

#     counter = 0
#     for i in range(len(unique)):
#         if unique[i] == 0:
#             continue
#         obj = map == unique[i]

#         s = s.astype(np.uint8)
#         s[:, :, counter] = obj
#         counter += 1
#     return s[:, :, 0:counter]


# def read_stack_from_files(files):
#     if len(files) == 0:
#         return None

#     for i, f in enumerate(files):
#         obj = ski_io.imread(f)
#         obj = obj[:, :, 0] if len(obj.shape) == 3 else obj

#         if i == 0:
#             s = np.zeros((obj.shape[0], obj.shape[1], len(files)), dtype=np.uint8)
#         s[:, :, i] = (obj>0).astype(np.uint8)

#     return s


def boundary_of_label_map(map):

    map = map.copy()

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(map.astype(np.uint16), kernel, iterations=1)
    dilation = cv2.dilate(map.astype(np.uint16), kernel, iterations=1)
    boundary = np.not_equal(erosion, dilation)
    return boundary

def distance_map(map, normalize=False):

    map = map.copy()
    boundary = boundary_of_label_map(map)
    map = np.multiply(map, 1-boundary)

    dist_map = cv2.distanceTransform((map>0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    if normalize:
        unique = np.unique(map)
        for u in unique:
            if u == 0:
                continue
            max_dist = np.max(dist_map[map==u])
            if max_dist != 0:
                dist_map[map == u] = dist_map[map == u]/max_dist
    return dist_map


# def centroid_map(label_map):
#     c_map = np.zeros(np.squeeze(label_map).shape)
#     for obj in regionprops(label_map):
#         c = obj['centroid']
#         c_map[int(c[0]), int(c[1])]=1
#     return c_map
    

def get_neighbor_by_distance(label_map, distance=10, max_neighbor=50):

    label_map = label_map.copy()

    def _adjust_size(x):
        if len(x) >= max_neighbor:
            return x[0:max_neighbor]
        else:
            return np.pad(x, (0, max_neighbor-len(x)), 'constant',  constant_values=(0, 0))

    unique = np.unique(label_map)
    assert unique[0] == 0
    # only one object
    if len(unique) <= 2:
        return None

    neighbor_indice = np.zeros((len(unique)-1, max_neighbor))
    label_flat = label_map.reshape((-1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance * 2 + 1, distance * 2 + 1))
    for i, label in enumerate(unique[1:]):
        assert i+1 == label
        mask = label_map == label
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).reshape((-1))
        neighbor_pixel_ind = np.logical_and(dilated_mask > 0, label_flat != 0)
        neighbor_pixel_ind = np.logical_and(neighbor_pixel_ind, label_flat != label)
        neighbors = np.unique(label_flat[neighbor_pixel_ind])
        neighbor_indice[i,:] = _adjust_size(neighbors) 

    return neighbor_indice.astype(np.int32)



if __name__ == '__main__':
    # test
    import os
    im = ski_io.imread('D:\Datasets\BBBC006_U2OScell\ground_truth\mcf-z-stacks-03212011_a12_s1_w197a9b240-1624-42e2-86a3-d50f7b607ff6.png')
     
    # dist = get_neighbor_by_distance(im, distance=20, max_neighbor=5)
    # print(dist)
    dist = distance_map(im, normalize=True)
    ski_io.imsave('test.png', im)


