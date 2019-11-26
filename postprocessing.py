import numpy as np
from skimage.measure import regionprops, label
from skimage.measure import label as assign_label
from skimage.morphology import erosion as im_erosion
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from skimage.feature import peak_local_max

from skimage.segmentation import slic
import skimage.future.graph as gf


def smooth_emb(emb, radius):
    from scipy import ndimage
    from skimage.morphology import disk
    emb = emb.copy()
    w = disk(radius)/np.sum(disk(radius))
    for i in range(emb.shape[-1]):
        emb[:, :, i] = ndimage.convolve(emb[:, :, i], w, mode='reflect')
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb


def get_seeds(dist_map, thres=0.7):
    c = np.squeeze(dist_map)
    mask = peak_local_max(dist_map, min_distance=10, threshold_abs=thres * c.max(), indices=False)
    # mask = c > thres * c.max()
    return mask


def mask_from_seeds(embedding, seeds, similarity_thres=0.7):
    embedding = np.squeeze(embedding)
    seeds = label(seeds)
    props = regionprops(seeds)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]
        emb_mean = np.mean(embedding[row, col], axis=0)
        emb_mean = emb_mean/np.linalg.norm(emb_mean)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3))

        front_r, front_c = np.nonzero(seeds != dilated)

        similarity = [np.dot(embedding[r, c, :], mean[dilated[r, c]])
                      for r, c in zip(front_r, front_c)]
        
        # bg = seeds[front_r, front_c] == 0
        # add_ind = np.logical_and([s > similarity_thres for s in similarity], bg)
        add_ind = np.array([s > similarity_thres for s in similarity])

        if np.all(add_ind == False):
            break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds

def remove_noise(l_map, d_map, min_size=10, min_intensity=0.1):
    max_instensity = d_map.max()
    props = regionprops(l_map, intensity_image=d_map)
    for p in props:
        if p.area < min_size:
            l_map[l_map==p.label] = 0
        if p.mean_intensity/max_instensity < min_intensity:
            l_map[l_map==p.label] = 0
    return label(l_map)


