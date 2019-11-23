import numpy as np
import skimage as ski

RED = (210, 20, 55)
BLUE = (25, 100, 230)


def mask_color_img(img, mask, color='red', alpha=0.3):
    if color == 'red':
        color = RED
    elif color == 'blue':
        color = BLUE
    elif isinstance(color, str):
        color = BLUE
    else:
        color = color

    if img.ndim != 3:
        img = ski.color.gray2rgb(img)
    mask = (mask > 0).astype(np.uint8)
    layer = img.copy()
    for i in range(3):
        layer[:, :, i] = np.multiply(layer[:, :, i], 1-mask)+color[i]*mask
    res = (1-alpha)*img + alpha*layer
    return res.astype(np.uint8)


def visulize_mask(img, label_map, fill=False, color='blue'):
    b = get_boundary_from_label_map(label_map)
    overlayed = mask_color_img(img, b, color=color, alpha=0.8)
    if fill:
        mask = np.multiply((label_map > 0).astype(np.uint8), 1-b)
        overlayed = mask_color_img(overlayed, mask, color=color, alpha=0.2)
    return overlayed


def get_boundary_from_label_map(map):
    map = map.copy()

    kernel = np.ones((3, 3), np.uint8)
    erosion = ski.morphology.erosion(map, kernel)
    dilation = ski.morphology.dilation(map, kernel)
    boundary = np.not_equal(erosion, dilation)
    return boundary


if __name__ == "__main__":
    from skimage.io import imsave, imread
    import skimage as ski
    pred = imread('./test/pre_2.png', as_gray=True)
    gt = imread('./test/gt_2.png')
    pred = ski.measure.label(pred)

    im = visulize_mask(gt, pred, fill=True)
    imsave('./vis.png', im.astype(np.uint8))
