import skimage as ski
import numpy as np
import os
from .visulize import visulize_mask


class Evaluator(object):

    def __init__(self, thres=None):
        # self.type = type
        self.APs = []
        self.mAP = None
        if thres is None:
            self.thres = np.arange(start=0.5, stop=0.95, step=0.05)
        else:
            self.thres = thres
        self.ap_dict = {i: [] for i, _ in enumerate(self.thres)}
        self.examples = []

    def add_example(self, pred, gt):
        e = Example(pred, gt)
        self.examples.append(e)

        aps = []
        for i, t in enumerate(self.thres):
            ap_ = e.get_ap(t)
            self.ap_dict[i].append(ap_)
            aps.append(ap_)
        return aps

    def save_last_as_image(self, fname, bg_image, thres=0.5, isBGR=False):
        self.examples[-1].save_as_image(fname, bg_image, thres=thres, isBGR=isBGR)

    def score(self):
        for i, _ in enumerate(self.thres):
            self.APs.append(np.mean(self.ap_dict[i]))
        self.mAP = np.mean(self.APs)
        return self.mAP, self.APs


class Example(object):

    """
    class for a prediction-ground truth pair
    """

    def __init__(self, pred, gt):
        self.pred = pred
        self.gt = gt
        self.gt_num = len(np.unique(gt)) - 1
        self.IoU_dict = {}  # (prediction label)-(IoU)
        self.match_dict = {}  # (prediction label)-(matched gt label)

        self.match_non_overlap(pred, gt)

    def match_non_overlap(self, pred, gt):
        pred_area = self.get_area_dict(pred)
        gt_area = self.get_area_dict(gt)
        unique = np.unique(pred)

        for label in unique:
            if label == 0:
                continue
            u, c = np.unique(gt[pred == label], return_counts=True)
            ind = np.argsort(c, kind='mergesort')
            if len(u) == 1 and u[ind[-1]] == 0:
                # only contain background
                self.IoU_dict[label] = 0
                self.match_dict[label] = None
            else:
                # take the gt label with the largest overlap
                i = ind[-2] if u[ind[-1]] == 0 else ind[-1]
                union = c[i]
                intersect = pred_area[label] + gt_area[u[i]] - c[i]
                self.IoU_dict[label] = union/intersect
                self.match_dict[label] = u[i]

    def get_area_dict(self, label_map):
        props = ski.measure.regionprops(label_map)
        return {p.label: p.area for p in props}

    def get_ap(self, thres):
        """
        compute ap for a certain dice value
        :param thres: dice value
        :return: ap
        """
        tp = 0
        match_gt = []
        for k, value in self.IoU_dict.items():
            if value > thres:
                tp = tp + 1
                match_gt.append(self.match_dict[k])
        tp_fp = len(self.IoU_dict)
        fn = self.gt_num -len(match_gt)
        return tp/(tp_fp+fn)

    def save_as_image(self, fname, bg_image, thres, isBGR=False):
        """
        save a visualization image, plot match in blue, non-match in red
        :param fname: path to save the image
        :param bg_image: original image
        :param thres: the dice value to determine match/non-match
        :param isBGR:
        :return:
        """
        if len(bg_image.shape) == 3 and isBGR:
            bg_image = bg_image[:, :, ::-1]
        tp = self.pred.copy()
        fp = self.pred.copy()
        for k, value in self.IoU_dict.items():
            if value > thres:
                fp[fp == k] = 0
            else:
                tp[tp == k] = 0
        res = visulize_mask(bg_image, tp, fill=True, color='blue')
        res = visulize_mask(res, fp, fill=True, color='red')
        print("Vis saved in: " + fname)
        ski.io.imsave(fname, res)
        return res


if __name__ == "__main__":
    pred = ski.io.imread('./test/pre_1.png', as_gray=True)
    gt = ski.io.imread('./test/gt_1.png', as_gray=True)
    pred = ski.measure.label(pred)
    e = Evaluator()
    e.add_example(pred, gt)
    e.save_last_as_image('./test.png', gt, 0.5)
    # pred = ski.io.imread('./test/pre_2.png', as_gray=True)
    # gt = ski.io.imread('./test/gt_2.png', as_gray=True)
    # pred = ski.measure.label(pred)
    # e.evaluate_single(pred, gt)
    # print(e.score())
    # # pred = gt
    # IoU_dict, match_dict = match_non_overlap(pred, gt)
    # print(IoU_dict)
    # print(match_dict)
    # print(get_ap_non_overlap(pred, gt, 0.8))
    # print(evaluate_dir_no_overlap('./test', "pre", "gt", 0.8))
