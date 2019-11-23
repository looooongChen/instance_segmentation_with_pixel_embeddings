import skimage as ski
from skimage.morphology import binary_dilation, disk
import numpy as np
import os
from scipy.spatial import distance_matrix


class Evaluator(object):

    def __init__(self, thres=None, gt_type="mask", line_match_thres=3):
        # self.type = type

        if thres is None:
            # self.thres = np.arange(start=0.5, stop=1, step=0.1)
            self.thres = [0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            self.thres = thres

        self.gt_type = gt_type
        self.line_match_thres = line_match_thres

        self.examples = []
        self.total_pred = 0
        self.total_gt = 0
        
        # self.IoU = []  # (prediction label)-(IoU)
        # self.recall = []
        # self.precision = []
        

    def add_example(self, pred, gt):
        e = Example(pred, gt, self.gt_type, self.line_match_thres)
        self.examples.append(e)

        self.total_pred += e.pred_num
        self.total_gt += e.gt_num
        print("example added, total: ", len(self.examples))
        # self.IoU[0:0] = list(e.IoU.values())
        # self.recall[0:0] = list(e.recall.values())
        # self.precision[0:0] = list(e.precision.values())

    def eval(self, metric='IoU'):

        res = {}
        for t in self.thres:
            pred_match = 0
            gt_match = 0
            for e in self.examples:
                p_m, g_m = e.return_match_num(t, metric)
                pred_match += p_m
                gt_match += g_m
            res[metric + '_' + str(t)] = [pred_match/self.total_pred, gt_match/self.total_gt]

        for k, v in res.items():
            print(k, v)


    # def save_last_as_image(self, fname, bg_image, thres=0.5, isBGR=False):
    #     self.examples[-1].save_as_image(fname, bg_image, thres=thres, isBGR=isBGR)

    # def score(self):
    #     for i, _ in enumerate(self.thres):
    #         self.APs.append(np.mean(self.ap_dict[i]))
    #     self.mAP = np.mean(self.APs)
    #     return self.mAP, self.APs


class Example(object):

    """
    class for a prediction-ground truth pair
    single_slide: faster when object number is high, but can not handle overlap
    type: "line or "mask"
    """

    def __init__(self, pred, gt, gt_type='mask', line_match_thres=3):
        self.gt_type = gt_type
        self.line_match_thres = line_match_thres

        pred = np.squeeze(pred)
        gt = np.squeeze(gt)

        if pred.ndim == 2 and gt.ndim == 2:
            self.single_slide = True
            self.pred = ski.measure.label(pred>0)
            self.gt = ski.measure.label(gt>0)
            self.gt_num = len(np.unique(self.gt)) - 1
            self.pred_num = len(np.unique(self.pred)) - 1
        else:
            self.single_slide = False
            self.pred = self.map2stack(pred)
            self.gt = self.map2stack(gt)
            self.gt_num = self.gt.shape[0]
            self.pred_num = self.pred.shape[0]

        self.match_dict = {}  # (prediction label)-(matched gt label)
        self.IoU = {}  # (prediction label)-(IoU)
        self.recall = {}
        self.precision = {}
        
        self._match_non_overlap()

        # print(len(self.match_dict), len(self.IoU), len(self.recall), len(self.precision), self.gt_num, self.pred_num)

    def _match_non_overlap(self):
        self.pred_area = self.get_area_dict(self.pred)
        self.gt_area = self.get_area_dict(self.gt)

        for label, pred_area in self.pred_area.items():
            self.IoU[label] = 0
            self.match_dict[label] = 0
            self.recall[label] = 0
            self.precision[label] = 0
            if self.gt_type == "mask":
                if self.single_slide:
                    u, c = np.unique(self.gt[self.pred == label], return_counts=True)
                    ind = np.argsort(c, kind='mergesort')
                    if len(u) == 1 and u[ind[-1]] == 0:
                        # only contain background
                        self.IoU[label] = 0
                        self.match_dict[label] = 0
                        self.recall[label] = 0
                        self.precision[label] = 0
                    else:
                        # take the gt label with the largest overlap
                        i = ind[-2] if u[ind[-1]] == 0 else ind[-1]
                        intersect = c[i]
                        union = pred_area + self.gt_area[u[i]] - intersect
                        self.IoU[label] = intersect/union
                        self.match_dict[label] = u[i]
                        self.recall[label] = intersect/self.gt_area[u[i]]
                        self.precision[label] = intersect/pred_area
                else:
                    intersect = np.multiply(self.gt, np.expand_dims(self.pred[label-1], axis=0))
                    intersect = np.sum(intersect, axis=(1,2))
                    ind = np.argsort(intersect, kind='mergesort')
                    if intersect[ind[-1]] == 0:
                        # no overlapp with any object
                        self.IoU[label] = 0
                        self.match_dict[label] = 0
                        self.recall[label] = 0
                        self.precision[label] = 0
                    else:
                        # take the gt label with the largest overlap
                        union = pred_area + self.gt_area[ind[-1]+1] - intersect[ind[-1]]
                        self.IoU[label] = intersect[ind[-1]]/union
                        self.match_dict[label] = ind[-1] + 1
                        self.recall[label] = intersect[ind[-1]]/self.gt_area[ind[-1]+1]
                        self.precision[label] = intersect[ind[-1]]/pred_area
            else:
                intersect = []
                if self.single_slide:
                    pts_pred = np.transpose(np.array(np.nonzero(self.pred==label)))
                    for l in np.unique(self.gt):
                        if l == 0:
                            continue
                        pts_gt = np.transpose(np.array(np.nonzero(self.gt==l)))
                        bpGraph = distance_matrix(pts_pred, pts_gt) < self.line_match_thres
                        g = GFG(bpGraph)
                        intersect.append(g.maxBPM())
                else:
                    pts_pred = np.transpose(np.array(np.nonzero(self.pred[label-1]>0)))
                    for g in self.gt:
                        pts_gt = np.transpose(np.array(np.nonzero(g>0)))
                        bpGraph = distance_matrix(pts_pred, pts_gt) < self.line_match_thres
                        g = GFG(bpGraph)
                        intersect.append(g.maxBPM())
                
                if len(intersect) != 0:
                    intersect = np.array(intersect)
                    ind = np.argsort(intersect, kind='mergesort')
                    if intersect[ind[-1]] != 0:
                        # take the gt label with the largest overlap
                        union = pred_area + self.gt_area[ind[-1]+1] - intersect[ind[-1]]
                        self.IoU[label] = intersect[ind[-1]]/union
                        self.match_dict[label] = ind[-1] + 1
                        self.recall[label] = intersect[ind[-1]]/self.gt_area[ind[-1]+1]
                        self.precision[label] = intersect[ind[-1]]/pred_area

    def get_area_dict(self, label_map):
        if self.single_slide:
            props = ski.measure.regionprops(label_map)
            area_dict = {p.label: p.area for p in props}
        else:
            area_dict = {i+1: np.sum(label_map[i]>0) for i in range(label_map.shape[0])}
        if 0 in area_dict.keys():
            del area_dict[0]
        return area_dict

    def map2stack(self, map):
        map = np.squeeze(map)
        if map.ndim == 2:
            stack = []
            for l in np.unique(map):
                if l == 0:
                    continue
                stack.append(map==l)
            return np.array(stack)>0
        else:
            return map>0

    def return_match_num(self, thres, metric='IoU'):
        match_label = np.array(list(self.match_dict.values()))
        if metric=='F':
            ind = (np.array(list(self.precision.values())) + np.array(list(self.recall.values())))/2 > thres
        else:
            ind = np.array(list(self.IoU.values())) > thres
        return np.sum(ind), len(np.unique(match_label[ind]))
  
class GFG:   
    # maximal Bipartite matching. 
    def __init__(self,graph): 
          
        # residual graph 
        self.graph = graph  
        self.ppl = len(graph) 
        self.jobs = len(graph[0]) 
  
    # A DFS based recursive function 
    # that returns true if a matching  
    # for vertex u is possible 
    def bpm(self, u, matchR, seen): 
  
        # Try every job one by one 
        for v in range(self.jobs): 
  
            # If applicant u is interested  
            # in job v and v is not seen 
            if self.graph[u][v] and seen[v] == False: 
                  
                # Mark v as visited 
                seen[v] = True 
  
                '''If job 'v' is not assigned to 
                   an applicant OR previously assigned  
                   applicant for job v (which is matchR[v])  
                   has an alternate job available.  
                   Since v is marked as visited in the  
                   above line, matchR[v]  in the following 
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v],  
                                               matchR, seen): 
                    matchR[v] = u 
                    return True
        return False
  
    # Returns maximum number of matching  
    def maxBPM(self): 
        '''An array to keep track of the  
           applicants assigned to jobs.  
           The value of matchR[i] is the  
           applicant number assigned to job i,  
           the value -1 indicates nobody is assigned.'''
        matchR = [-1] * self.jobs 
          
        # Count of jobs assigned to applicants 
        result = 0 
        for i in range(self.ppl): 
              
            # Mark all jobs as not seen for next applicant. 
            seen = [False] * self.jobs 
              
            # Find if the applicant 'u' can get a job 
            if self.bpm(i, matchR, seen): 
                result += 1
        return result 
