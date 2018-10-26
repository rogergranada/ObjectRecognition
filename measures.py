#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Calculate Precision Recall and F-measure for images described in a JSON file in the form:

{"[name of the file].jpg": [
    [<class>, <score>, <xmin>, <ymin>, <xmax>, <ymax>],
    [<class>, <score>, <xmin>, <ymin>, <xmax>, <ymax>],
    ...
  ],
 "[name of the file].jpg": [
    [<class>, <score>, <xmin>, <ymin>, <xmax>, <ymax>],
    [<class>, <score>, <xmin>, <ymin>, <xmax>, <ymax>],
    ...
  ]
}  
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
import sys
from os.path import join, isdir, dirname, basename, splitext, realpath
import json
import numpy as np
import utils


def accurary_scores(dresults):
    """
    Calculate the precision and recall of predicted bounding boxes

    Parameters:
    -----------
    dresults : dict
        Dictionary containing true positives, false positives and false negatives
        The dictionary has the form:
        {'true_pos': int, 'false_pos': int, 'false_neg': int}
    """
    tp = dresults['true_pos']
    fp = dresults['false_pos']
    fn = dresults['false_neg']
    try:
        precision = float(tp)/(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(tp)/(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f_score = 2*(precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f_score = 0.0
    return (precision, recall, f_score)

####################
def calculate_iou(g_bbox, p_bbox):
    """
    Calculate Intersection over Union (IoU) of a pair of bounding boxes

    Parameters:
    -----------
        g_bbox list
            ground truth bounding box in the form [xmin, ymin, xmax, ymax]
        p_bbox: list
            predicted bounding box in the form [xmin, ymin, xmax, ymax]

    Returns:
    --------
        float: value of the IoU
    """
    g_xmin, g_ymin, g_xmax, g_ymax = g_bbox
    p_xmin, p_ymin, p_xmax, p_ymax = p_bbox

    if (g_xmin > g_xmax) or (g_ymin > g_bbox) or \
       (p_xmin > p_xmax) or (p_ymin > p_bbox):
        logger.error('Bounding box contain errors, e.g., xmin>max')
        sys.exit(0)

    if (g_xmax < p_xmin or p_xmax < g_xmin or \
        g_ymax < p_ymin or p_ymax < g_ymin):
        return 0.0

    far_x = np.min([g_xmax, p_xmax])
    near_x = np.max([g_xmin, p_xmin])
    far_y = np.min([g_ymax, p_ymax])
    near_y = np.max([g_ymin, p_ymin])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    p_bbox_area = (p_xmax - p_xmin + 1) * (p_ymax - p_ymin + 1)
    iou = float(inter_area) / (true_box_area + p_bbox_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


######################
def select_by_class(dic):
    """ For each image, create a dictionary with {label: [[bbox1], [bbox2],...]}
    """
    dclass = {}
    for img in sorted(dic):
        dcontent = {}
        for label, bbox in zip(dic[img]['classes'], dic[img]['boxes']):
            if dcontent.has_key(label):
                dcontent[label].append(bbox)
            else:
                dcontent[label] = [bbox]
        dclass[img] = dcontent
    return dclass


def image_results(g_img, p_img):
    all_classes = set(g_img.keys()+p_img.keys())
    results = { 'false_pos': 0, 'true_pos': 0, 'false_neg': 0 }
    for label in all_classes:
        if not g_img.has_key(label): g_img[label] = []
        if not p_img.has_key(label): p_img[label] = []
        dres = get_single_image_results(g_img[label], p_img[label], 0.5)
        results['false_pos'] += dres['false_pos']
        results['true_pos'] += dres['true_pos']
        results['false_neg'] += dres['false_neg']
    return results

    
def generate_results(file_ground, file_pred, output=None):
    if not output:
        fname, _ = splitext(basename(file_pred))
        output = join(dirname(file_pred), 'scores_'+fname+'.txt')
    logger.info('Saving file %s' % output)
    fout = open(output, 'w')

    with open(file_ground) as infile:
        dgt = json.load(infile)
    dg = select_by_class(dgt)
    with open(file_pred) as infile:
        dpred = json.load(infile)
    dp = select_by_class(dpred)

    # g_: ground p_: predicted
    for id, img in enumerate(sorted(dp)):
        g_img = dg[img]
        p_img = dp[img]
        dresults = image_results(g_img, p_img)
        scores = accurary_scores(dresults)
        fout.write('%s %f %f %f\n' % (img, scores[0], scores[1], scores[2]))
    fout.close()

def calculate_image(dpred, dground, iou):
    labels_pred = [obj[0] for obj in dpred[image]]
    labels_ground = [obj[0] for obj in dground[image]]
    labels = set(labels_pred).union(set(labels_ground))
    for label in labels:
        if dground.has_key(label):
            vg = dground[label]
        else:
            vg = []
        if dpred.has_key(label):
            vp = dpred[label]
        else:
            vp = []
        scores = get_single_image_results(vg, vp, 0.5)
        print scores
        

def calculate(file_predict, file_ground, output, threshold):
    dground = utils.read_json(file_ground) 
    dpred = utils.read_json(file_predict) 

    for image in dpred:
        dp = {}
        for obj in dpred[image]:
            label, _, xmin, ymin, xmax, ymax = obj
            if dp.has_key(label):
                dp[label].append([xmin, ymin, xmax, ymax])
            else:
                dp[label] = [[xmin, ymin, xmax, ymax]]
        dg = {}
        for obj in dground[image]:
            label, _, xmin, ymin, xmax, ymax = obj
            if dg.has_key(label):
                dg[label].append([xmin, ymin, xmax, ymax])
            else:
                dg[label] = [[xmin, ymin, xmax, ymax]]
        for label in 
        calculate_image(dp, dg, iou=threshold)


def main(file_predict, file_ground, output, threshold):
    file_predict = realpath(file_predict)
    file_ground = realpath(file_ground)
    if not output:
        dirin = dirname(file_predict)
        output = join(dirin, 'output.csv')

    calculate(file_predict, file_ground, output, threshold)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted', metavar='file_predicted', help='File containing predicted bounding boxes')
    parser.add_argument('groundtruth', metavar='file_ground', help='File containing ground truth for all images')
    parser.add_argument('-o', '--output', help='File to save the generated json file', default=None)
    parser.add_argument('-t', '--threshold', help='Apply threshold on Intersection over Union (IoU)', default=0.5)
    #parser.add_argument('-m', '--mode', help='Mode of pre-processing (align_files|apply_threshold)', default='align_files')
    args = parser.parse_args()

    main(args.predicted, args.groundtruth, args.output, args.threshold)
