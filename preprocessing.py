#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Pre-processing for ground truth and predicted files. We assume that files 
are previously converted to JSON format in the form:

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

# Pipeline of Pre-Processing:
  - Remove classes from faster and leannet that do not belong to ground truth
     $ python preprocessing.py -o faster_gt.json -m check_classes -g GT.json faster.json 
     $ python preprocessing.py -o leannet_gt.json -m check_classes -g GT.json leannet.json 

  - Apply threshold on faster_cor.json and leannet_cor.json [set to 0.5]
     $ python preprocessing.py -o leannet_0.5.json -m apply_threshold leannet_gt.json
     $ python preprocessing.py -o faster_0.5.json -m apply_threshold faster_gt.json

  - Remove all images that do not appear the three: faster.json, leannet.json and GT.json
     $ python preprocessing.py -o leannet_tmp.json -m align_files -g GT.json leannet_0.5.json
     $ python preprocessing.py -o faster_0.5f.json -m align_files -g leannet_tmp.json faster_0.5.json
     $ python preprocessing.py -o leannet_0.5f.json -m align_files -g faster_0.5f.json leannet_tmp.json
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
import sys
from os.path import join, dirname
from os.path import realpath, isfile
import json
import progressbar
import misc
import utils


def apply_threshold(file_predict, output, threshold):
    """
    Apply threshold on the scores of a predicted file, reducing
    the number of predicted bounding boxes.
    """
    dpred = utils.read_json(file_predict)
    
    dic = {}
    discarded = 0
    pb = progressbar.ProgressBar(len(dpred))
    for image in sorted(dpred):
        for content in dpred[image]:
            if content[1] >= threshold:
                if dic.has_key(image):
                    dic[image].append(content)
                else:
                    dic[image] = [content]
            else:
                discarded += 1
        pb.update()
    utils.save_json(output, dic)
    logger.info('Total of discarded bounding boxes: %d' % discarded)


def align_files(file_predict, file_ground, output):
    """
    Read ground truth and predicted files and keep only images that 
    appear in both files.
    """
    dground = utils.read_json(file_ground)
    dpredict = utils.read_json(file_predict)

    dic = {}
    aligned = 0
    for image in sorted(dground):
        if dpredict.has_key(image):
            dic[image] = dpredict[image]
            aligned += 1
        else:
            logger.info('Discarding image: %s' % image)
    utils.save_json(output, dic)
    logger.info('Total of aligned images: %d' % aligned)


def check_classes(file_predict, file_ground, output):
    """
    Ensure that predicted labels correspond to the ground truth
    """
    dground = utils.read_json(file_ground)
    dg = {}
    for image in dground:
        for obj in dground[image]:
            dg[obj[0]] = ''

    dic = {}
    dpredict = utils.read_json(file_predict)
    for image in dpredict:
        for obj in dpredict[image]:
            if dg.has_key(obj[0]):
                if dic.has_key(image):
                    dic[image].append(obj)
                else:
                    dic[image] = [obj]
            else:
                logger.info('Discarding bounding box of class: %s' % obj[0])
    utils.save_json(output, dic)


def main(file_predict, file_ground, output, mode, threshold):
    if not output:
        dirin = dirname(file_predict)
        output = join(dirin, 'output.json')

    if mode.lower() == 'align_files':
        align_files(file_ground, file_predict, output)
    elif mode.lower() == 'apply_threshold':
        apply_threshold(file_predict, output, threshold)
    elif mode.lower() == 'check_classes':
        check_classes(file_predict, file_ground, output)
    else:
        logger.error('Mode for pre-processing is not correct: %s' % mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted', metavar='file_predicted', help='File containing predicted bounding boxes', default=None)
    parser.add_argument('-g', '--groundtruth', help='File containing ground truth for all images', default=None)
    parser.add_argument('-o', '--output', help='File to save the generated json file', default=None)
    parser.add_argument('-t', '--threshold', help='Apply threshold on predicted scores', default=0.5)
    parser.add_argument('-m', '--mode', help='Mode of pre-processing (align_files|apply_threshold|check_classes)', default='align_files')
    args = parser.parse_args()

    main(args.predicted, args.groundtruth, args.output, args.mode, args.threshold)
