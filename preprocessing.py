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


def check_file(input):
    input = realpath(input)
    if not isfile(input):
        logger.error('Input is not a file: %s' % input)
        sys.error(0)
    return input


def save_json(output, dic):
    logger.info('Saving file %s' % output)
    with open(output, 'w') as outfile:
        json.dump(dic, outfile)


def read_json(input):
    logger.info('Reading file %s' % input)
    with open(input) as infile:
        dic = json.load(infile)
    return dic
        

def apply_threshold(file_predict, output, threshold):
    """
    Apply threshold on the scores of a predicted file, reducing
    the number of predicted bounding boxes.
    """
    dpred = read_json(file_predict)
    
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
    save_json(output, dic)
    logger.info('Total of discarded bounding boxes: %d' % discarded)


def align_files(file_predict, file_ground, output):
    """
    Read ground truth and predicted files and keep only images that 
    appear in both files.
    """
    dground = read_json(file_ground)
    dpredict = read_json(file_predict)

    dic = {}
    aligned = 0
    for image in sorted(dground):
        if dpredict.has_key(image):
            dic[image] = dpredict[image]
            aligned += 1
        else:
            logger.info('Discarding image: %s' % image)
    save_json(output, dic)
    logger.info('Total of aligned images: %d' % aligned)


def main(file_predict, file_ground, output, mode, threshold):
    if not output:
        dirin = dirname(file_predict)
        output = join(dirin, 'output.json')

    if mode.lower() == 'align_files':
        align_files(file_ground, file_predict, output)
    elif mode.lower() == 'apply_threshold':
        apply_threshold(file_predict, output, threshold)
    else:
        logger.error('Mode for pre-processing is not correct: %s' % mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted', metavar='file_predicted', help='File containing predicted bounding boxes', default=None)
    parser.add_argument('-g', '--groundtruth', help='File containing ground truth for all images', default=None)
    parser.add_argument('-o', '--output', help='File to save the generated json file', default=None)
    parser.add_argument('-t', '--threshold', help='Apply threshold on predicted scores', default=0.5)
    parser.add_argument('-m', '--mode', help='Mode of pre-processing (align_files|apply_threshold)', default='align_files')
    args = parser.parse_args()

    main(args.predicted, args.groundtruth, args.output, args.mode, args.threshold)
