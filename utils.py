#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Functions that may help some tasks. 

"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys
from os.path import realpath, isfile
import progressbar
import json

def check_file(input):
    input = realpath(input)
    if not isfile(input):
        logger.error('Input is not a file: %s' % input)
        sys.error(0)
    return input


def save_json(output, dic):
    logger.info('Saving file %s' % output)
    with open(output, 'w') as outfile:
        json.dump(dout, outfile)


def read_json(input):
    logger.info('Reading file %s' % input)
    with open(input) as infile:
        dic = json.load(infile)
    return dic


