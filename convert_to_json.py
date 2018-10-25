#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Convert files from XML (ground truth) or TXT (predicted) into JSON files. 
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
import sys
from os import walk
from os.path import join, isdir, dirname, basename, splitext
from os.path import realpath
from xml.dom import minidom
import json
import progressbar


def convert_txt(folder_input, output):
    """
    Convert predicted files from LeanNet and Faster R-CNN from plain
    text files (TXT) to JSON file. Unlike ground truth files, predicted
    files are grouped by class instead of images. Thus, there is a single
    file for `bed` with its occurrences in all images. Input files 
    have the form:

    b_bedroom_indoor_0087 0.763 85.8 78.8 267.1 223.9
    b_bedroom_indoor_0087 0.205 7.2 109.3 185.1 204.9
    b_bedroom_indoor_0087 0.051 135.1 111.4 252.5 162.6

    where `b_bedroom_indoor_0087` is the name of the image, followed by
    predicted score, xmin, ymin, xmax, ymax values. Output file contains
    the form:

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
    dic = {}
    for root, dirs, files in walk(folder_input):
        pb = progressbar.ProgressBar(len(files))
        for name in sorted(files):
            pb.update()
            fname = join(root, name)
            _, ext = splitext(name)
            if ext != '.txt':
                logger.info("Skipping non TXT file: %s" % fname)
                continue

            # remove meta-tags from the name of the file
            label = name.replace('comp4_det_test_in_', '')
            label = label.replace('.txt', '')
            
            with open(fname) as fin:
                for line in fin:
                    arr = line.strip().split()
                    image = arr[0]+'.jpg'
                    score = float(arr[1])
                    xmin = int(float(arr[2]))
                    ymin = int(float(arr[3]))
                    xmax = int(float(arr[4]))
                    ymax = int(float(arr[5]))
                    if dic.has_key(image):
                        dic[image].append([label, score, xmin, ymin, xmax, ymax])
                    else:
                        dic[image] = [[label, score, xmin, ymin, xmax, ymax]]

    logger.info('Saving file %s' % output)
    with open(output, 'w') as outfile:
        json.dump(dic, outfile)
                    
    
def convert_xml(folder_input, output):
    """
    Convert ground truth files from XML format to JSON format. 
    Each input file represents the bounding boxes identified in
    an image with the same name. Input files have the form:

    <Annotation>
        <folder>SUN2009</folder>
        <filename>[name of the file].jpg</filename>
        <size>
            <width>300</width>
            <height>225</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>bed</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>16</xmin>
                <ymin>107</ymin>
                <xmax>273</xmax>
                <ymax>224</ymax>
            </bndbox>
        </object>
    </Annotation>
    
    The output file contains JSON dictionary with all files in the form:

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
    dic = {}
    for root, dirs, files in walk(folder_input):
        pb = progressbar.ProgressBar(len(files))
        for name in sorted(files):
            pb.update()
            fname = join(root, name)
            _, ext = splitext(name)
            if ext != '.xml':
                logger.info("Skipping non XML file: %s" % fname)
                continue
            xmlfile = minidom.parse(fname)
            if not xmlfile.getElementsByTagName('object'):
                continue
            itemlist = xmlfile.getElementsByTagName('filename')
            filename = itemlist[0].childNodes[0].data
            
            itemlist = xmlfile.getElementsByTagName('object')
            for obj in itemlist:
                for elem in obj.childNodes:
                    if elem.tagName == 'name':
                        label = elem.childNodes[0].data
                    if elem.tagName == 'bndbox':
                        for item in elem.childNodes:
                            if item.tagName == 'xmin': xmin = int(item.childNodes[0].data)
                            if item.tagName == 'ymin': ymin = int(item.childNodes[0].data)
                            if item.tagName == 'xmax': xmax = int(item.childNodes[0].data)
                            if item.tagName == 'ymax': ymax = int(item.childNodes[0].data)
                if dic.has_key(filename):
                    dic[filename].append([label, 1, xmin, ymin, xmax, ymax])
                else:
                    dic[filename]= [[label, 1, xmin, ymin, xmax, ymax]]

    logger.info('Saving file %s' % output)
    with open(output, 'w') as outfile:
        json.dump(dic, outfile)


def main(folder_input, type_input, output=None):
    folder_input = realpath(folder_input)
    if not isdir(folder_input):
        logger.error('Input is not a folder: %s' % folder_input)
        sys.error(0)

    if not output:
        fname = basename(normpath(folder_input))
        output = join(folder_input, fname+'.json')

    if type_input.lower() == 'xml':
        convert_xml(folder_input, output)
    elif type_input.lower() == 'txt':
        convert_txt(folder_input, output)
    else:
        logger.error('Type of files is not correct: %s' % type_input)
        sys.error(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfolder', metavar='folder_input', help='Folder containing files to be converted.')
    parser.add_argument('type', metavar='file_type', help='Type of input files (default: xml)', default='xml')
    parser.add_argument('-o', '--output', help='File to save the generated json file', default=None)
    args = parser.parse_args()

    main(args.inputfolder, args.type, output=args.output)
