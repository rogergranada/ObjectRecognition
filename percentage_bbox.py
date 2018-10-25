#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Functions that may help some tasks. 

"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
import sys
from os import walk
from os.path import join, isdir, dirname, basename, realpath, splitext
from xml.dom import minidom
import progressbar

def percentage_bounding_boxes(folder_input):
    """
    Generates the number of images for each value of percentage, i.e., size of
    the bouding boxes in relation to the size of the image. Input XML files have
    the form:

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
    """
    dic = {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 
           0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0}
    for root, dirs, files in walk(folder_input):
        total_images = len(files)
        pb = progressbar.ProgressBar(total_images)
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

            itemlist = xmlfile.getElementsByTagName('width')
            width = int(itemlist[0].childNodes[0].data)
            itemlist = xmlfile.getElementsByTagName('height')
            height = int(itemlist[0].childNodes[0].data)
            
            area_bbox = 0
            itemlist = xmlfile.getElementsByTagName('object')
            for obj in itemlist:
                for bndbox in obj.getElementsByTagName("bndbox"):
                    xmin = int(float(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data))
                    xmax = int(float(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data))
                    ymin = int(float(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data))
                    ymax = int(float(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data))
    
                area_bbox += (xmax - xmin) * (ymax - ymin)
            area_img = width * height
            ratio = float(area_bbox) / area_img
            if   ratio <= 0.1: dic[0.1] += 1
            elif ratio <= 0.2: dic[0.2] += 1
            elif ratio <= 0.3: dic[0.3] += 1
            elif ratio <= 0.4: dic[0.4] += 1
            elif ratio <= 0.5: dic[0.5] += 1
            elif ratio <= 0.6: dic[0.6] += 1
            elif ratio <= 0.7: dic[0.7] += 1
            elif ratio <= 0.8: dic[0.8] += 1
            elif ratio <= 0.9: dic[0.9] += 1
            elif ratio <= 1.0: dic[1.0] += 1
    for ratio in sorted(dic):
        logger.info('Ratio - Images: %f : %d' % (ratio, dic[ratio]))


def main(file_ground):
    file_ground = realpath(file_ground)
    percentage_bounding_boxes(file_ground)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('groundtruth', metavar='file_ground', help='File containing ground truth for all images')
    args = parser.parse_args()

    main(args.groundtruth)
