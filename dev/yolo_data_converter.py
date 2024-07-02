# import necessary libraries
import os
import shutil
import glob
import random
import tqdm
import xml.etree.ElementTree as ET

# define label map
label_map = {
    "D00": 0, # Longitudinal Crack
    "D10": 1, # Transverse Crack
    "D20": 2, # Alligator Crack
    "D40": 3 # Rutting, bump, pothole, separation
}

# convert bounding box to YOLO format
def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0] # inverse of width
    dh = 1. / size[1] # inverse of height
    x = (box[0] + box[1]) / 2.0 # calculate center x
    y = (box[2] + box[3]) / 2.0 # calculate center y
    w = box[1] - box[0] # calculate width
    h = box[3] - box[2] # calculate height
    
    # normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# parse an XML annotation file and convert annotations to YOLO format
def parse_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file) # parse the XML file
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text) # get the width
    height = int(size.find('height').text) # get the height
    
    yolo_annotations = [] # initialize a list to store YOLO annotations
    for obj in root.findall('object'): # iterate over all the elements
        class_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        
        # extract bounding box coordinates from XML
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_bbox_to_yolo((width, height), b) # convert bounding boxes to YOLO format
        yolo_annotations.append((class_name, bb))
    return yolo_annotations

# save YOLO annotations to a file
def save_yolo_annotations(yolo_annotations, class_mapping, output_file):
    with open(output_file, 'w') as f:
        for class_name, bbox in yolo_annotations: # iterate over YOLO annotations
            if class_name in class_mapping.keys(): # check if class name is in the mapping
                class_id = class_mapping[class_name] # get the class ID from the mapping
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

# Set up paths and load data
data_root_dir = "../data-collection/Japan_filtered/filtered"

source_image_path = os.path.join(data_root_dir, "images")
source_annot_path = os.path.join(data_root_dir, "annotations") 

# Ensure paths exist
assert os.path.exists(source_image_path), print("Image path not found")
assert os.path.exists(source_annot_path), print("Annotation path not found")

# get a list of all XML annotation files in the source directory
annot_list = glob.glob(source_annot_path+"/*.xml")

random.seed(42) # set the random seed for reproducibility

random.shuffle(annot_list) # shuffle the list

# split the list into training and validation sets
split_index = int(len(annot_list) * 0.8)
train_annots = annot_list[:split_index]
val_annots = annot_list[split_index:]

# define the directory path for images and labels
train_image_dir = os.path.join(data_root_dir, "train", "images")
train_label_dir = os.path.join(data_root_dir, "train", "labels")
val_image_dir = os.path.join(data_root_dir, "val", "images")
val_label_dir = os.path.join(data_root_dir, "val", "labels")

# create the directory for images and labels
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# process a list of annotation files, copying corresponding images and converting annotations to YOLO format
def process_annotations(annots, image_dir, label_dir):
    for annot_file in tqdm.tqdm(annots): # iterate over each annotation file with a progress bar
        image_file = os.path.join(source_image_path, os.path.basename(annot_file).replace(".xml", ".jpg"))
        if os.path.exists(image_file): # check if the image file exists 
            shutil.copy(image_file, image_dir) # Copy the image file to the target directory
            yolo_annotations = parse_xml_to_yolo(annot_file) # Parse the XML annotation and convert it to YOLO format
            output_file = os.path.join(label_dir, os.path.basename(annot_file).replace(".xml", ".txt")) # Construct the output file path for the YOLO annotation
            save_yolo_annotations(yolo_annotations, label_map, output_file) # Save the YOLO annotations to the output file
        else:
            print(f"Image file {image_file} not found for annotation {annot_file}")

process_annotations(train_annots, train_image_dir, train_label_dir)
process_annotations(val_annots, val_image_dir, val_label_dir)

print("Conversion to YOLO format completed successfully!")