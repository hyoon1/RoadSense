import glob
import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def main():
    # 1. Set up paths and load data
    data_root_dir = "../data-collection/RDD2022_Japan"
    nationality = "Japan"
    mode = "train"

    # Construct destination paths
    image_destination_path = os.path.join(data_root_dir, nationality, mode, "images")
    annot_destination_path = os.path.join(data_root_dir, nationality, mode, "annotations/xmls")

    # Ensure paths exist
    assert os.path.exists(image_destination_path), "Image path not found"
    assert os.path.exists(annot_destination_path), "Annotation path not found"

    # Get list of image and annotation files
    image_list = glob.glob(image_destination_path + "/*.jpg")
    annot_list = glob.glob(annot_destination_path + "/*.xml")

    # Define label map
    label_map = {
        "D00": "Wheel mark part",
        "D01": "Construction joint part",
        "D10": "Equal interval",
        "D11": "Construction joint part",
        "D20": "Partial pavement",
        "D40": "Rutting, bump, pothole, separation",
        "D43": "Crosswalk blur",
        "D44": "White line blur",
        "D50": "Unknown"
    }

    # Define functions for reading and parsing XML
    def read_and_parse_xml(xml_file_path):
        """Reads and parses an XML file."""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        return root

    def encoded_obj_annotations(annotations):
        """Encodes object annotations."""
        objects_names = []
        for obj in annotations.findall('object'):
            label_code = obj.find('name').text
            objects_names.append(label_code)
        return objects_names

    def process_annotation_file(xml_file_path):
        """Processes a single annotation file."""
        annotations = read_and_parse_xml(xml_file_path)
        objects_names = encoded_obj_annotations(annotations)
        return xml_file_path, objects_names

    # Function to process all annotation files in parallel and show progress
    def process_all_annotations(xml_files):
        """Processes all annotation files in parallel and shows progress."""
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_xml = {executor.submit(process_annotation_file, xml_file): xml_file for xml_file in xml_files}
            for future in tqdm(as_completed(future_to_xml), total=len(future_to_xml), desc="Processing annotations"):
                xml_file = future_to_xml[future]
                try:
                    xml_file_path, objects_names = future.result()
                    results.append((xml_file_path, objects_names))
                except Exception as exc:
                    print(f"{xml_file} generated an exception: {exc}")
        return results

    print("Processing annotation files...")
    # Process all annotations
    results = process_all_annotations(annot_list)

    # Convert the data to a DataFrame
    df = pd.DataFrame(results, columns=['xml_file_path', 'objects'])

    # Define the labels you want to keep
    include_labels = {"D00", "D10", "D20", "D40"}

    # Ensure all entries in the 'objects' column are lists
    df['objects'] = df['objects'].apply(lambda x: list(x))

    # Function to filter rows based on labels
    def filter_labels(objects):
        return set(objects).issubset(include_labels)

    print("Filtering dataset based on specified labels...")
    # Apply the filter to the dataset
    filtered_df = df[df['objects'].apply(filter_labels)]

    # Create a new directory for the filtered data
    filtered_image_dir = os.path.join(data_root_dir, "filtered", "images")
    filtered_annot_dir = os.path.join(data_root_dir, "filtered", "annotations")

    os.makedirs(filtered_image_dir, exist_ok=True)
    os.makedirs(filtered_annot_dir, exist_ok=True)

    print("Copying filtered images and annotations to new directories...")
    # Copy the filtered images and annotations to the new directory
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying files"):
        xml_file_path = row['xml_file_path']
        image_file_name = os.path.basename(xml_file_path).replace(".xml", ".jpg")
        image_file_path = os.path.join(image_destination_path, image_file_name)
        
        # Ensure both the image and annotation files exist before copying
        if os.path.exists(image_file_path) and os.path.exists(xml_file_path):
            shutil.copy(image_file_path, filtered_image_dir)
            shutil.copy(xml_file_path, filtered_annot_dir)

    print(f"Filtered images and annotations have been saved to:")
    print(f"Images: {filtered_image_dir}")
    print(f"Annotations: {filtered_annot_dir}")

    # Verify the filtered data
    print("Starting verification of the filtered data...")
    filtered_annot_list = glob.glob(os.path.join(filtered_annot_dir, "*.xml"))

    # Function to verify annotations
    def verify_annotations(xml_files, include_labels):
        label_counts = {label: 0 for label in include_labels}
        total_files = len(xml_files)
        valid_files = 0

        for xml_file in tqdm(xml_files, desc="Verifying annotations"):
            annotations = read_and_parse_xml(xml_file)
            objects = encoded_obj_annotations(annotations)
            if set(objects).issubset(include_labels):
                valid_files += 1
                for obj in objects:
                    if obj in label_counts:
                        label_counts[obj] += 1
            else:
                print(f"Invalid labels found in {xml_file}")

        return label_counts, valid_files, total_files

    # Verify the filtered annotations and count occurrences
    label_counts, valid_files, total_files = verify_annotations(filtered_annot_list, include_labels)

    print(f"Total files processed: {total_files}")
    print(f"Valid files with specified labels: {valid_files}")
    print("Label counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()