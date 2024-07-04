# import necessary libraries
import pandas as pd
import os
import shutil
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Define the path to the meatadata and image directory
meta_data_dir = "../damage_assessment/Dataset_Info.csv"
image_dir = "../damage_assessment/images"

# Load the dataset information from the metadata
dataset_info = pd.read_csv(meta_data_dir)
# Drop the 'Unnamed: 4' column from the dataframe
dataset_info = dataset_info.drop(columns=['Unnamed: 4'])

# Add a new column 'Image_Path' to the DataFrame
# This column contains the relative path to each image file based on the 'Image ID' column
dataset_info['Image_Path'] = dataset_info['Image ID'].apply(lambda x: os.path.relpath(os.path.join(image_dir, x + '.jpg'), start=os.curdir))

# Convert the 'Level' column from categorical to numerical
level_mapping = {'C': 0, 'B': 1, 'A': 2, 'S': 3}
dataset_info['Severity_Level'] = dataset_info['Level'].map(level_mapping)

# Separate the data with 'C' level (now 0) from other levels
c_label_data = dataset_info[dataset_info['Severity_Level'] == 0]
other_labels_data = dataset_info[dataset_info['Severity_Level'] != 0]

# Download the'C' level data to have 3000 samples
c_label_data_downsampled = resample(c_label_data, 
                                    replace=False, 
                                    n_samples=3000, 
                                    random_state=42)

# Concatenate the downsampled 'C' level data with other levels data to create a balanced dataset
balanced_dataset = pd.concat([c_label_data_downsampled, other_labels_data])

# Split the balanced dataset into training and testing sets
train_data, test_data = train_test_split(balanced_dataset, test_size=0.2, random_state=42, stratify=balanced_dataset['Level'])

# Define paths to save the training and testing metadata
train_csv_path = "../damage_assessment/train_Dataset_Info.csv"
test_csv_path = "../damage_assessment/test_Dataset_Info.csv"

# Save the training and testing metadata to CSV files
train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
print(f"Train data saved to: {train_csv_path}")
print(f"Test data saved to: {test_csv_path}")