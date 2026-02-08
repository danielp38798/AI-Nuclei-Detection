import sys
import os
import shutil
import cv2
import uuid
import time
import threading
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QDialog, QCheckBox, QProgressBar
from PySide6.QtCore import Qt
from PySide6.QtCore import QObject, Signal
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFileDialog

import numpy as np
import matplotlib.pyplot as plt
import json
from cellpose import models
from manage_coco_datasets import merge_from_dir, split_from_file

def center_window(window):
    """
    Center the given QWidget window on the screen.
    """
    screen_geometry = QApplication.primaryScreen().availableGeometry()
    window_geometry = window.frameGeometry()

    x = screen_geometry.width() // 2 - window_geometry.width() // 2
    y = screen_geometry.height() // 2 - window_geometry.height() // 2

    window.move(x, y)



class ProgressWindow(QDialog):
    # Define a signal to update the progress bar
    progress_updated = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.process_complete = False
        self.progress_bar = QProgressBar(self)
        self.finish_button = None
        self.current_step_1 = QtWidgets.QLabel()  # Label to display the current image name/path
        self.current_step_2 = QtWidgets.QLabel()  # Label to display the current image name/path
        self.layout = QVBoxLayout()
        self.setWindowTitle("Progress Window")
        self.setGeometry(100, 100, 200, 100)

        #self.style = QtWidgets.QStyleFactory.create('Fusion')
        #QtWidgets.QApplication.setStyle(self.style)

        #layout = QVBoxLayout()
        self.layout.addWidget(self.current_step_1)
        self.layout.addWidget(self.current_step_2)
        self.layout.addWidget(self.progress_bar)
        # align the progress bar to the center
        self.progress_bar.setAlignment(Qt.AlignCenter)

        self.setLayout(self.layout)
        self.progress_updated.connect(self.update_progress)

    def process_complete(self):
        self.finish_button = QPushButton("Finish", self)
        self.finish_button.clicked.connect(self.finish_process) 
        #self.finish_button.setEnabled(False)  # Disabled initially
        self.layout.addWidget(self.finish_button)
        self.finish_button.setEnabled(True)

    def finish_process(self):
        # Actions to be performed when the process is complete
        print("Process is complete!")
        # show a message box if the process is completed
        #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
        # Close the progress window or perform any other actions
        self.accept()
    
    def update_progress(self, value):
        # Method to update the progress bar
        self.progress_bar.setValue(value)

class ImportImagesDialog(ProgressWindow):
     # define a signal to fetch wether the process is complete
    process_completed = QtCore.Signal(bool)
    def __init__(self, output_dir, total_images):
        super().__init__()
        self.output_dir = output_dir
        self.process_completed.connect(self.process_completed_handler)
        self.stop_process = False
        self.total_images = total_images
        self.init_ui()
        center_window(self)
    
    def get_total_images(self):
        return self.total_images

    def init_ui(self):
        self.setWindowTitle("Image Import")
        self.current_step_1.setText("Select Image Import to import images")
        self.convert_checkbox = QCheckBox("Convert images to 8-bit", checked=True)
        self.layout.addWidget(self.convert_checkbox)

        # add checkbox to select the type of images to import
        self.import_dapi_checkbox = QCheckBox("Import DAPI channel", checked=True)
        self.layout.addWidget(self.import_dapi_checkbox)

        # add checkbox to select the type of images to import
        self.import_fitc_checkbox = QCheckBox("Import FITC channel", checked=False)
        self.layout.addWidget(self.import_fitc_checkbox)

        self.import_button = QPushButton('Import Images', self)
        self.import_button.clicked.connect(self.import_images)
        self.layout.addWidget(self.import_button)

        self.cancel_button = QtWidgets.QPushButton('Cancel Process')
        self.cancel_button.clicked.connect(self.cancel_process)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)
        self.setWindowTitle('Import Images')

                   
    def get_directories(self):
        #dictionary to store the directories
        directories = {}
        directories['base_dir'] = self.base_dir
        directories['converted_output_dir'] = self.converted_output_dir
        return directories
    
    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.import_button.hide()
            self.cancel_button.hide()
            self.finish_button = QPushButton("Finish", self)
            self.finish_button.clicked.connect(self.finish_process) 
            #self.finish_button.setEnabled(False)  # Disabled initially
            self.layout.addWidget(self.finish_button)
            self.finish_button.show()
            self.finish_button.setEnabled(True)
    
    def run_import(self):
        print("Running import...")
        if self.process_completed:
            self.process_completed.emit(False)  # Reset the completion status

        self.annotation_thread = threading.Thread(target=self.import_images)
        self.annotation_thread.start()


    def import_images(self):
        try:
            # ask the user to select a folder
            self.base_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
            if not self.base_dir:
                #show message box that no folder was selected
                QtWidgets.QMessageBox.warning(self, "Warning", "No folder selected")
                
            #create a folder to store the output images
            self.converted_output_dir = self.output_dir #os.path.join(self.output_dir, "images") #os.path.join(self.base_dir, "images")
            
            if not os.path.exists(self.converted_output_dir ):
                os.makedirs(self.converted_output_dir, exist_ok=True)
            # store complete paths to the images in a list
            # walk over the files in the directory and check if the file name contains EDF_RAW_ch00 and not Merged
            self.images_file_paths = []
            for root, dirs, files in os.walk(self.base_dir):
                # also consider subdirectories
                for file in files:
                    if self.import_dapi_checkbox.isChecked():
                        if file.endswith(".tif") and "EDF_RAW_ch00" in file and not "Merged" in file:
                            self.images_file_paths.append(os.path.join(root, file))
                    if self.import_fitc_checkbox.isChecked():
                        if file.endswith(".tif") and "EDF_RAW_ch01" in file and not "Merged" in file:
                            self.images_file_paths.append(os.path.join(root, file))

            #self.images_file_paths = [file for file in os.listdir(root) if "EDF_RAW_ch00" in file and not "Merged" in file]
            print(f"Found {len(self.images_file_paths)} images in {root}")
            #print(f"Images found: {self.images_file_paths}")

            self.update_progress(0)
            #update the label with the current state
            self.current_step_1.setText("Importing images ...")
            for idx, image_file in enumerate(self.images_file_paths):
                #self.current_step_2.setText(f"Importing image {idx + 1} of {len(self.images_file_paths)}")
                # Update the progress bar using the signal
                progress_value = int((idx + 1) / len(self.images_file_paths) * 100)
                self.progress_updated.emit(progress_value)
                if self.stop_process:
                    break
                source_path = image_file #os.path.join(root, image_file)
                file_name, file_extension = os.path.splitext(image_file)
                random_filename = str(uuid.uuid4())[:20]
                new_filename = f"{random_filename}{file_extension}"
                destination_path = os.path.join(self.converted_output_dir, new_filename)

                if self.convert_checkbox:
                    # Read the 16-bit image
                    img_16bit = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
                    if img_16bit is None:
                        print(f"Error: Unable to read the image from {source_path}")
                        return
                    img_8bit = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    resolution_unit = 2  # 2 indicates inches
                    x_dpi = 72
                    y_dpi = 72
                    cv2.imwrite(destination_path, img_8bit, [
                        cv2.IMWRITE_TIFF_COMPRESSION, 1,
                        cv2.IMWRITE_TIFF_RESUNIT, resolution_unit,
                        cv2.IMWRITE_TIFF_XDPI, x_dpi,
                        cv2.IMWRITE_TIFF_YDPI, y_dpi
                    ])
    
                    #print(f"Conversion complete. 8-bit image saved at {destination_path}")
                else:
                    shutil.copyfile(source_path, destination_path)
                    print(f"Unconverted 8-bit image {source_path} copied to {destination_path}")
            
                #set progress bar to 100% after the process is complete
                self.update_progress(100)
            

            if not self.stop_process:
                self.process_completed.emit(True)  # Signal process completion
                            
                # update the label with the current state
                self.current_step_1.setText("Importing images finished")
                self.total_images += len(self.images_file_paths)
                self.current_step_2.setText(f"Imported {len(self.images_file_paths)} images. \n Total images imported: {self.total_images} ")
            
            # hide the run button
            #self.import_button.hide()
            #self.cancel_button.hide()
            # show the finish button
            #super().process_complete()  # Call the process_complete method in the parent class
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            self.process_completed.emit(False)  # Signal process failure

    def cancel_process(self):
        super().finish_process()  # Call the finish_process method in the parent class


class AnnotationDialog(ProgressWindow):
    # define a signal to fetch wether the process is complete
    process_completed = QtCore.Signal(bool)

    def __init__(self, imports_dir):
        super().__init__()
        self.imports_dir = imports_dir
        self.process_completed.connect(self.process_completed_handler)
        self.stop_process = False
        self.init_ui()
        center_window(self)

    def init_ui(self):
        self.setWindowTitle("Pre-Annotation Progress")

        self.run_annotation_button = QtWidgets.QPushButton('Run Pre-Annotation')
        self.run_annotation_button.clicked.connect(self.run_annotation)
        self.layout.addWidget(self.run_annotation_button)

        self.cancel_button = QtWidgets.QPushButton('Cancel Process')
        self.cancel_button.clicked.connect(self.cancel_process)
        self.layout.addWidget(self.cancel_button)

        #create a button to open the output folder
        self.open_output_folder_button = QtWidgets.QPushButton('Open Output Folder')
        self.open_output_folder_button.clicked.connect(self.open_annotation_output_folder)
        self.layout.addWidget(self.open_output_folder_button)
        # hide the open output folder button
        self.open_output_folder_button.hide()

        self.setLayout(self.layout)

    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.run_annotation_button.hide()
            self.cancel_button.hide()
            self.open_output_folder_button.show()
            self.finish_button = QPushButton("Finish", self)
            self.finish_button.clicked.connect(self.finish_process) 
            #self.finish_button.setEnabled(False)  # Disabled initially
            self.layout.addWidget(self.finish_button)
            self.finish_button.show()
            self.finish_button.setEnabled(True)


    def open_annotation_output_folder(self):
        # ask the user if they want to open the output folder
        open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if open_folder == QtWidgets.QMessageBox.Yes:
            os.startfile(self.annotations_output_dir)

    def cancel_process(self):
        self.stop_process = True
        # terminate the thread
        self.annotation_thread.join()

        #show message box that the process was cancelled
        QtWidgets.QMessageBox.warning(self, "Warning", "Process cancelled")
        super().finish_process()  # Call the finish_process method in the parent class

    def update_progress(self, value):
        # Method to update the progress bar
        self.progress_bar.setValue(value)
        #if value == 100:
            #self.finish_process()

    def check_imports(self):
        # store paths to the images and annotations folders in a dictionary
        self.imports = {}

        # walk thrugh the output directory and check if the images and annotations folder exist in the subdirectories
        for root, dirs, files in os.walk(self.imports_dir):
            print(f"Found {len(dirs)} subdirectories")
            # check if the images and annotations folders exist in the subdirectories
            # store the paths to the images and annotations folders on a sub dir basis in a dictionary
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations')):
                    self.imports[dir] = {}
                    self.imports[dir]['images'] = os.path.join(root, dir, 'images')
                    self.imports[dir]['annotations'] = os.path.join(root, dir, 'annotations')
                    print(f"Found images and annotations folder in {dir}")
                else:
                    print(f"Images and annotations folder not found in {dir}")
            
        # print the imports dictionary
        # throw a warning if no images and annotations folder was found
        if len(self.imports) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No images imported. Please import images first.")
            return
        

    def run_annotation(self):
        self.check_imports()
        print("Running annotation...")
        if self.process_completed:
            self.process_completed.emit(False)  # Reset the completion status

        self.annotation_thread = threading.Thread(target=self.binary_masks_to_coco_annotation)
        self.annotation_thread.start()


    def binary_masks_to_coco_annotation(self):
        try:
            # loop through the subdirectories in the imports directory
            for idx, (key, value) in enumerate(self.imports.items()):
                self.current_step_1.setText(f"Currently annotating import {idx +1} of {len(self.imports)}")
                self.image_dir = value['images']
                self.annotations_output_dir = value['annotations']
                print(f"Currently annotating images in directory: {self.image_dir}")
                print(f"Image directory: {self.image_dir }")
                print(f"Output directory: { self.annotations_output_dir }")
                if not os.path.exists(os.path.join(self.annotations_output_dir, "instances_default.json")):
                    os.makedirs(self.annotations_output_dir , exist_ok=True)
                coco_annotations = {
                    "info": {},
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "nucleus", "supercategory": "object"}],
                }
                image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
                model = models.Cellpose(gpu=True, model_type='nuclei')

                for i, image_file in enumerate(image_files):
                    if self.stop_process:
                        break

                    if i == 0:
                        #self.update_text("Pre-Annotation of images started ...")
                        self.current_step_2.setText("Pre-Annotation of images started ...")  # Update label with current image name

                    image_path = os.path.join(self.image_dir, image_file)
                    img = cv2.imread(image_path)
                    width, height = img.shape[1], img.shape[0]
                    if len(img.shape) < 3:
                        plt.imshow(img, cmap='gray')
                        plt.show()
                        masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0, 0])
                    elif len(img.shape) == 3:
                        masks, flows, styles, diams = model.eval(img, diameter=30, channels=[3, 0])
                    else:
                        print('Not a valid image format')
                    current_state = f"Processed image {i + 1} of {len(image_files)}"
                    self.current_step_2.setText(current_state)  # Update label with current image name
                    # Update the progress bar using the signal
                    progress_value = int((i + 1) / len(image_files) * 100)
                    self.progress_updated.emit(progress_value)

                    image_info = {
                        "id": i + 1,
                        "width": width,
                        "height": height,
                        "file_name": image_file,
                        "license": None,
                        "date_captured": None,
                    }
                    coco_annotations["images"].append(image_info)
                    annotations = []
                    overlay_image = img.copy()
                    if len(overlay_image.shape) == 2:
                        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)
                    for j in range(1, np.max(masks) + 1):
                        if self.stop_process:
                            return
                        mask = np.uint8(masks == j)
                        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 5)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            area = cv2.contourArea(contour)
                            contour_list = contour.reshape(-1).tolist()
                            if len(contour_list) < 6:
                                continue
                            annotation = {
                                "id": len(annotations) + 1,
                                "image_id": i + 1,
                                "category_id": 1,
                                "segmentation": [contour_list],
                                "area": area,
                                "bbox": [x, y, w, h],
                                "iscrowd": 0,
                            }
                            annotations.append(annotation)
                    coco_annotations["annotations"].extend(annotations)
                    self.update()
                    with open(os.path.join(self.annotations_output_dir , "instances_default.json"), "w") as json_file:
                        json.dump(coco_annotations, json_file)

            print("Annotations saved to {}".format( self.annotations_output_dir ))
            self.current_step_1.setText("Pre-Annotation of images finished for all imports")
            if not self.stop_process:
                self.process_completed.emit(True)  # Signal process completion
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            self.process_completed.emit(False)  # Signal process failure

class MergeDataDialog(ProgressWindow):
    # define a signal to fetch wether the process is complete
    process_completed = QtCore.Signal(bool)

    def __init__(self, imports_dir, output_dir):
        super().__init__()
        self.imports_dir = imports_dir
        self.output_dir = output_dir
        self.process_completed.connect(self.process_completed_handler)
        self.stop_process = False
        self.init_ui()
        center_window(self)

    def init_ui(self):
        self.setWindowTitle("Dataset Merging")

        self.run_merge_button = QtWidgets.QPushButton('Run Dataset Merging')
        self.run_merge_button.clicked.connect(self.run_merge_v1)
        self.layout.addWidget(self.run_merge_button)

        self.cancel_button = QtWidgets.QPushButton('Cancel Process')
        self.cancel_button.clicked.connect(self.cancel_process)
        self.layout.addWidget(self.cancel_button)

        #create a button to open the output folder
        self.open_output_folder_button = QtWidgets.QPushButton('Open Output Folder')
        self.open_output_folder_button.clicked.connect(self.open_merge_output_folder)
        self.layout.addWidget(self.open_output_folder_button)
        # hide the open output folder button
        self.open_output_folder_button.hide()

        self.setLayout(self.layout)

    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.run_merge_button.hide()
            self.cancel_button.hide()
            self.open_output_folder_button.show()
            self.finish_button = QPushButton("Finish", self)
            self.finish_button.clicked.connect(self.finish_process) 
            #self.finish_button.setEnabled(False)  # Disabled initially
            self.layout.addWidget(self.finish_button)
            self.finish_button.show()
            self.finish_button.setEnabled(True)

    def check_imports(self):
        # store paths to the images and annotations folders in a dictionary
        self.imports = {}

        # walk thrugh the output directory and check if the images and annotations folder exist in the subdirectories
        for root, dirs, files in os.walk(self.imports_dir):
            #print(f"Found {len(dirs)} subdirectories")
            # check if the images and annotations folders exist in the subdirectories
            # store the paths to the images and annotations folders on a sub dir basis in a dictionary
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations')):
                    self.imports[dir] = {}
                    self.imports[dir]['images'] = os.path.join(root, dir, 'images')
                    self.imports[dir]['annotations'] = os.path.join(root, dir, 'annotations')
                    #print(f"Found images and annotations folder in {dir}")
                #else:
                    #print(f"Images and annotations folder not found in {dir}")

        # also check the the output directory if there is any pre-annotated data
        for root, dirs, files in os.walk(self.output_dir):
        #print(f"Found {len(dirs)} subdirectories")
        # check if the images and annotations folders exist in the subdirectories
            # store the paths to the images and annotations folders on a sub dir basis in a dictionary
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations')):
                    self.imports[dir] = {}
                    self.imports[dir]['images'] = os.path.join(root, dir, 'images')
                    self.imports[dir]['annotations'] = os.path.join(root, dir, 'annotations')
                #print(f"Found images and annotations folder in pre-annotated")
            
        # print the imports dictionary
        # throw a warning if no images and annotations folder was found
        if len(self.imports) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No images or datasets imported. Please run import first.")
            return
        
    


    def open_merge_output_folder(self):
        # ask the user if they want to open the output folder
        open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if open_folder == QtWidgets.QMessageBox.Yes:
            os.startfile(self.merged_data_dir)

    def cancel_process(self):
        self.stop_process = True
        # terminate the thread
        self.merging_thread.join()

        #show message box that the process was cancelled
        QtWidgets.QMessageBox.warning(self, "Warning", "Process cancelled")
        super().finish_process()  # Call the finish_process method in the parent class


    def run_merge_v1(self):
        print("Running dataset merging...")
        if self.process_completed:
            self.process_completed.emit(False)  # Reset the completion status
        self.merging_thread = threading.Thread(target=self.merge_datasets_v1)
        self.merging_thread.start()

    def merge_datasets_v1(self):
        
        # check if the imports directory exists
        if not os.path.exists(self.imports_dir):
            print(f"Imports directory {self.imports_dir} does not exist")
            QtWidgets.QMessageBox.warning(self, "Warning", "No images imported. Please import images first to merge datasets.")
            return
        
        # then check if the imports directory contains subdirectories with images and annotations folders
        self.check_imports()
        print("Running dataset merging...")
        self.current_step_1.setText("Dataset merging started ...")
        # create a subdirectory for the merged data
        #self.merged_data_dir = os.path.join(self.imports_dir, "merged_data")
        self.merged_data_dir = os.path.join(self.output_dir, "merged_data", time.strftime("%Y%m%d-%H%M%S"))
        # Create the output folder if it doesn't exist
        if not os.path.exists(os.path.join(self.merged_data_dir, 'images')):
            os.makedirs(os.path.join(self.merged_data_dir, 'images'))
        merged_annotations = {"licenses": [], "info": {}, "categories": [], "images": [], "annotations": []}
        image_id_mapping = {}
        # loop through the subdirectories in the imports directory
        
        # only consider imports of self.imports which contain an annotations folder with instances_default.json in it and an images folder with images
        # create a dictionary to store the paths to the images and annotations folders
        imports_with_annotations = {}
        
        for idx, (key, value) in enumerate(self.imports.items()):
            annotations_path = os.path.join(value['annotations'], 'instances_default.json')
            images_path = value['images']
            # only update the dictionary if the annotations and images folder exist and contain a instances_default.json file; images must contain image files
            if os.path.exists(annotations_path) and os.path.exists(images_path):
                imports_with_annotations[key] = value

        print(f"Found {len(imports_with_annotations)} imports with annotations")
        print(f"Imports with annotations: {imports_with_annotations}"   )

        self.update_progress(0)

        # if only one import with annotations was found, copy the images and annotations folder to the merged_data_dir
        if len(imports_with_annotations) == 1:
            self.current_step_1.setText(" ")
            # copy the images and annotations folder of the first item to the merged_data_dir
            # select the first item in the dictionary
            imports_with_annotations_list = list(imports_with_annotations.items())
            #print(f"imports_with_annotations: {imports_with_annotations}"   )
            # extract the key of the first item
            first_item_key = imports_with_annotations_list[0][0]
            shutil.copytree(os.path.join(imports_with_annotations[first_item_key]['annotations']), os.path.join(self.merged_data_dir, 'annotations'), dirs_exist_ok=True)
            shutil.copytree(imports_with_annotations[first_item_key]['images'], os.path.join(self.merged_data_dir, 'images'), dirs_exist_ok=True)
        else: # if more than one import with annotations was found, merge the datasets
            self.current_step_1.setText(" ")
            for idx, (key, value) in enumerate(imports_with_annotations.items()):
                self.update_progress(int((idx + 1) / len(imports_with_annotations) * 100))  
                self.current_step_1.setText(f"Merging imports ...")
                self.current_step_2.setText(f"Merging import {idx + 1} of {len(imports_with_annotations)}")
                annotations_path = os.path.join(value['annotations'], 'instances_default.json')
                images_path = value['images']
                #annotations_path = os.path.join(folder, 'annotations', 'instances_default.json')
                #images_path = os.path.join(folder, 'images')
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r') as file:
                        data = json.load(file)
                        # Merge licenses, info, and categories
                        merged_annotations["licenses"].extend(data.get("licenses", []))
                        merged_annotations["info"].update(data.get("info", {}))
                        # Add categories
                        if idx == 0:
                            merged_annotations["categories"].extend(data.get("categories", []))
                        # Update image IDs for the current dataset
                        image_offset = len(merged_annotations["images"])
                        for image in data.get("images", []):
                            old_image_id = image["id"]
                            new_image_id = old_image_id + image_offset
                            # Update image IDs in annotations
                            for annotation in data.get("annotations", []):
                                if annotation["image_id"] == old_image_id:
                                    annotation["image_id"] = new_image_id
                            # Store the mapping of old to new image IDs
                            image_id_mapping[old_image_id] = new_image_id
                            # Update the image ID
                            image["id"] = new_image_id
                        merged_annotations["images"].extend(data.get("images", []))
                        # Update annotation image IDs for the current dataset
                        annotation_offset = len(merged_annotations["annotations"])
                        for annotation in data.get("annotations", []):
                            annotation["id"] += annotation_offset
                        merged_annotations["annotations"].extend(data.get("annotations", []))
                    # Move images to the output folder
                    for file_name in os.listdir(images_path):
                        file_path = os.path.join(images_path, file_name)
                        shutil.copy(file_path, os.path.join(self.merged_data_dir, 'images'))
            # Save the merged annotations to the output folder
            output_annotations_path = os.path.join(self.merged_data_dir, 'annotations', 'instances_default.json')
            # create output file
            if not os.path.exists(os.path.dirname(output_annotations_path)):
                os.makedirs(os.path.dirname(output_annotations_path))
            with open(output_annotations_path, 'w') as output_file:
                json.dump(merged_annotations, output_file)

     
        self.process_completed.emit(True) # Signal process completion
        self.current_step_1.setText("Merging datasets finished")
        self.update_progress(100)


    def run_merge(self):
        print("Running dataset merging...")
        self.merging_thread = threading.Thread(target=self.merge_datasets)
        self.merging_thread.start()

    def merge_datasets(self):
        """
        Merge datasets from the selected directories
        """
        root_dir = self.imports_dir
        # Create the output folder if it doesn't exist
        if not os.path.exists(os.path.join(self.output_dir, 'merged_data')):
            os.makedirs(os.path.join(self.output_dir, 'merged_data'))
        # Merge the datasets
        since = time.time()
        merge_from_dir(root_dir, self.output_dir)
        time_elapsed = time.time() - since
        print(f'Merging complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')




class NewMergeDataDialog(QDialog):
    process_completed = QtCore.Signal(int)
    def __init__(self, imports_dir, output_dir):
        super().__init__()
        self.imports_dir = imports_dir
        self.output_dir = output_dir
        self.process_completed.connect(self.process_completed_handler)
        self.init_ui()
        center_window(self)

    def init_ui(self):
        self.setWindowTitle("Select Data Folders to Merge")
        self.setGeometry(100, 100, 400, 200)
        #self.style = QtWidgets.QStyleFactory.create('Fusion')
        #QtWidgets.QApplication.setStyle(self.style)
        self.create_widgets()
        self.create_layout()
    def create_widgets(self):
        self.input_dir_label = QtWidgets.QLabel(self, text="Select Input Directory:")
        self.input_dir_label_var = QtWidgets.QLineEdit(self)
        self.input_dir_button = QtWidgets.QPushButton(self, text="Browse", clicked=self.select_input_dir)
        self.output_dir_label = QtWidgets.QLabel(self, text="Select Output Directory:")
        self.output_dir_label_var = QtWidgets.QLineEdit(self)
        self.output_dir_button = QtWidgets.QPushButton(self, text="Browse", clicked=self.select_output_dir)
        #self.progress_bar = QtWidgets.QProgressBar(self)
        self.merging_status_label = QtWidgets.QLabel(self, text="Merging Status:")
        self.merge_button = QtWidgets.QPushButton(self, text="Merge Datasets", clicked=self.merge_datasets)
        self.cancel_button = QtWidgets.QPushButton(self, text="Cancel Process", clicked=self.cancel_process)
        self.finish_button = QPushButton("Finish", self)
        self.finish_button.clicked.connect(self.finish_process)

    def create_layout(self):
        layout = QtWidgets.QVBoxLayout(self)
        input_dir_layout = QtWidgets.QHBoxLayout(self)
        input_dir_layout.addWidget(self.input_dir_label)
        input_dir_layout.addWidget(self.input_dir_label_var)
        input_dir_layout.addWidget(self.input_dir_button)
        output_dir_layout = QtWidgets.QHBoxLayout(self)
        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(self.output_dir_label_var)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addLayout(input_dir_layout)
        layout.addLayout(output_dir_layout)
        #layout.addWidget(self.progress_bar)
        layout.addWidget(self.merging_status_label)
        layout.addWidget(self.merge_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.finish_button)

        self.setLayout(layout)

    def cancel_process(self):
        self.stop_process = True
        # terminate the thread
        if self.merging_thread:
            self.merging_thread.join()
        #show message box that the process was cancelled
        QtWidgets.QMessageBox.warning(self, "Warning", "Process cancelled")
        self.finish_process()

    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.merge_button.hide()
            self.cancel_button.hide()
            self.finish_button.show()
            self.finish_button.setEnabled(True)
    def finish_process(self):
        # Actions to be performed when the process is complete
        print("Process is complete!")
        # show a message box if the process is completed
        #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
        # Close the progress window or perform any other actions
        self.accept()
    def update_progress(self, value):
        # Method to update the progress bar
        self.progress_bar.setValue(value)
        #if value == 100:
            #self.finish_process()
        
    def select_input_dir(self):
        self.input_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if self.input_dir:
            print("Selected Input Directory:", self.input_dir)
        
        # update the label with the selected directory
        self.input_dir_label_var.setText(self.input_dir)
    
    def select_output_dir(self):
        self.output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_dir:
            print("Selected Output Directory:", self.output_dir)
        # update the label with the selected directory
        self.output_dir_label_var.setText(self.output_dir)

    def merge_datasets(self):
        """
        Merge datasets from the selected directories
        """
        self.merging_status_label.setText("Merging datasets ...")
        # Create the output folder if it doesn't exist
        merged_data_dir = os.path.join(self.output_dir, 'merged_data')
        if not os.path.exists(merged_data_dir):
            os.makedirs(merged_data_dir)
        # Merge the datasets
        self.merging_thread = threading.Thread(target=self.run_merge(self.input_dir, merged_data_dir))
        self.merging_thread.start()
        self.merging_status_label.setText("Merging complete")
        self.process_completed.emit(True) # Signal process completion
        #self.update_progress(100)
        # ask the user if they want to open the output folder
        open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if open_folder == QtWidgets.QMessageBox.Yes:
            os.startfile(merged_data_dir)



    def run_merge(self, root_dir, output_dir):
        # Merge the datasets
        since = time.time()
        merge_from_dir(root_dir, output_dir)
        time_elapsed = time.time() - since
        print(f'Merging complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')    
        










class MergeDataDialogManual(QtWidgets.QDialog):
    process_completed = QtCore.Signal(int)
    def __init__(self, imports_dir, output_dir):
        super().__init__()
        self.imports_dir = imports_dir
        self.output_dir = output_dir
        self.process_completed.connect(self.process_completed_handler)
        self.init_ui()
        center_window(self)

    def init_ui(self):
        self.setWindowTitle("Select Data Folders to Merge")
        self.setGeometry(100, 100, 200, 100)
        #self.style = QtWidgets.QStyleFactory.create('Fusion')
        #QtWidgets.QApplication.setStyle(self.style)
        self.create_widgets()
        self.create_layout()
    def create_widgets(self):
        self.input_dir1_label = QtWidgets.QLabel(self, text="Select Input Directory 1:")
        self.input_dir1_label_var = QtWidgets.QLineEdit(self)
        self.input_dir1_button = QtWidgets.QPushButton(self, text="Browse", clicked=self.select_input_dir1)
        self.input_dir2_label = QtWidgets.QLabel(self, text="Select Input Directory 2:")
        self.input_dir2_label_var = QtWidgets.QLineEdit(self)
        self.input_dir2_button = QtWidgets.QPushButton(self, text="Browse", clicked=self.select_input_dir2)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.merge_button = QtWidgets.QPushButton(self, text="Merge Datasets", clicked=self.merge_data)
        self.cancel_button = QtWidgets.QPushButton(self, text="Cancel Process", clicked=self.cancel_process)
        self.finish_button = QPushButton("Finish", self)
        self.finish_button.clicked.connect(self.finish_process) 
        #self.finish_button.setEnabled(False)  # Disabled initially
        self.finish_button.hide()

    def create_layout(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.input_dir1_label)
        layout.addWidget(self.input_dir1_label_var)
        layout.addWidget(self.input_dir1_button)
        layout.addWidget(self.input_dir2_label)
        layout.addWidget(self.input_dir2_label_var)
        layout.addWidget(self.input_dir2_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.merge_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.finish_button)
    def cancel_process(self):
        self.stop_process = True
        # terminate the thread
        if self.merging_thread:
            self.merging_thread.join()
        #show message box that the process was cancelled
        QtWidgets.QMessageBox.warning(self, "Warning", "Process cancelled")
        self.finish_process()    

    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.merge_button.hide()
            self.cancel_button.hide()
            self.finish_button.show()
            self.finish_button.setEnabled(True)
    def finish_process(self):
        # Actions to be performed when the process is complete
        print("Process is complete!")
        # show a message box if the process is completed
        #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
        # Close the progress window or perform any other actions
        self.accept()
    def update_progress(self, value):
        # Method to update the progress bar
        self.progress_bar.setValue(value)
        #if value == 100:
            #self.finish_process()

    def select_input_dir1(self):
        self.input_dir1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory 1", self.output_dir)
        if self.input_dir1:
            print("Selected Input Directory 1:", self.input_dir1)
        
        # update the label with the selected directory
        self.input_dir1_label_var.setText(self.input_dir1)

    def select_input_dir2(self):
        self.input_dir2 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory 2", self.output_dir)
        if self.input_dir2:
            print("Selected Input Directory 2:", self.input_dir2)
        # update the label with the selected directory
        self.input_dir2_label_var.setText(self.input_dir2)


    def merge_data(self):

        # show a message box if no input folder is selected
        if self.input_dir1 == "" or self.input_dir1 == "":
            QtWidgets.QMessageBox.warning(self, "Error", "No input folder selected")
            return

        #throw an error message if the selcted dir does not contain a annotations and images folder
        self.input_folders = [self.input_dir1, self.input_dir2]
        for folder in self.input_folders:
            if not os.path.exists(os.path.join(folder, 'annotations', 'instances_default.json')):
                QtWidgets.QMessageBox.warning(self, "Error", f"No annotations file found in the selected folder {folder}")
                return
            if not os.path.exists(os.path.join(folder, 'images')):
                QtWidgets.QMessageBox.warning(self, "Error", "No images folder found in the selected folder")
                return  

        try:
            # Call merge_datasets function with the selected directories
            self.merged_data_dir = os.path.join(self.output_dir, "merged_data", time.strftime("%Y%m%d-%H%M%S"))
            self.merging_thread = threading.Thread(target=self.merge_datasets(self.input_folders, self.merged_data_dir))
            open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if open_folder == QtWidgets.QMessageBox.Yes:
                os.startfile(self.merged_data_dir)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    # functions for the dataset merging process
    def merge_datasets(self, input_folders, output_folder):
        # Create the output folder if it doesn't exist
        if not os.path.exists(os.path.join(output_folder, 'images')):
            os.makedirs(os.path.join(output_folder, 'images'))
        merged_annotations = {"licenses": [], "info": {}, "categories": [], "images": [], "annotations": []}
        image_id_mapping = {}
        #set progress bar to 0%
        self.update_progress(0)
        for idx, folder in enumerate(input_folders):
            annotations_path = os.path.join(folder, 'annotations', 'instances_default.json')
            images_path = os.path.join(folder, 'images')
            # update the progress bar
            self.update_progress(int((idx + 1) / len(input_folders) * 100))
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as file:
                    data = json.load(file)
                    # Merge licenses, info, and categories
                    merged_annotations["licenses"].extend(data.get("licenses", []))
                    merged_annotations["info"].update(data.get("info", {}))
                    # Add categories
                    if idx == 0:
                        merged_annotations["categories"].extend(data.get("categories", []))
                    # Update image IDs for the current dataset
                    image_offset = len(merged_annotations["images"])
                    for image in data.get("images", []):
                        old_image_id = image["id"]
                        new_image_id = old_image_id + image_offset
                        # Update image IDs in annotations
                        for annotation in data.get("annotations", []):
                            if annotation["image_id"] == old_image_id:
                                annotation["image_id"] = new_image_id
                        # Store the mapping of old to new image IDs
                        image_id_mapping[old_image_id] = new_image_id
                        # Update the image ID
                        image["id"] = new_image_id
                    merged_annotations["images"].extend(data.get("images", []))
                    # Update annotation image IDs for the current dataset
                    annotation_offset = len(merged_annotations["annotations"])
                    for annotation in data.get("annotations", []):
                        annotation["id"] += annotation_offset
                    merged_annotations["annotations"].extend(data.get("annotations", []))
                # Move images to the output folder
                for file_name in os.listdir(images_path):
                    file_path = os.path.join(images_path, file_name)
                    shutil.copy(file_path, os.path.join(output_folder, 'images'))
        # Save the merged annotations to the output folder
        output_annotations_path = os.path.join(output_folder, 'annotations', 'instances_default.json')
        # create output file
        if not os.path.exists(os.path.dirname(output_annotations_path)):
            os.makedirs(os.path.dirname(output_annotations_path))
        with open(output_annotations_path, 'w') as output_file:
            json.dump(merged_annotations, output_file)
        # set progress bar to 100%
        self.update_progress(100)
        self.process_completed.emit(True) # Signal process completion


class SplitDataDialog(QtWidgets.QDialog):
    process_completed = QtCore.Signal(int)
    def __init__(self):
        super().__init__()

        self.process_completed.connect(self.process_completed_handler)
        self.init_ui()
        center_window(self)

    def init_ui(self):
        self.setWindowTitle("Select Data Folder to Split")
        self.setGeometry(100, 100, 200, 100)
        #self.style = QtWidgets.QStyleFactory.create('Fusion')
        #QtWidgets.QApplication.setStyle(self.style)
        self.create_widgets()
        self.create_layout()
    def create_widgets(self):
        self.input_file_label = QtWidgets.QLabel(self, text="Select JSON File:")
        self.input_file_label_var = QtWidgets.QLineEdit(self)
        self.import_file_button = QtWidgets.QPushButton(self, text="Browse", clicked=self.select_json_file)
        self.splitting_status_label = QtWidgets.QLabel(self, text="Splitting Status:")

        self.split_button = QtWidgets.QPushButton(self, text="Split Dataset", clicked=self.split_data)
        self.cancel_button = QtWidgets.QPushButton(self, text="Cancel Process", clicked=self.cancel_process)
        self.finish_button = QPushButton("Finish", self)
        self.finish_button.clicked.connect(self.finish_process) 
        #self.finish_button.setEnabled(False)  # Disabled initially
        self.finish_button.hide()

    def create_layout(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.input_file_label)
        layout.addWidget(self.input_file_label_var)
        layout.addWidget(self.import_file_button)
        layout.addWidget(self.splitting_status_label)
        layout.addWidget(self.split_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.finish_button)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.setLayout(layout)
    def cancel_process(self):
        self.stop_process = True
        # terminate the thread
        if self.splitting_thread:
            self.splitting_thread.join()
        #show message box that the process was cancelled
        QtWidgets.QMessageBox.warning(self, "Warning", "Process cancelled")
        self.finish_process()    

    def process_completed_handler(self, completed):
        if completed:
            #QtWidgets.QMessageBox.information(self, "Process Complete", "Process is complete!")
            self.split_button.hide()
            self.cancel_button.hide()
            self.finish_button.show()
            self.finish_button.setEnabled(True)
    def finish_process(self):
        # Actions to be performed when the process is complete
        print("Process is complete!")
        # show a message box if the process is completed

    def select_json_file(self):
        self.input_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select JSON File", filter="JSON Files (*.json)")[0]
        if self.input_file:
            print("Selected JSON File:", self.input_file)
            self.input_file_label_var.setText(self.input_file)

    
    def split_data(self):
        # show a message box if no input folder is selected
        if self.input_file == "":
            QtWidgets.QMessageBox.warning(self, "Error", "No input folder selected")
            return
        
        # ask for the number of images per split
        images_per_split, ok = QtWidgets.QInputDialog.getInt(self, "Images per Split", "Enter the number of images per split:", 10, 1, 100, 1)

        # ask for the output directory
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")

        if images_per_split:
            print(f"Images per split: {images_per_split}")
            self.splitting_status_label.setText("Splitting dataset ...")
            self.splitting_thread = threading.Thread(target=self.split_datasets(self.input_file, images_per_split, out_dir))
            self.splitting_thread.start()
            self.splitting_status_label.setText("Splitting dataset finished")
            self.process_completed.emit(True)
            #self.update_progress(100)
            # ask the user if they want to open the output folder
            open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if open_folder == QtWidgets.QMessageBox.Yes:
                os.startfile(out_dir)
        else:
            return

        
    def split_datasets(self, input_folder, images_per_split, out_dir):
        
        split_from_file(cocojson=self.input_file, images_per_split=images_per_split, ratios=None, names=None, do_shuffle=False, output_dir=out_dir)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        center_window(self)
        self.setWindowTitle("Nuclei Pre-Annotation")

    def init_ui(self):

        self.output_dir = os.path.join(os.getcwd(), "trainer_toolbox", "output")
        self.imports_dir = os.path.join(os.getcwd(), "trainer_toolbox", "imports")
        self.imports = None

        # create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # create the imports directory if it doesn't exist
        if not os.path.exists(self.imports_dir):
            os.makedirs(self.imports_dir, exist_ok=True)
        
        layout = QVBoxLayout()
        # set the size of the window
        self.setWindowTitle("Nuclei Pre-Annotation")
        self.setGeometry(100, 100, 250, 150)
        #self.style = QtWidgets.QStyleFactory.create('Fusion')
        #QtWidgets.QApplication.setStyle(self.style)

        # create a label to display the current state
        self.current_state_label = QtWidgets.QLabel(self, text="Current State:")
        layout.addWidget(self.current_state_label)
        self.current_state_label.setText(f"No images data existant yet. \n Please import images first.")
        image_importer = QPushButton('Image Import', self)
        self.import_window_opened = 0
        image_importer.clicked.connect(self.open_image_importer)
        layout.addWidget(image_importer)

        nuclei_annotator_button = QPushButton('Nuclei Pre-Annotation', self)
        nuclei_annotator_button.clicked.connect(self.open_nuclei_annotator)
        layout.addWidget(nuclei_annotator_button)

        dataset_importer = QPushButton('COCO Dataset Import', self)
        dataset_importer.clicked.connect(self.open_dataset_importer)
        layout.addWidget(dataset_importer)

        dataset_merger_button = QPushButton('Dataset Merging', self)
        dataset_merger_button.clicked.connect(self.open_dataset_merger)
        layout.addWidget(dataset_merger_button)

        dataset_splitter_button = QPushButton('Dataset Splitting', self)
        dataset_splitter_button.clicked.connect(self.open_dataset_splitter)
        layout.addWidget(dataset_splitter_button)

        image_converter_button = QPushButton('Image Converter', self)
        image_converter_button.clicked.connect(self.open_image_converter)
        layout.addWidget(image_converter_button)

        self.export_button = QPushButton("Export Data", self)
        self.export_button.clicked.connect(self.export_data)
        layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear Data", self)
        self.clear_button.clicked.connect(self.clear_data)
        layout.addWidget(self.clear_button)

        self.setLayout(layout)
        self.setWindowTitle('Main Window')
        self.show()

    def open_dataset_splitter(self):
        # if the user clicks yes, split all data
        self.process_dialog = SplitDataDialog()
        self.process_dialog.exec_()


    def open_image_importer(self):
        if self.import_window_opened == 0:
            self.total_images = 0
             # update the paths
            self.update_paths()
            self.input_dialog = ImportImagesDialog(self.images_output_dir, self.total_images)
            self.input_dialog.exec_()
            self.total_images = self.input_dialog.get_total_images()
            self.current_state_label.setText(f"Current State: {self.total_images} images imported")
            self.import_window_opened += 1
        else:
            self.total_images = self.input_dialog.get_total_images()
            self.input_dialog = ImportImagesDialog(self.images_output_dir, self.total_images)
            self.input_dialog.exec_()
            self.total_images = self.input_dialog.get_total_images()
            self.current_state_label.setText(f"Current State: {self.total_images} images imported")
            self.import_window_opened += 1

    def open_nuclei_annotator(self):
        # Check if ImportImagesDialog is instantiated and executed
        #if hasattr(self, 'input_dialog'):
            # Get directories from the ImportImagesDialog
            #directories = self.input_dialog.get_directories()
            #self.base_dir = directories['base_dir']
            #self.image_dir = directories['converted_output_dir']
            # Pass the directories to the AnnotationDialog instance
            #self.process_dialog = AnnotationDialog(self.output_dir)
            #self.process_dialog.exec_()
        #else:
            #QtWidgets.QMessageBox.warning(self, "Warning", "Please import images first.")
        # check if the output directory contains images and annotations folders
        self.check_for_annotated_data()
        if len(self.annotated_data) > 0:
            # if annotated data was found ask the user if they really want to perform pre-annotation
            pre_annotation_question = QtWidgets.QMessageBox(self)
            pre_annotation_question.setIcon(QtWidgets.QMessageBox.Question)
            pre_annotation_question.setWindowTitle("Pre-Annotation")
            pre_annotation_question.setText("Pre-Annotation files detected. \n Do you really want to perform pre-annotation again? \n This will overwrite the existing pre-annotation files.")
            # Add custom buttons with text
            pre_annotation_question.addButton("Yes", QtWidgets.QMessageBox.YesRole)
            pre_annotation_question.addButton("No", QtWidgets.QMessageBox.NoRole)
            result = pre_annotation_question.exec_()
            if result == 0:
                # if the user clicks yes, perform pre-annotation
                self.process_dialog = AnnotationDialog(self.output_dir)
                self.process_dialog.exec_()
                self.current_state_label.setText(f"Current State: Pre-Annotation finished")
            else:
                # if the user clicks no, return
                return
            return
        else:
            self.process_dialog = AnnotationDialog(self.output_dir)
            self.process_dialog.exec_()
            self.current_state_label.setText(f"Current State: Pre-Annotation finished")
        
    def open_dataset_importer(self):

        # allow the user to select one directory, next copy all sub dirs to the output dir
        imports_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Imports Directory")
        if imports_dir:
            print("Selected Imports Directory:", imports_dir)
            self.imports = {}
            # walk thrugh the output directory and check if the images and annotations folder exist in the subdirectories
            for root, dirs, files in os.walk(imports_dir):
                print(f"Found {len(dirs)} subdirectories")
                # check if the images and annotations folders exist in the subdirectories
                # store the paths to the images and annotations folders on a sub dir basis in a dictionary
                for dir in dirs:
                    if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations')):
                        self.imports[dir] = {}
                        self.imports[dir]['images'] = os.path.join(root, dir, 'images')
                        self.imports[dir]['annotations'] = os.path.join(root, dir, 'annotations')
                        print(f"Found images and annotations folder in {dir}")
                        # copy the complete subdirectory to the output directory
                        self.copy_directory(os.path.join(root, dir), os.path.join(self.imports_dir, dir))
                    else:
                        print(f"Images and annotations folder not found in {dir}")
            # print the imports dictionary
            print(self.imports)
            # throw a warning if no images and annotations folder was found
            if len(self.imports) == 0:
                QtWidgets.QMessageBox.warning(self, "Warning", "No datasets imported.")
                return
            else:
                QtWidgets.QMessageBox.information(self, "Info", "Datasets imported successfully")
                self.current_state_label.setText(f"Current State: {len(self.imports)} imports found")

    def copy_directory(self, src, dest):
        # copy the contents of the src directory to the dest directory
        try:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        except Exception as e:
            print(f"An error occurred while trying to import datasets: {str(e)}")

    def open_dataset_merger(self):
        # if the user clicks yes, merge all data
        self.process_dialog = NewMergeDataDialog(self.imports_dir, self.output_dir)
        self.process_dialog.exec_()

    def open_image_converter(self): 

        #ask the user to select a directory containing images to convert
        self.input_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory", self.output_dir)
        if self.input_dir:
            print("Selected Input Directory:", self.input_dir)
            self.convert_images(self.input_dir)
        else:
            return
    
    def convert_images(self, input_folder):
        from PIL import Image
        import os

        # Input and output folder paths
        output_folder = os.path.join(input_folder, "converted_images")

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get a list of all TIFF files in the input folder
        tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]

        # Loop through each TIFF file and convert it to PNG
        for tiff_file in tiff_files:
            tiff_path = os.path.join(input_folder, tiff_file)
            output_file = os.path.splitext(tiff_file)[0] + ".png"
            output_path = os.path.join(output_folder, output_file)

            try:
                # Open the TIFF file
                with Image.open(tiff_path) as img:
                    # Save it as a PNG file
                    img.save(output_path, "PNG")
                    print(f"Converted {tiff_file} to {output_file}")
            except Exception as e:
                print(f"Failed to convert {tiff_file}: {str(e)}")

        print("Conversion completed.")


    def update_paths(self):
        # create a subdirectory for the current run
        self.run_subdir = os.path.join(self.output_dir, time.strftime("%Y%m%d-%H%M%S"))
        self.images_output_dir = os.path.join(self.run_subdir, "images")
        self.annotations_output_dir = os.path.join(self.run_subdir, "annotations")
        # create the directories if they don't exist
        if not os.path.exists(self.images_output_dir):
            os.makedirs(self.images_output_dir, exist_ok=True)
        if not os.path.exists(self.annotations_output_dir):
            os.makedirs(self.annotations_output_dir, exist_ok=True)

    def clear_data(self, ask_question=True, warn=True):
        # check if the output directory contains images and annotations folders
        self.check_for_annotated_data()
        if len(self.annotated_data) == 0 and warn:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data existant. Deletion cancelled.")
            return
        else:
            if ask_question:
                # ask the user if they really want to clear the data
                clear_data_question = QtWidgets.QMessageBox(self)
                clear_data_question.setIcon(QtWidgets.QMessageBox.Question)
                clear_data_question.setWindowTitle("Clear Data")
                clear_data_question.setText("Do you really want to clear the data? \n This will delete all images and annotations.")
                # Add custom buttons with text
                clear_data_question.addButton("Yes", QtWidgets.QMessageBox.YesRole)
                clear_data_question.addButton("No", QtWidgets.QMessageBox.NoRole)
                result = clear_data_question.exec_()
                if result == 0:
                    # if the user clicks yes, clear the data
                    # delete data
                    shutil.rmtree(self.imports_dir) 
                    shutil.rmtree(self.output_dir)
                    # show a message box that the data was cleared
                    QtWidgets.QMessageBox.information(self, "Clear Data", "Data cleared successfully")
                    self.current_state_label.setText(f"Current State: No data existant")
                else:
                    # if the user clicks no, return
                    QtWidgets.QMessageBox.information(self, "Clear Data", "Data deletion cancelled")
                    return
            else:
                # delete data 
                shutil.rmtree(self.imports_dir) 
                shutil.rmtree(self.output_dir)
                # show a message box that the data was cleared
                #QtWidgets.QMessageBox.information(self, "Clear Data", "Data cleared successfully")
                self.current_state_label.setText(f"Current State: No data existant")


    def check_for_annotated_data(self):
        # store paths to the images and annotations folders in a dictionary
        self.annotated_data = {}
        # walk thrugh the output directory and check if the images and annotations folder exist in the subdirectories
        for root, dirs, files in os.walk(self.output_dir):
            #print(f"Found {len(dirs)} subdirectories")
            # check if the images and annotations folders exist in the subdirectories
            # store the paths to the images and annotations folders on a sub dir basis in a dictionary
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations', 'instances_default.json')):
                    self.annotated_data[dir] = {}
                    self.annotated_data[dir]['dataset'] = os.path.join(root, dir)
                    self.annotated_data[dir]['images'] = os.path.join(root, dir, 'images')
                    self.annotated_data[dir]['annotations'] = os.path.join(root, dir, 'annotations', 'instances_default.json')
                    #print(f"Found images and annotations folder in {dir}")
                #else:
                    #print(f"Images and annotations folder not found in {dir}")
        
        # also check the imports directory for annotated data
        for root, dirs, files in os.walk(self.imports_dir):
            # check if the images and annotations folders exist in the subdirectories
            # store the paths to the images and annotations folders on a sub dir basis in a dictionary
            for dir in dirs:
                if os.path.exists(os.path.join(root, dir, 'images')) and os.path.exists(os.path.join(root, dir, 'annotations', 'instances_default.json')):
                    self.annotated_data[dir] = {}
                    self.annotated_data[dir]['dataset'] = os.path.join(root, dir)
                    self.annotated_data[dir]['images'] = os.path.join(root, dir, 'images')
                    self.annotated_data[dir]['annotations'] = os.path.join(root, dir, 'annotations', 'instances_default.json')
                    #print(f"Found images and annotations folder in {dir}")
                #else:
                    #print(f"Images and annotations folder not found in {dir}")


        print(f"Found {len(self.annotated_data)} imports with annotations")
        print(f"Imports with annotations: {self.annotated_data}"  )

        return self.annotated_data
        
    def export_data(self, manual_selection):
        """
        creates an export button once merging or pre-annotation is finished
        it lets the user decide if they want to export the merged data (if merging was performed) or selected data (if pre-annotation was performed)
        
        """
        # check the self.output_dir for annotated data
        self.annotated_data = self.check_for_annotated_data()
        # if no annotated data was found, show a message box
        if len(self.annotated_data) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No annotated data found. Please import and annotate data first.")
            return

        # if more than one annotated data was found, show a message box asking the user if they want to export all data or selected data
        if len(self.annotated_data) > 1:
            # show a message box asking the user if they want to export all data or selected data
            
            export_data = QtWidgets.QMessageBox(self)
            export_data.setIcon(QtWidgets.QMessageBox.Question)
            export_data.setWindowTitle("Export Data")
            export_data.setText("Do you want to export all data or selected data?")

            # Add custom buttons with text
            export_data.addButton("Export All", QtWidgets.QMessageBox.YesRole)
            export_data.addButton("Export Selected", QtWidgets.QMessageBox.NoRole)
            result = export_data.exec_()
            if export_data.clickedButton().text() == "Export All":
                # export all data
                self.export_all_data()
            else:
                # export selected data
                self.export_selected_data()
        else:
            # export all data
            self.export_all_data()

    def export_all_data(self):
        # export all data
        default_export_dir = self.output_dir
        # ask the user to select an output directory
        self.export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", default_export_dir)
        # if no output directory was selected, show a message box
        if self.export_dir == "":
            QtWidgets.QMessageBox.warning(self, "Warning", "No output directory selected")
            return
        # if an output directory was selected, export the data
        else:
            # copy all data stored in self.output_dir to the selected output directory
            shutil.copytree(self.output_dir, self.export_dir, dirs_exist_ok=True)
            # show a message box that the data was exported
            QtWidgets.QMessageBox.information(self, "Export Data", "Data exported successfully")
            self.open_folder(folder=self.export_dir)

    def select_directories(self):
         # show a dialog to select the directories
        QtWidgets.QMessageBox.information(self, "Select Directories", "Please select the directories you want to export.\n When you are done, close the selection dialog.")
    
        # Set up options for the directory dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        #options |= QFileDialog.DontUseNativeDialog  # This line is optional but can be useful for some platforms

        # Get multiple directory selections
        directories = []
        while True:
            directory = QFileDialog.getExistingDirectory(None, "Select Directory to Export", dir=self.output_dir, options=options)
            if not directory:
                break  # User pressed Cancel or closed the dialog
            directories.append(directory)

        print("Selected Directories:")
        for directory in directories:
            print(directory)
        self.selected_data_paths = directories

    def export_selected_data(self):
        # export selected data
    
        # Call the function to select directories
        self.select_directories()

        # if no data was selected, show a message box
        if self.selected_data_paths == []:
            QtWidgets.QMessageBox.warning(self, "Warning", "No data selected")
            return
        # ask the user to select an output directory
        self.export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        # if no output directory was selected, show a message box
        if self.export_dir == "":
            QtWidgets.QMessageBox.warning(self, "Warning", "No output directory selected")
            return
        
        if self.selected_data_paths != [] and self.export_dir != "":
            # if an output directory was selected, export the data
            # copy all data stored in self.output_dir to the selected output directory
            for selected_data_path in self.selected_data_paths:
                # get the name of the selected data
                selected_data_name = os.path.basename(selected_data_path)
                # copy the selected data to the output directory
                shutil.copytree(selected_data_path, os.path.join(self.export_dir, selected_data_name), dirs_exist_ok=True)
            # show a message box that the data was exported
            QtWidgets.QMessageBox.information(self, "Export Data", "Data exported successfully")
            os.startfile(self.export_dir) 

    def open_folder(self, folder=None):
        # ask the user if they want to open the output folder
        open_folder = QtWidgets.QMessageBox.question(self, "Open Output Folder", "Do you want to open the output folder?",
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if open_folder == QtWidgets.QMessageBox.Yes and folder is None:
            os.startfile(self.output_dir)   
        elif open_folder == QtWidgets.QMessageBox.Yes and folder is not None:
            os.startfile(folder)  



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
    window.clear_data(ask_question=False, warn=False)   # Clear data when the window is closed



if __name__ == '__main__':
    main()
