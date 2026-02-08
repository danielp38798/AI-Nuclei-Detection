from pycocotools.coco import COCO
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def split_into_training_and_validation_lists(sorted_list):
    """
    Split the sorted list into training and validation lists
    :param sorted_list: sorted list
    :return: training and validation lists
    """
    # Initialize training and validation lists
    training_list = []
    validation_list = []

    # Start by assigning the first element to the training list
    if sorted_list:
        training_list.append(sorted_list.pop(0))
    
    while sorted_list: # While the sorted list is not empty
        # Calculate the current ratio between validation and training lists
        sum_training = sum(training_list)
        sum_validation = sum(validation_list)
        ratio = (sum_validation / sum_training) if sum_training > 0 else 0

        # Decide based on the ratio which list to fill
        if ratio < 0.2:
            # Add to the validation list if the ratio is below 20%
            validation_list.append(sorted_list.pop(0))
        else:
            # Add to the training list until the ratio is met again
            training_list.append(sorted_list.pop(0))
        
        # Check if the ratio needs to be adjusted after adding
        sum_training = sum(training_list)
        sum_validation = sum(validation_list)
        ratio_after_adding = (sum_validation / sum_training) if sum_training > 0 else 0

        # If the ratio falls above 80% for the training list, add to the validation list
        if ratio_after_adding >= 0.8:
            if sorted_list:
                validation_list.append(sorted_list.pop(0))

    return training_list, validation_list




def balance_coco_dataset(annFile):
    """
    Balance the COCO dataset by balancing across classes, so that the training and test data have a similar ratio between the classes
    :param annFile: Path to the COCO annotation file
    :return: Training and test ImgIDs
    """

    # Initialize the COCO API
    coco = COCO(annFile)

    # Get the category IDs
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    nms=[cat['name'] for cat in cats]
    print('\nCategories: \n', nms)

    # Get the number of images
    imgIds = coco.getImgIds()
    print(f"\nNumber of images: {len(imgIds)}")

    # 1. Get the count of each cell class across all data
    cell_counts = {cat['name']: 0 for cat in cats}
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            category = coco.loadCats(ann['category_id'])[0]['name']
            cell_counts[category] += 1
            
    print(f"\nNumber of cell types: {cell_counts}")
    
    # Calculate the global ratio between the classes
    total_cells = sum(cell_counts.values())
    print(f"Total number of cells: {total_cells}")
    cell_ratios = {key: value / total_cells for key, value in cell_counts.items()}
    print("\nClass ratios: ", cell_ratios)

    
    # Weight the rare class with the factor between the classes
    class_min = min(cell_ratios, key=cell_ratios.get)
    print("Rarest cell type: ", class_min)
    class_max = max(cell_ratios, key=cell_ratios.get)
    print("Most common cell type: ", class_max)
    
    print("\nClass ratios (max/min): ", cell_ratios[class_max] / cell_ratios[class_min])
 
    #factor_between_classes = cell_ratios[class_max] / cell_ratios[class_min]
    #print("\nFactor between classes: ", factor_between_classes)
    
    class_weights = {key: cell_ratios[class_max] / value for key, value in cell_ratios.items()}
    print("\nCell type weights: ")
    for key, value in class_weights.items():
        print(f"{key}: {value}")
    
    # 2. Sum up this virtual count - obtaining a total count of objects per image
    cell_counts_without_weights_complete_dataset = {}
    cell_counts_with_weights_complete_dataset = {}

    
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        
        total_cells_per_image = {cat['name']: 0 for cat in cats}
        total_cells_per_image_with_weights = {cat['name']: 0 for cat in cats}
        
        
        # Get the total count of objects per image
        for ann in anns:
            category = coco.loadCats(ann['category_id'])[0]['name']
            if category in total_cells_per_image:
                total_cells_per_image[category] += 1
        
        # Multiply the count of cell types per image with the weighted cell types -> Obtain the virtual count of objects per image
        for key, value in total_cells_per_image.items():
            # Weight the cell types
            # actual number of cells * weighting factor stored in class_weights
            weighted_cell_count = value * class_weights[key]
            total_cells_per_image_with_weights[key] = weighted_cell_count
            
        #print("\nImage ID: ", imgId)
        #print("Number of cell types per image: ")
        #for key, value in total_cells_per_image.items():
        #    print(f"{key}: {value}")
        
        #print("Number of weighted cell types per image: ")
        #for key, value in total_cells_per_image_with_weights.items():
            #print(f"{key}: {value}")
        cell_counts_without_weights_complete_dataset[imgId] = {}
        cell_counts_with_weights_complete_dataset[imgId] = {}
        
        cell_counts_without_weights_complete_dataset[imgId]['PER CLASS'] = total_cells_per_image
        cell_counts_with_weights_complete_dataset[imgId]['PER CLASS'] = total_cells_per_image_with_weights
        
        cell_counts_without_weights_complete_dataset[imgId]['TOTAL'] = sum(total_cells_per_image.values())
        cell_counts_with_weights_complete_dataset[imgId]['TOTAL'] = sum(total_cells_per_image_with_weights.values())
    
    #print("\nTotal number of cells per image (unweighted): ")
    #print(cell_counts_without_weights_complete_dataset)
    
    #print("\nNumber of added cells per image: ")
    diff = {key: value['TOTAL'] - cell_counts_without_weights_complete_dataset[key]['TOTAL'] for key, value in cell_counts_with_weights_complete_dataset.items()}
    #print(diff)
    
    # Calculate the percentage contribution of 'added' cells from each image. -> For both cell classes combined
    # -> Calculate the percentage ratio to the virtual total count per image
    percentage_of_added_cells_per_image = {}
    
    for key, value in cell_counts_with_weights_complete_dataset.items():
        # key: image ID, value: number of virtual cells
        diff = value['TOTAL'] - cell_counts_without_weights_complete_dataset[key]['TOTAL']
        percentage = (diff / value['TOTAL']) * 100
        
        percentage_of_added_cells_per_image[key] = percentage
        
    #print("\nPercentage contribution of added cells per image: ")
    #print(percentage_of_added_cells_per_image)
    
    #print("\nTotal number of cells per image (weighted): ")
    #print(cell_counts_with_weights_complete_dataset)
    

    
    # Calculate the percentage ratio of the number of cells per image to the total cell count
    percentage_total_cell_count_per_image = {}
    total_virtual_cell_count = sum([value['TOTAL'] for value in cell_counts_with_weights_complete_dataset.values()])
    for key, value in cell_counts_with_weights_complete_dataset.items():
        percentage = (value['TOTAL'] / total_virtual_cell_count) * 100
        percentage_total_cell_count_per_image[key] = percentage
    
    #print("\nPercentage ratio of cells per image to the total cell count: ")
    #print(percentage_total_cell_count_per_image)
    
    #print("\nTotal number of cells: ", total_virtual_cell_count)
    #print("Sum of percentages: ", sum(percentage_total_cell_count_per_image.values()))
    
    
    # 3. Sort the images based on the percentage ratio to the virtual total count
    #sorted_percentage_of_added_cells_per_image = {k: v for k, v in sorted#(percentage_of_added_cells_per_image.items(), key=lambda item: item[1], reverse=True)}
    #print("\nSorted images based on percentage contribution: ")
    #print(sorted_percentage_of_added_cells_per_image)
    
    percentage_list = list(percentage_total_cell_count_per_image.values())
    sorted_img_ids = [img_id for img_id, percentage in sorted(percentage_total_cell_count_per_image.items(), key=lambda item: item[1], reverse=True)]
    
    #print("\nPercentage contributions of images: ")
    #print(percentage_list)
    
    #print("\nSorted images based on percentage contribution: ")
    #print(sorted_img_ids)
    
    
    # 4. Split the images into training and test data
    train_img_ids, test_img_ids = split_into_training_and_validation_lists(sorted_img_ids)
        

          
    print("\nTraining ImgIDs: ", train_img_ids)
    print("Test ImgIDs: ", test_img_ids)
    
    print("\nNumber of training ImgIDs: ", len(train_img_ids))
    print("Number of test ImgIDs: ", len(test_img_ids))
    
    # Check if the class ratios in the training and test data are similar
    # i.e. the number of ImgIDs should be roughly 80/20
    # also, the ratio between cell types should be similar in both lists
    train_cell_counts = {cat['name']: 0 for cat in cats}
    test_cell_counts = {cat['name']: 0 for cat in cats}
    
    for img_id in train_img_ids:
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            category = coco.loadCats(ann['category_id'])[0]['name']
            train_cell_counts[category] += 1
            
    for img_id in test_img_ids:
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            category = coco.loadCats(ann['category_id'])[0]['name']
            test_cell_counts[category] += 1
            
    print("\nNumber of cell types in the training data: ", train_cell_counts)
    print("Ratio of cell types in the training data: ")
    for key, value in train_cell_counts.items():
        print(f"{key}: {value / sum(train_cell_counts.values())}") 

    print("\nNumber of cell types in the test data: ", test_cell_counts)   
    print("Ratio of cell types in the test data: ")
    for key, value in test_cell_counts.items():
        print(f"{key}: {value / sum(test_cell_counts.values())}")

   
    return train_img_ids, test_img_ids

def create_balanced_coco_datasets(annFile, train_img_ids, test_img_ids, output_dir):
    """
    Create the balanced COCO datasets
    :param annFile: Path to the COCO annotation file
    :param train_img_ids: Training ImgIDs
    :param test_img_ids: Test ImgIDs
    :param output_dir: Output directory
    :return: None
    """

    print("\nCreating the balanced COCO datasets...")
    
    # Initialize the COCO API
    coco = COCO(annFile)
    
    # Get the category IDs
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    
    # Create the training and test datasets
    #train_data = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'], 'categories': coco.dataset['categories'], 'images': [], 'annotations': []}
    #test_data = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'], 'categories': coco.dataset['categories'], 'images': [], 'annotations': []}
    train_data = {'info': coco.dataset['info'], 'categories': coco.dataset['categories'], 'images': [], 'annotations': []}
    test_data = {'info': coco.dataset['info'], 'categories': coco.dataset['categories'], 'images': [], 'annotations': []}

    # Get the images and annotations
    imgIds = coco.getImgIds()
    annIds = coco.getAnnIds()
    
    # Create the training and test datasets
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        
        if imgId in train_img_ids:
            train_data['images'].append(img)
            train_data['annotations'].extend(anns)
        elif imgId in test_img_ids:
            test_data['images'].append(img)
            test_data['annotations'].extend(anns)

    # Save the training and test datasets
    annotations_output_dir = os.path.join(output_dir, 'annotations')
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir) 
    # Save the training and test datasets
    train_data_output = os.path.join(annotations_output_dir, 'instances_train2017.json')
    with open(train_data_output, 'w') as f:
        json.dump(train_data, f, indent=3)
    print(f"Training data saved to {train_data_output}")
        
    test_data_output = os.path.join(annotations_output_dir, 'instances_val2017.json')
    with open(test_data_output, 'w') as f:
        json.dump(test_data, f, indent=3)
    print(f"Test data saved to {test_data_output}")

    # Save the images
    images_dir_train = os.path.join(output_dir, 'train2017')
    images_dir_test = os.path.join(output_dir, 'val2017')
    if not os.path.exists(images_dir_train):
        os.makedirs(images_dir_train)
    if not os.path.exists(images_dir_test):
        os.makedirs(images_dir_test)

    # Copy the images to the output directory
    print("\nCopying the images to the output directory...")
    original_images_dir = os.path.join(os.path.dirname(os.path.dirname(annFile)), 'images')
    for img in train_data['images']:
        img_path = os.path.join(original_images_dir, img['file_name'])
        os.system(f'cp {img_path} {images_dir_train}')
    for img in test_data['images']:
        img_path = os.path.join(original_images_dir, img['file_name'])
        os.system(f'cp {img_path} {images_dir_test}')

    print("\nDone!")

        
    return None

# Path to the COCO annotation file
annFile = r'/home/pod44433/disk/AI_nuclei_detection/train_data/annotations/instances_default.json'
train_img_ids, test_img_ids = balance_coco_dataset(annFile)

print("Training ImgIDs: ", train_img_ids)
print("Test ImgIDs: ", test_img_ids)

print("Number of training ImgIDs: ", len(train_img_ids))
print("Number of test ImgIDs: ", len(test_img_ids))
total_images = len(train_img_ids) + len(test_img_ids)

print(f"RATIO of Training to Test data: {len(train_img_ids) / total_images} : {len(test_img_ids) / total_images}")

# Create the balanced COCO datasets
#output_dir = r'/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data'
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#create_balanced_coco_datasets(annFile, train_img_ids, test_img_ids, output_dir=output_dir)

# Save the training and test ImgIDs
#np.save('path/to/train_img_ids.npy', train_img_ids)
#np.save('path/to/test_img_ids.npy', test_img_ids)


