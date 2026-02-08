import json
import os
import random
import cv2


# set random seed for reproducibility
random.seed(42)

def load_coco_data(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def display_random_samples(coco_data, images_dir, num_samples=3, output_dir=None):
    image_ids = [image['id'] for image in coco_data['images']]
    sample_ids = random.sample(image_ids, min(num_samples, len(image_ids)))

    for image_id in sample_ids:
        image_info = [image for image in coco_data['images'] if image['id'] == image_id][0]
        print(f"Reading image: {image_info['file_name']}")
        annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_id]
        
        image_path = os.path.join(images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        
        for annotation in annotations:
            bbox = annotation['bbox']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                          (0, 255, 0), 2)
            segmentation = annotation['segmentation']
            
            # draw polygon
            #for polygon in segmentation:
                #polygon = [int(x) for x in polygon]
                #polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                #cv2.polylines(image, [polygon], True, (0, 255, 0), 2)


        
        #cv2.imshow('Image with Annotations', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # save the image with annotations
        file_name = f'{image_info["file_name"][:-4]}_annotated.jpg'	
        out_file = os.path.join(output_dir, file_name)
        cv2.imwrite(out_file, image)

def display_samples_by_file_name(coco_data, images_dir, file_name, output_dir):
    image_info = [image for image in coco_data['images'] if image['file_name'] == file_name][0]
    image_id = image_info['id']
    annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_id]
    
    image_path = os.path.join(images_dir, image_info['file_name'])
    image = cv2.imread(image_path)
    
    for annotation in annotations:
        bbox = annotation['bbox']
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), 
                      (0, 255, 0), 2))
        segmentation = annotation['segmentation']
        
        # draw polygon
        #for polygon in segmentation:
            #polygon = [int(x) for x in polygon]
            #polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            #cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
    
    #cv2.imshow('Image with Annotations', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # save the image with annotations
    file_name = f'{image_info["file_name"][:-4]}_annotated.jpg'	
    out_file = os.path.join(output_dir, file_name)
    cv2.imwrite(out_file, image)


def inspect_coco_dataset(train_ann_file, val_ann_file):
    from pycocotools.coco import COCO

    # Create COCO instances for both training and validation sets
    train_coco = COCO(train_ann_file)
    val_coco = COCO(val_ann_file)

    # Get the number of images in each split
    num_train_images = len(train_coco.getImgIds())
    num_val_images = len(val_coco.getImgIds())

    # Get the number of annotations in each split
    num_train_annotations = len(train_coco.getAnnIds())
    num_val_annotations = len(val_coco.getAnnIds())

    # Get the number of categories in each split
    num_train_categories = len(train_coco.getCatIds())
    num_val_categories = len(val_coco.getCatIds())

    # Get category information for each split
    train_category_info = train_coco.loadCats(train_coco.getCatIds())
    val_category_info = val_coco.loadCats(val_coco.getCatIds())

    # Get the amount of instances for each category in each split
    train_cat_ids = train_coco.getCatIds()
    val_cat_ids = val_coco.getCatIds()
    train_cat_instances = train_coco.loadCats(train_cat_ids)
    val_cat_instances = val_coco.loadCats(val_cat_ids)


    # Print the results
    print("\n-----------------------------------\n")
    print("Training Set:")
    print("\n")
    print("Number of images:", num_train_images)
    print("Number of annotations:", num_train_annotations)
    print("Number of categories:", num_train_categories)
    print("Category Information:")
    for cat_info in train_category_info:
        print("Category ID:", cat_info['id'], "Category Name:", cat_info['name'])
        print("Number of instances:", len(train_coco.getAnnIds(catIds=cat_info['id'])))
        print(f"Percentage of training instances: {round(len(train_coco.getAnnIds(catIds=cat_info['id'])) / num_train_annotations * 100, 2)}%")
        print(f"Percentage of validation instances: {round(len(val_coco.getAnnIds(catIds=cat_info['id'])) / num_val_annotations * 100, 2)}%")
        print("\n")

    print("\n-----------------------------------\n")
    print("Validation Set:")
    print("\n")
    print("Number of images:", num_val_images)
    print("Number of annotations:", num_val_annotations)
    print("Number of categories:", num_val_categories)
    print("Category Information:")
    for cat_info in val_category_info:
        print("Category ID:", cat_info['id'], "Category Name:", cat_info['name'])
        print("Number of instances:", len(val_coco.getAnnIds(catIds=cat_info['id'])))
        print(f"Percentage of training instances: {round(len(train_coco.getAnnIds(catIds=cat_info['id'])) / num_train_annotations * 100, 2)}%")
        print(f"Percentage of validation instances: {round(len(val_coco.getAnnIds(catIds=cat_info['id'])) / num_val_annotations * 100, 2)}%")
        print("\n")

    
    # calculate the inverse frequency weights for each category to be used in the loss function
    # Call the function for both training and validation sets
    train_weights = calculate_inverse_frequency_weights(train_coco, train_cat_ids, "training set")
    val_weights = calculate_inverse_frequency_weights(val_coco, val_cat_ids, "validation set")

    # Print the weights
    print("Inverse frequency weights for training set:")
    print(train_weights)
    print("\nInverse frequency weights for validation set:")
    print(val_weights)

def calculate_inverse_frequency_weights(coco_instance, cat_ids, split):
    """
    Calculate the inverse frequency weights for each category in the dataset.
    1. Calculate the total number of annotations in the dataset.
    2. Calculate the frequency of each category.
    3. Calculate the weight for each category.
    4. Return the weights as a dictionary.
    The weights calculated represent the importance assigned to each class during the training process,
    where less frequent classes are weighted more heavily to give them higher importance.
    """
    total_annotations = sum([len(coco_instance.getAnnIds(catIds=cat_id)) for cat_id in cat_ids])
    print(f"Total number of annotations in {split}: {total_annotations}")
    category_weights = {}
    for cat_id in cat_ids:
        instances = len(coco_instance.getAnnIds(catIds=cat_id))
        print(f"Number of instances for category {cat_id}: {instances}")
        frequency = instances / total_annotations
        print(f"Frequency for category {cat_id}: {frequency}")
        weight = 1 / frequency
        category_weights[cat_id] = weight
    print("\n")
    return category_weights
        
#annotations_file_train = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_train2017.json"
#annotations_file_val = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_val2017.json"
#images_dir = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/train2017"
#output_dir = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/analysis"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
coco_data = load_coco_data(r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/annotations/instances_train2017.json")
images_dir = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/train2017"
output_dir = r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
display_random_samples(coco_data, images_dir, num_samples=3, output_dir=output_dir)

#inspect_coco_dataset(train_ann_file=annotations_file_train, val_ann_file=annotations_file_val)