import json

def sanitise_coco_file(file_path):
    """
    Remove invalid segmentations from a COCO file. 
    A segmentation polygon is invalid if it has less than 3 pairs of (x, y) coordinates.
    """
    print(f'Sanitising COCO file: {file_path}')
    with open(file_path, 'r') as file:
        coco_data = json.load(file)

    for annotation in coco_data['annotations']:
        segmentations = annotation['segmentation']
        sanitized_segmentations = []
        for segmentation in segmentations:
            if len(segmentation) >= 6:  # Each (x, y) pair takes 2 elements, so 3 pairs = 6 elements
                #print(f'Found a segmentation with {len(segmentation)} elements, keeping it')
                sanitized_segmentations.append(segmentation)
            else:
                print(f'Found a segmentation with {len(segmentation)} elements, discarding it')
        annotation['segmentation'] = sanitized_segmentations

        # if the label of the annotation is anything other than 1 set it to 1
        #if annotation['category_id'] != 1:
        #    annotation['category_id'] = 1

    new_file_path = file_path.replace('.json', '_sanitized.json')

    #with open(new_file_path, 'w') as file:
    #    json.dump(coco_data, file)
    
    #print('Sanitisation complete')

    # remove annotations which belong to a category that is not in the categories list
    categories = coco_data['categories']
    category_ids = [category['id'] for category in categories]

    new_annotations = []
    print(f'Number of annotations before removing invalid categories: {len(coco_data["annotations"])}')
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in category_ids:
            new_annotations.append(annotation)

    coco_data['annotations'] = new_annotations
    print(f'Number of annotations after removing invalid categories: {len(coco_data["annotations"])}')

    new_file_path = file_path.replace('.json', '_sanitized.json')

    with open(new_file_path, 'w') as file:
        json.dump(coco_data, file)



#file_path = r'coco.json'
file_path = r"C:\Users\pod38798\Downloads\new_training_data\annotations\instances_default.json"
sanitise_coco_file(file_path)
