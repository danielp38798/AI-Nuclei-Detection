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

    new_file_path = file_path.replace('.json', '_sanitized.json')

    with open(new_file_path, 'w') as file:
        json.dump(coco_data, file)
    
    print('Sanitisation complete')


file_path = r'coco_new.json'
sanitise_coco_file(file_path)
