import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from gui.core.trainer.train_utils.utils import *
from gui.core.trainer.train_utils.utils import ImageTiler
from torchvision.transforms import ToPILImage
from pycocotools import mask as coco_mask
from gui.core.sahi.slicing import slice_coco

# ---------------------- custom data set and data loader ------------------------------- #
class COCODataset_old(torch.utils.data.Dataset):
    def __init__(self, root, annotation, stage, transforms=None):
        self.root = root
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        self.create_patches = True
        self.stage = stage
        self.patches_images_dir = os.path.join(self.root, f"patches_{self.stage}", "images")
        self.patches_annotations_dir = os.path.join(self.root, f"patches_{self.stage}", "annotations")
        self.initial_read = True
        self.update_coco_annotation()

        
    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        if self.initial_read:
            print(f"initial read: {self.initial_read}")
            image_path = os.path.join(self.root, path)
        elif self.create_patches == True and self.initial_read == False:
            print(f"create patches: {self.create_patches}")
            image_path = os.path.join(self.patches_images_dir, path)
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        #img = np.array(img).transpose(1,2,0) # convert to numpy array and transpose to H x W x C
        #img = np.array(img, dtype=np.float32).transpose(2,1,0) 
        #print(f"initial img.shape: {img.shape}")
        
        # number of objects in the image
        num_objs = len(coco_annotation)
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = []
        for i in range(num_objs):
            original_id = coco_annotation[i]['category_id']
            shifted_id = original_id #- 1  # subtract 1 to shift the class IDs (for 0-based indexing)
            labels.append(shifted_id)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = int(img_id) #torch.tensor([img_id], dtype=torch.int64) #must be int for the coco evaluator!
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # create masks of the segmentation
        masks = []
        for i in range(num_objs):
            masks.append(coco.annToMask(coco_annotation[i]))
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["masks"] = masks

        if self.transforms is not None:
            img = self.transforms(img)

        #print(f"img.shape: {img.shape}")

        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)
        #return len(self.imgIds)
    
    def set_transform(self, transform):
        self.transforms = transform
        
    def update_coco_annotation(self):

        if self.create_patches and self.initial_read:
            
            self.tiler = ImageTiler(patchsize=400, overlap=0.2)
    
            # annotation IDs need to start at 1, not 0, see torchvision issue #1530
            #ann_id = 1
            dataset = {"images": [], "categories": [], "annotations": []}
            categories = set()
            
            unique_patch_id = 1
            for idx, (img, target) in enumerate(self):
                
                print(f"updating coco annotation for image: {idx+1} of {len(self)}")
                # convert image from tensor to PIL image
                img = ToPILImage()(img)
                # get the patches and targets from the images and annotations
                indices, patch_anno_dict = self.tiler.process_image(img, target=target) # set to training mode by passing in the target
                
                #patches = []
                #targets = []
                ann_id = 1

                for patch_num, (patch_bounds, value) in enumerate(patch_anno_dict.items()):
                    patch = value[0]
                    target = value[1]
                    #patches.append(patch) 
                    #targets.append(target)
                    
                    # store patch as an image under the same image id
                    
                    # convert patch to PIL image
                    patch_data = ToPILImage()(patch)
                    # save patch to file
                    
                    if not os.path.exists(self.patches_images_dir):
                        os.makedirs(self.patches_images_dir)
                    patch_file_name = f"{self.stage}_patch_{unique_patch_id}.png"
                    file_path = os.path.join(self.patches_images_dir, patch_file_name)
                    patch_data.save(file_path)

                    # add patch as an image
                    img_dict = {}
                    img_id = unique_patch_id
                    img_dict["id"] = img_id
                    #print(f"img_id: {img_id}")
                    img_dict["width"] = patch.shape[-1]
                    img_dict["height"] = patch.shape[-2]
                    img_dict["file_name"] = patch_file_name
                    dataset["images"].append(img_dict)
                    bboxes = target["boxes"].clone()
                    # avoid error when there are no bounding boxes
                    if len(bboxes) == 0:
                        continue
                    #print(bboxes)
                    bboxes[:, 2:] -= bboxes[:, :2]
                    bboxes = bboxes.tolist()
                    labels = target["labels"].tolist()
                    areas = target["area"].tolist()
                    iscrowd = target["iscrowd"].tolist()
                    if "masks" in target:
                        masks = target["masks"]
                        # make masks Fortran contiguous for coco_mask
                        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
                    if "keypoints" in target:
                        keypoints = target["keypoints"]
                        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
                    num_objs = len(bboxes)
                    for i in range(num_objs):
                        ann = {}
                        ann["image_id"] = img_id
                        ann["bbox"] = bboxes[i]
                        ann["category_id"] = labels[i]
                        categories.add(labels[i])
                        ann["area"] = areas[i]
                        ann["iscrowd"] = iscrowd[i]
                        ann["id"] = ann_id
                        if "masks" in target:
                            #ann["segmentation"] = coco_mask.encode(masks[i].numpy())
                            #ann["area"] = coco_mask.area(ann["segmentation"])
                            # store masks in RLE format as coco does but to be able to dump them to json we need to decode them
                            encoded_seg = coco_mask.encode(masks[i].numpy())
                            encoded_seg['counts'] = encoded_seg['counts'].decode('utf-8')
                            ann["segmentation"] = encoded_seg
                            
                        if "keypoints" in target:
                            ann["keypoints"] = keypoints[i]
                            ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
                        dataset["annotations"].append(ann)
                        ann_id += 1
                    unique_patch_id += 1
            dataset["categories"] = [{"id": i} for i in sorted(categories)]
            
            #save COCO file
            #output_dir = os.path.join(self.root, f"patches_{self.stage}", "annotations")
            if not os.path.exists(self.patches_annotations_dir):
                os.makedirs(self.patches_annotations_dir)
            file_name = os.path.join(self.patches_annotations_dir, f"patches_annotations_{self.stage}.json")
            # create an empty json file and write the data to it, this will overwrite any existing file
            with open(file_name, 'w') as f:
                json.dump(dataset, f)
                
            new_coco = COCO(file_name)
            #self.new_coco.createIndex()
            #return self.new_coco
            self.coco = new_coco
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.catIds = self.coco.getCatIds()
            self.imgIds = self.coco.getImgIds()
            print(f"len(self.ids): {len(self.imgIds)}")
        
            self.initial_read = False
        


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, stage, transforms=None):
        self.root = root
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.catIds = self.coco.getCatIds()
        self.imgIds = self.coco.getImgIds()
        self.create_patches = True
        self.stage = stage
        #self.patches_images_dir = os.path.join(self.root, f"patches_{self.stage}", "images")
        #self.patches_annotations_dir = os.path.join(self.root, f"patches_{self.stage}", "annotations")
        #self.initial_read = True
        #self.update_coco_annotation()

        
    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        #if self.initial_read:
            #print(f"initial read: {self.initial_read}")
            #image_path = os.path.join(self.root, path)
        #elif self.create_patches == True and self.initial_read == False:
            #print(f"create patches: {self.create_patches}")
        image_path = os.path.join(self.root, path)
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        #img = np.array(img).transpose(1,2,0) # convert to numpy array and transpose to H x W x C
        #img = np.array(img, dtype=np.float32).transpose(2,1,0) 
        #print(f"initial img.shape: {img.shape}")
        
        # number of objects in the image
        num_objs = len(coco_annotation)
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = []
        for i in range(num_objs):
            original_id = coco_annotation[i]['category_id']
            shifted_id = original_id #- 1  # subtract 1 to shift the class IDs (for 0-based indexing)
            labels.append(shifted_id)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = int(img_id) #torch.tensor([img_id], dtype=torch.int64) #must be int for the coco evaluator!
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # create masks of the segmentation
        masks = []
        for i in range(num_objs):
            masks.append(coco.annToMask(coco_annotation[i]))
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["masks"] = masks

        if self.transforms is not None:
            img = self.transforms(img)

        #print(f"img.shape: {img.shape}")

        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)
        #return len(self.imgIds)
    
    def set_transform(self, transform):
        self.transforms = transform


# ---------------------- custom data set and data loader ------------------------------- #
# custom data loader which only return one patch of the image
class PatchDataloader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.coco_annotation = dataset.coco
        self.imgIds = list(sorted(self.coco_annotation.imgs.keys()))
        self.catIds = self.coco_annotation.getCatIds()
        self.imgIds = self.coco_annotation.getImgIds()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.tiler = ImageTiler(patchsize=400, overlap=0.2)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        
        
    def get_filename(self, index):
        img_id = self.imgIds[index]
        path = self.coco_annotation.loadImgs(img_id)[0]['file_name']
        return path
    
    def get_annotation(self, index):
        img_id = self.imgIds[index]
        ann_ids = self.coco_annotation.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco_annotation.loadAnns(ann_ids)
        return coco_annotation
    
 
    def __iter__(self):
        # get the iterator from the super class
        super_iter = super().__iter__()
        
        # iterate through the super class iterator
        for batch in super_iter:
            # get the images and annotations from the batch
            images, annotations = batch

            # get the patches and targets from the images and annotations
            #patches, targets = self.patch_transform(images[0], annotations[0])
            
            # convert image from tensor to PIL image
            
            # first transpose the image to the correct format  C x H x W
            #print(f"type(images[0]): {type(images[0])}")

            if type(images[0]) == torch.Tensor:
                #img = images[0].permute(1,2,0)
                img = ToPILImage()(images[0])

            indices, patch_anno_dict = self.tiler.process_image(img, target=annotations[0]) # set to training mode by passing in the target

            patches = []
            targets = []
            for patch_bounds, value in patch_anno_dict.items():
                patch = value[0]
                target = value[1]
                patches.append(patch) 
                targets.append(target)

            # create a new batch with the patches and targets
            new_batch = (patches, targets, indices)

            # return the new batch
            yield new_batch




def update_coco_annotation(coco_annotation_file_path, image_dir, stage):

    #coco_annotation_file_path = r"data/annotations/instances_default_train.json"
    #image_dir = r"data\images"

    images_output_dir = os.path.join(image_dir, f"patches_{stage}")
    if not os.path.exists(images_output_dir):
        print(f"creating patches directory: {images_output_dir}")
        os.makedirs(images_output_dir)
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            slice_height=400,
            slice_width=400,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            output_dir= images_output_dir,
            output_coco_annotation_file_name=f"instances_{stage}_sliced",
            verbose=False,
            ignore_negative_samples=True,
        )
        return coco_dict, coco_path, images_output_dir
    
    # if the coco annotation file has already been sliced
    else:
        print(f"patches directory already exists: {images_output_dir}")
        # check if the annotation file exists
        annotation_file = os.path.join(images_output_dir, f"instances_{stage}_sliced_coco.json")
        if os.path.exists(annotation_file):
            print(f"sliced annotation file exists: {annotation_file}")
            # load the coco annotation file
            coco_dict = COCO(annotation_file)
            return coco_dict, annotation_file, images_output_dir
        else:
            print(f"sliced annotation file does not exist: {annotation_file}")
            # if the annotation file does not exist, then slice the coco annotation file
            coco_dict, coco_path = slice_coco(
                coco_annotation_file_path=coco_annotation_file_path,
                image_dir=image_dir,
                slice_height=400,
                slice_width=400,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                output_dir= images_output_dir,
                output_coco_annotation_file_name=f"instances_{stage}_sliced",
                verbose=False,
                ignore_negative_samples=True,
            )
            return coco_dict, coco_path, images_output_dir


def create_dataset(images_folder, train, annotation_file, perform_slicing):
    # add the transform to the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(0.5),
        #normalize,
    ])
    print(f"annotation_file: {annotation_file}")
    if train:
        if perform_slicing:
            # update the coco annotation file
            coco_dict, coco_path, images_output_dir = update_coco_annotation(coco_annotation_file_path=annotation_file, 
                                                                            image_dir=images_folder, stage='train',
                                                                            )
            annotation_file = coco_path
            images_folder = images_output_dir

        dataset = COCODataset(root=images_folder, annotation=annotation_file, stage='train', transforms=None)
        dataset.set_transform(transform)
        return dataset
    else:
        if perform_slicing:
            # update the coco annotation file
            coco_dict, coco_path, images_output_dir = update_coco_annotation(coco_annotation_file_path=annotation_file, 
                                                                            image_dir=images_folder, stage='val',
                                                                            )
            annotation_file = coco_path
            images_folder = images_output_dir
        dataset = COCODataset(root=images_folder, annotation=annotation_file, stage='val', transforms=None)
        dataset.set_transform(transform)
        return dataset


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    # create a dataloader using the dataset and the transform
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    #dataloader = PatchDataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f"len(dataloader): {len(dataloader)}")

    return dataloader

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

