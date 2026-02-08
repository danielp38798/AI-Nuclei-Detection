from sahi.slicing import slice_coco

import os
import argparse

def setup_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Slice COCO dataset into patches")
    parser.add_argument("--coco_annotation_file_path", 
                        default=r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/annotations/instances_val2017.json",
                        type=str, help="Path to COCO annotation file")
    parser.add_argument("--image_dir", 
                        default=r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/val2017",
                        type=str, help="Path to directory containing images")
    parser.add_argument("--slice_height", 
                        default=400,
                        type=int, help="Height of each slice")
    parser.add_argument("--slice_width", 
                        default=400,
                        type=int,  help="Width of each slice")
    parser.add_argument("--overlap_height_ratio", 
                        default=0.2,
                        type=float, help="Overlap ratio in height direction")
    parser.add_argument("--overlap_width_ratio", 
                        default=0.2,
                        type=float, help="Overlap ratio in width direction")
    parser.add_argument("--output_dir", 
                        default=r"/home/pod44433/disk/AI_nuclei_detection/train_data/new_balanced_data/sliced_coco/val2017",
                        type=str, help="Output directory to save sliced images")
    parser.add_argument("--output_coco_annotation_file_name", 
                        default="instances_val2017",
                        type=str, help="Output path to save sliced COCO annotation file")
    args = parser.parse_args()

    return args



def main():
    args = setup_args()

    print(f"Reading COCO annotation file: {args.coco_annotation_file_path}")
    print(f"Reading images from: {args.image_dir}")
    print(f"Slice height: {args.slice_height}")
    print(f"Slice width: {args.slice_width}")
    print(f"Overlap height ratio: {args.overlap_height_ratio}")
    print(f"Overlap width ratio: {args.overlap_width_ratio}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output COCO annotation file: {args.output_coco_annotation_file_name}")    
          
    coco_dict, output_dir = slice_coco(
                                    coco_annotation_file_path=args.coco_annotation_file_path,
                                    output_dir=args.output_dir,
                                    output_coco_annotation_file_name=args.output_coco_annotation_file_name,
                                    image_dir=args.image_dir,
                                    slice_height=args.slice_height,
                                    slice_width=args.slice_width,
                                    overlap_height_ratio=args.overlap_height_ratio,
                                    overlap_width_ratio=args.overlap_width_ratio,
                                    verbose=False
                                    )
    
    print(f"Saved sliced COCO annotation file: {output_dir}")

if __name__ == "__main__":
    main()
