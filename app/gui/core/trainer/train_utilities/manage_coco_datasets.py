from pathlib import Path
from datetime import date
from collections import defaultdict
from warnings import warn
import json
from shutil import copy
from pathlib import Path
import filecmp
from functools import reduce
from operator import getitem
from tqdm import tqdm
from copy import deepcopy
from itertools import accumulate

from random import shuffle

# let the user select a json file to split
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog
from PySide6.QtCore import Qt
import sys

import os


IMG_EXTS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"]
IMG_EXTS = [x.lower() for x in IMG_EXTS] + [x.upper() for x in IMG_EXTS]


def read_json(json_path):
    with open(json_path, "r") as f:
        print(f"Reading json from {json_path}")
        d = json.load(f)
    return d

def write_json(json_path, dic):
    with open(json_path, "w") as f:
        json.dump(dic, f)
    print(f"Wrote json to {json_path}")

def path(str_path, is_dir=False, mkdir=False):
    path_ = Path(str_path)
    if is_dir:
        if mkdir:
            path_.mkdir(parents=True, exist_ok=True)
        assert path_.is_dir(), path_
    else:
        assert path_.is_file(), path_
    return path_


def assure_copy(src, dst):
    assert Path(src).is_file()
    if Path(dst).is_file() and filecmp.cmp(src, dst, shallow=True):
        return
    Path(dst).parent.mkdir(exist_ok=True, parents=True)
    copy(src, dst)


def get_imgnames_dict(coco_dict_images):
    return {d["id"]: d["file_name"] for d in coco_dict_images}


def get_img2annots(coco_dict_annots):
    img2annots = defaultdict(list)
    for annot in coco_dict_annots:
        img2annots[annot["image_id"]].append(annot)
    return img2annots


def get_ltrbwh(bbox):
    l, t, w, h = bbox
    r = l + w
    b = t + h
    ltrbwh = [int(x) for x in [l, t, r, b, w, h]]
    return ltrbwh


def get_setname(cocodict, json_path):
    try:
        set_name = cocodict["info"]["description"]
        #print(f"Processing {set_name} (name from json info description)")
    except KeyError:
        json_path_p = Path(json_path)
        set_name = f"{json_path_p.parent.stem}_{json_path_p.stem}"
        #print(f"Processing {set_name} (name derived from json path)")
    return set_name


def get_flatten_name(subpath):
    subpath = Path(subpath)
    elems = [d.stem for d in subpath.parents if d.stem][::-1]
    elems.append(subpath.stem)
    return "_".join(elems)


def read_coco_json(coco_json):
    coco_dict = read_json(coco_json)
    setname = get_setname(coco_dict, coco_json)
    return coco_dict, setname


def get_imgs_from_dir(dirpath):
    return sorted(
        [img for img in dirpath.rglob("*") if img.is_file() and img.suffix in IMG_EXTS]
    )


def dict_val_from_keys_list(dic, keys_list):
    return reduce(getitem, keys_list, dic)


def write_json_in_place(orig_coco_json, coco_dict, append_str="new", out_json=None):
    if out_json is None:
        orig_json_path = Path(orig_coco_json)
        out_json_path = (
            orig_json_path.parent / f"{orig_json_path.stem}_{append_str}.json"
        )
    else:
        out_json_path = Path(out_json)
    write_json(out_json_path, coco_dict)


def parse(json_path, imgroot, outdir=None):
    json_path = path(json_path)
    imgroot_path = path(imgroot, is_dir=True)

    coco_dict = read_json(json_path)

    if outdir:
        outdir = Path(outdir)
        outroot_path = outdir / "images"
        outroot_path.mkdir(exist_ok=True, parents=True)
        return coco_dict, json_path, imgroot_path, outdir, outroot_path
    else:
        return coco_dict, json_path, imgroot_path


def merge_cats_get_id(cats, this_cat):
    for cat in cats:
        if cat["name"] == this_cat["name"]:
            return cat["id"]
    else:
        this_cat["id"] = len(cats) + 1
        cats.append(this_cat)
        return this_cat["id"]

def merge_from_dir(root_dir, output_dir):
    grandparent = Path(root_dir)
    datasets = {i for i in grandparent.iterdir() if i.is_dir()}
    print(f"Found {len(datasets)} datasets in {grandparent}")
 
    jsons_list = []
    img_roots_list = []
    for dataset in datasets:
        cocos = find_coco_in(dataset)
        if len(cocos) == 0:
            warn(f"No COCO jsons found in {dataset}")
            continue
        jsons, img_roots = zip(*cocos.values())
        jsons_list.extend(jsons)
        img_roots_list.extend(img_roots)
    
    merge(jsons_list, img_roots_list, output_dir)
    

def find_coco_in(grandparent, get_images=True):
    grandparent = Path(grandparent)
    cocos = {}
    """
    for jp in grandparent.rglob("*.json"):
        if jp.stem == jp.parent.stem:
            if get_images:
                imagedir = jp.parent / "images"
                if imagedir.is_dir():
                    cocos[jp.stem] = (jp, imagedir)
            else:
                cocos[jp.stem] = jp
    return cocos
    """
    # list all jsons in grandparent and sub dirs until 3rd level
    jsons = list(grandparent.rglob("*.json"))
    for jp in jsons:
        # in each dir with json, look for images dir; images is two levels up form json and in "images" dir
        imagedir = jp.parent.parent / "images"
        if imagedir.is_dir():
            cocos[jp.stem] = (jp, imagedir)
        
    return cocos

def merge(jsons, img_roots, output_dir, cids=None, outname="merged"):
    assert len(img_roots) == len(jsons)

    out_dir_path = Path(output_dir)
    out_image_dir = out_dir_path / "images"

    current_image_id = 1
    current_annot_id = 1
    merged_dict = {
        "info": {"description": "", "data_created": f"{date.today():%Y/%m/%d}"},
        "annotations": [],
        "categories": [],
        "images": [],
    }
    merged_names = []
    # show a progress bar

    #for i, (json_path, images_dir_path) in enumerate(zip(jsons, img_roots)):
    for i, (json_path, images_dir_path) in enumerate(tqdm(zip(jsons, img_roots), total=len(jsons))):
        cocodict, set_name = read_coco_json(json_path)
        merged_names.append(set_name)
        """
        print(f"Processing {set_name} ({i+1}/{len(jsons)})")
        print(f"Images dir: {images_dir_path}")
        print(f"JSON path: {json_path}" + "\n")
        """

        if cids is not None:
            assert len(img_roots) == len(cids)
            cids_to_merge = cids[i]
        catid_old2new = {}
        for cat in cocodict["categories"]:
            if cids is not None and int(cat["id"]) not in cids_to_merge:
                continue
            orig_cat_id = cat["id"]
            catid_old2new[orig_cat_id] = merge_cats_get_id(
                merged_dict["categories"], cat
            )

        imgid_old2new = {}
        for img in cocodict["images"]:
            imgid_old2new[img["id"]] = current_image_id
            img["id"] = current_image_id
            current_image_id += 1

            old_img_path = Path(images_dir_path) / img["file_name"]
            img["file_name"] = str(Path(set_name) / img["file_name"])
            new_img_path = out_image_dir / img["file_name"]

            assure_copy(old_img_path, new_img_path)
            merged_dict["images"].append(img)

        for annot in cocodict["annotations"]:
            if cids is not None and cat["id"] not in cids_to_merge:
                continue
            annot["id"] = current_annot_id
            current_annot_id += 1
            annot["image_id"] = imgid_old2new[annot["image_id"]]
            annot["category_id"] = catid_old2new[annot["category_id"]]
            merged_dict["annotations"].append(annot)

    merged_dict["info"]["description"] = "+".join(merged_names)

    out_json = out_dir_path / f"{outname}.json"
    write_json(out_json, merged_dict)


"""
Split up a COCO JSON file by images into N sets defined by ratio of total images

To shuffle images, flag `shuffle=True`

"""

def split_from_file(cocojson, ratios, images_per_split=None, names=None, do_shuffle=False, output_dir=None):
    coco_dict, setname = read_coco_json(cocojson)
    import numpy as np
    if images_per_split != None:
        total_imgs = len(coco_dict["images"])
        print(f"Total imgs: {total_imgs}")
        splits_num = int(round(total_imgs / images_per_split)) 
        print(f"Splitting all images into {splits_num} sets of {images_per_split} images each")
        
        if total_imgs % images_per_split != 0:
            warn(f"Total images ({total_imgs}) is not a multiple of images_per_split ({images_per_split})")
            # round up to the nearest multiple of images_per_split
            splits_num += 1
            images_per_split = int(np.ceil(total_imgs / splits_num))
            print(f"Rounding up to {splits_num} sets of {images_per_split} images each")

        ratios = [1/splits_num for i in range(splits_num)]
        # assure sum ratios = 1
        if sum(ratios) != 1:
            ratios[-1] += 1 - sum(ratios)
            print(f"Adjusted Ratios: {ratios}")
        else:
            print(f"Ratios: {ratios}")
        print(f"Splitting into {splits_num} sets of {images_per_split} images each")
    

        # contruct names for the splits
        if names:
            assert len(ratios) == len(names)
        else:
            names = [f"split_{i}" for i in range(len(ratios))]

    images_folder = os.path.join(os.path.dirname(cocojson), "images")
    print(f"Images folder: {images_folder}")
    split_coco_dicts = split(
        coco_dict, ratios, 
        names=names, do_shuffle=do_shuffle, setname=setname, 
        output_dir=output_dir,
        images_folder=images_folder
    )

    """
    for name, new_cocodict in split_coco_dicts.items():
        #write_json_in_place(cocojson, new_cocodict, append_str=name)
        out_json = Path(cocojson).parent / f"{name}.json"
        write_json(out_json, new_cocodict)
    """


def default_coco_dict():
    new_dict = {
        "images": [],
        "annotations": [],
    }
    return new_dict

def copy_images_to_split_dir(split_name, split_images, output_dir, images_folder):

    split_dir = os.path.join(output_dir, split_name)
    split_img_dir = os.path.join(split_dir, "images")
    os.makedirs(split_img_dir, exist_ok=True)

    for img_dict in split_images:
        img_name = img_dict["file_name"]
        img_path = os.path.join(images_folder, img_name)
        img_name = os.path.basename(img_name)
        split_img_path = os.path.join(split_img_dir, img_name)
        print(f"Copying {img_path} to {split_img_path}")
        copy(img_path, split_img_path)


def split(coco_dict, ratios, names=None, do_shuffle=False, setname="", output_dir=None, images_folder=None):
    assert sum(ratios) == 1.0, "Ratios given does not sum up to 1.0"
    if names:
        assert len(ratios) == len(names)
    else:
        names = [f"split_{i+1}" for i in range(len(ratios))]

    total_imgs = len(coco_dict["images"])
    print(f"Total imgs: {total_imgs}")
    splits_num = [int(round(x * total_imgs)) for x in ratios]
    #assert sum(splits_num) == total_imgs
    splits_num[0] -= 1
    splits_acc = list(accumulate(splits_num))
    #assert splits_acc[-1] == total_imgs - 1

    if do_shuffle:
        shuffle(coco_dict["images"])

    split_coco_dicts = defaultdict(default_coco_dict)
    img_ids_maps = defaultdict(dict)
    oldimgid2name = {}
    split_idx = 0
    this_name = names[split_idx]
    this_split_images = split_coco_dicts[this_name]["images"]
    this_img_ids_map = img_ids_maps[this_name]
    for i, img_dict in enumerate(coco_dict["images"]):
        oldimgid2name[img_dict["id"]] = this_name

        new_img_dict = deepcopy(img_dict)
        new_img_id = len(this_split_images) + 1

        this_img_ids_map[img_dict["id"]] = new_img_id
        new_img_dict["id"] = new_img_id
        this_split_images.append(new_img_dict)

        if i >= splits_acc[split_idx]:
            if split_idx == len(splits_acc) - 1:
                break
            else:
                split_idx += 1
                this_name = names[split_idx]
                this_split_images = split_coco_dicts[this_name]["images"]
                this_img_ids_map = img_ids_maps[this_name]


    for annot_dict in coco_dict["annotations"]:
        name = oldimgid2name[annot_dict["image_id"]]
        new_annot_dict = deepcopy(annot_dict)
        new_annot_dict["id"] = len(split_coco_dicts[name]["annotations"]) + 1
        new_annot_dict["image_id"] = img_ids_maps[name][annot_dict["image_id"]]
        split_coco_dicts[name]["annotations"].append(new_annot_dict)

    for name, dic in split_coco_dicts.items():
        if "info" in coco_dict:
            dic["info"] = deepcopy(coco_dict["info"])
            dic["info"]["description"] = f"{setname}_{name}"
        if "licenses" in coco_dict:
            dic["licenses"] = deepcopy(coco_dict["licenses"])
        if "categories" in coco_dict:
            dic["categories"] = deepcopy(coco_dict["categories"])

    for split_name, split_coco_dict in split_coco_dicts.items():
        if output_dir:
            out_json = os.path.join(output_dir, split_name)
            os.makedirs(out_json, exist_ok=True)
            out_json = os.path.join(out_json, f"{split_name}.json")
            write_json(out_json, split_coco_dict)
            copy_images_to_split_dir(split_name, split_coco_dict["images"], output_dir, images_folder)

    #return split_coco_dicts


if __name__ == "__main__":

    root_dir = r"C:\Users\pod44433\Downloads\train_data_stage1_16_02_24\extracted_test"
    output_dir = r"C:\Users\pod44433\Downloads\train_data_stage1_16_02_24\extracted_test\output"
    merge_from_dir(root_dir, output_dir)

    #json_file = r"C:\Users\pod44433\Downloads\train_data_stage1_16_02_24\extracted_test\output\merged.json"
    #out_dir = r"C:\Users\pod44433\Downloads\train_data_stage1_16_02_24\extracted_test\output\split"
    #split_from_file(cocojson=json_file, images_per_split=20, ratios=None, names=None, do_shuffle=False, output_dir=out_dir)
    print("Done")




