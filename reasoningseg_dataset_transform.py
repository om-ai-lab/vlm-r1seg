from datasets import load_dataset
from torch.utils.data import DataLoader
import json
import PIL.Image as Image
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
import glob
import torch


def visualize_mask(image_path, mask):
    """
    可视化mask与原图的叠加效果
    :param image_path: 原图路径
    :param mask: mask图像
    """
    # 加载原图
    image = cv2.imread(image_path)
    # 创建一个与原图大小相同的彩色图像，用于显示mask
    mask_color = np.zeros_like(image)
    # 将mask区域设置为绿色（BGR格式）
    mask_color[mask == 1] = [0, 255, 0]
    # 将原图与mask彩色图像叠加，alpha为透明度
    alpha = 0.5
    beta = 1 - alpha
    gamma = 0
    overlay = cv2.addWeighted(image, alpha, mask_color, beta, gamma)
    # 显示结果
    cv2.imshow('Visualization', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


def dump_masks(data_dir, mask_dir):
    ignore_label = 255
    all_json = os.listdir(data_dir)
    all_json = [f for f in all_json if f.endswith("json")]
    all_fname = [f.split(".")[0] for f in all_json]
    for name in tqdm(all_fname):
        json_path = os.path.join(data_dir, name + ".json")
        image_path = os.path.join(data_dir, name + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]
        masks = np.stack(sampled_masks, axis=0)
        # masks = torch.from_numpy(masks)
        # label = torch.ones(masks.shape[1], masks.shape[2]) * ignore_label
        # for mask in masks[1:]:
        #     visualize_mask(image_path, mask)
        mask_path = os.path.join(mask_dir, name + ".jpg")
        mask = masks[0]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        cv2.imwrite(mask_path, mask * 255)
        # loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if "241852123_1c8229b3e7_o" in name:
            print()
        if image.shape[:2] != mask.shape:
            assert image.shape[2:] != mask.shape[2:]


def main():
    data_dir = "/home/zilun/Documents/R1-Seg/dataset/reasoningseg/test"
    mask_dir = "/home/zilun/Documents/R1-Seg/dataset/reasoningseg/mask_img_test"
    save_json_path = "reasonseg_test.json"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    # dump_masks(data_dir, mask_dir)
    all_json = os.listdir(data_dir)
    all_json = [f for f in all_json if f.endswith("json")]
    all_fname = [f.split(".")[0] for f in all_json]
    json_list = []
    for name in tqdm(all_fname):
        json_path = os.path.join(data_dir, name + ".json")
        image_path = os.path.join(data_dir, name + ".jpg")
        mask_path = os.path.join(mask_dir, name + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        for sent in sampled_sents:
            r1_data = dict()
            r1_data["image"] = image_path
            r1_data["problem"] = sent
            r1_data["solution"] = mask_path
            json_list.append(r1_data)

    with open(save_json_path, 'w') as json_file:
        json.dump(json_list, json_file, indent=4)
    return json_list


if __name__ == "__main__":
    main()