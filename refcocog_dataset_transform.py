from datasets import load_dataset
from torch.utils.data import DataLoader
import json
import PIL.Image as Image
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import numpy as np
import cv2


def dump_img(ds, img_dir):
    processed_list = []
    for data in tqdm(ds):
        img_pil = data["image"]
        name = data["id"].split("_")[-1] + ".png"
        img_path = os.path.join(img_dir, name)
        img_pil.save(img_path)
        processed_list.append(img_path)
    return processed_list


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


def resize_mask(mask, new_size):
    """
    调整mask图像的大小
    :param mask: 原始mask图像
    :param new_size: 目标大小，如(840, 840)
    :return: 调整大小后的mask图像
    """
    # 使用最近邻插值调整大小，保持二值特性
    resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return resized_mask


def dump_mask(image_path, id, dataid2seg_dict, height, width, target_mask_size, mask_dir):
    if os.path.exists(image_path):
        masks = dataid2seg_dict[id]
        polys = []
        mask = np.zeros((height, width), dtype=np.uint8)
        for seg in masks:
            # 将多边形坐标转换为整数并reshape为合适形状
            poly = np.array(seg, dtype=np.int32).reshape((int(len(seg) / 2), 2))
            polys.append(poly)
        # 填充多边形
        cv2.fillPoly(mask, polys, 1)
        if target_mask_size != mask.shape[:2]:
            mask = resize_mask(mask, target_mask_size)
        # visualize_mask(image_path, mask)
    mask_path = os.path.join(mask_dir, str(id) + ".png")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    cv2.imwrite(mask_path,  mask * 255)
    return mask_path


def dump_masks(recocog_maskgt, dataloader, target_mask_size, img_dir, mask_dir):
    mask_paths = []
    dataid2seg_dict = dict()
    for data in tqdm(recocog_maskgt):
        id = data["id"]
        masks = data["segmentation"]
        dataid2seg_dict[str(id)] = masks

    for single_data in tqdm(dataloader):
        id = single_data["id"][0].split("_")[-1]
        width = int(single_data["img_width"][0])
        height = int(single_data["img_height"][0])
        image_path = os.path.join(img_dir, str(id) + ".png")
        mask_path = dump_mask(image_path, id, dataid2seg_dict, height, width, target_mask_size, mask_dir)
        mask_paths.append(mask_path)
    return mask_paths


def make_vlmr1_json(dataloader, img_dir, mask_dir, save_json_path):
    json_list = []
    for single_data in tqdm(dataloader):
        r1_data = dict()
        id = single_data["id"][0].split("_")[-1]
        problem = single_data["problem"][0]
        image_path = os.path.join(img_dir, str(id) + ".png")
        mask_path = os.path.join(mask_dir, str(id) + ".png")
        r1_data["image"] = image_path
        r1_data["problem"] = problem
        r1_data["solution"] = mask_path
        json_list.append(r1_data)

    with open(save_json_path, 'w') as json_file:
        json.dump(json_list, json_file, indent=4)

    return json_list


def main():
    dataset_path = "/Users/zilun/Downloads/refCOCOg_2k_840"
    ds = load_dataset(dataset_path, split="train")
    recocog_json = "/Users/zilun/Desktop/R1-Seg/dataset/refcocog/instances.json"
    img_dir = "/Users/zilun/Desktop/R1-Seg/dataset/refcocog/img"
    mask_dir = "/Users/zilun/Desktop/R1-Seg/dataset/refcocog/mask_img"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    # processed_list = dump_img(ds, img_dir)
    dataloader = DataLoader(ds.with_format("torch"), batch_size=1)
    # recocog_maskgt = json.load(open(recocog_json, "r"))["annotations"]
    # target_mask_size = (840, 840)
    # mask_paths = dump_masks(recocog_maskgt, dataloader, target_mask_size, img_dir, mask_dir)
    save_json_path = "vlmr1_refcocog_2k_840.json"
    json_list = make_vlmr1_json(dataloader, img_dir, mask_dir, save_json_path)


if __name__ == "__main__":
    main()