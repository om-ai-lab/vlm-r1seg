import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk
from tqdm import tqdm
import pdb
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from metric_utils import AverageMeter, Summary, intersectionAndUnionGPU
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
import re
import cv2
import pytorch_lightning as pl


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
segmentation_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))  
    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_path=None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
            

def predict_sam2(image, input_box, input_point=None, input_label=None, save_path=None):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentation_model.set_image(image)
        if input_point is not None and input_label is not None:
            masks, scores, _ = segmentation_model.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
            )
        else:
            masks, scores, _ = segmentation_model.predict(
                box=input_box[None, :]
            )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
    if save_path is not None:
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, box_coords=input_box, save_path=save_path)
    return masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="./checkpoint/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--reasonseg_json", type=str, default="./dataset/reasonseg_test.json")
    parser.add_argument("--batch_size", type=int, default=20)
    return parser.parse_args()


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)
    # pdb.set_trace()
    if json_match:
        data = json.loads(json_match.group(0))
        # 查找bbox键
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        # pdb.set_trace()
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [round(int(content_bbox[0]) * x_factor), round(int(content_bbox[1]) * y_factor),
                            round(int(content_bbox[2]) * x_factor), round(int(content_bbox[3]) * y_factor)]
        # 查找points键
        points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0]) * x_factor), round(int(point1[1]) * y_factor)]
            point2 = [round(int(point2[0]) * x_factor), round(int(point2[1]) * y_factor)]
            points = [point1, point2]

    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
    return content_bbox, points, think_text


def calculate_giou_batch(output_texts, gt_masks, images, intersection_meter, union_meter, acc_iou_meter, xy_factors):
    intersection, union, acc_iou = 0.0, 0.0, 0.0
    for i, (gt_mask, output_text) in enumerate(zip(gt_masks, output_texts)):
        x_factor, y_factor = xy_factors[i]
        image = images[i]
        gt_mask = np.array(gt_mask)
        try:
            bbox, points, think = extract_bbox_points_think(output_text, x_factor, y_factor)
            print("Thinking process: ", think)    
            pred_mask = predict_sam2(image, bbox, points, [1, 1])[0]
            pred_mask = torch.from_numpy(pred_mask)
            gt_mask = gt_mask // 255
            gt_mask = torch.from_numpy(gt_mask).int().to("cuda")
            pred_mask = (pred_mask > 0).int().to("cuda")

            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_mask.contiguous().clone(), gt_mask.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        except Exception:
            pass  # Continue to next verification method if this fails
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
    acc_iou = acc_iou.cpu().numpy() / len(gt_masks)
    intersection_meter.update(intersection), union_meter.update(union)
    acc_iou_meter.update(acc_iou, n=len(gt_masks))
    # return intersection_meter, union_meter, acc_iou_meter


class SegDataset(Dataset):
    def __init__(self, image_path_list, mask_path_list, text_list):
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.text_list = text_list
        
        assert len(self.image_path_list) == len(self.mask_path_list) == len(self.text_list), \
            "The lengths of image_path_list, mask_path_list, and text_list must be the same."
    
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        mask_path = self.mask_path_list[idx]
        text = self.text_list[idx]
        
        return image_path, mask_path, text


def custom_collate_fn(batch):
    """
    自定义 collate_fn，用于处理每个 batch 的数据。
    
    参数:
        batch: 一个列表，包含多个样本，每个样本是 (image_path, mask_path, text)
    
    返回:
        一个元组，包含：
            - image_paths: 当前 batch 的所有 image 路径
            - mask_paths: 当前 batch 的所有 mask 路径
            - texts: 当前 batch 的所有 text
    """
    image_paths = []
    mask_paths = []
    texts = []
    
    for data in batch:
        image_path, mask_path, text = data
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        texts.append(text)
    
    return image_paths, mask_paths, texts


def main():
    pl.seed_everything(2024)
    args = parse_args()
    reasonseg_json = json.load(open(args.reasonseg_json, "r"))

    image_path_list = []
    mask_path_list = []
    text_list = []

    for data in tqdm(reasonseg_json):
        image_path = data["image"]
        mask_path = data["solution"]
        text = data["problem"]
        image_path_list.append(image_path)
        mask_path_list.append(mask_path)
        text_list.append(text)

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox and points." \
        "Compare the difference between objects and find the most closely matched one." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"
            
    dataset = SegDataset(image_path_list, mask_path_list, text_list)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=args.batch_size, shuffle=False)
    for index, batch in tqdm(enumerate(dataloader)):
        image_path_list, mask_path_list, text_list = batch    
        # if index == 20:
        #     break
        messages = []
        gt_masks = []
        images = []
        xy_factors = []
        for (image_path, mask_path, text) in zip(image_path_list, mask_path_list, text_list):
            image = Image.open(image_path)
            original_width, original_height = image.size
            resize_size = 840
            x_factor, y_factor = original_width/resize_size, original_height/resize_size
            xy_factors.append([x_factor, y_factor])
            message = [{
                "role": "user",
                "content": [
                {
                        "type": "image", 
                        "image": image.resize((resize_size, resize_size), Image.BILINEAR) 
                    },
                    {   
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=text.lower().strip("."), 
                                                            Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")
                    }
                ]
            }]
            messages.append(message)
            # gt_mask = Image.open(mask_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_masks.append(gt_mask)
            images.append(image)
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        calculate_giou_batch(
            output_texts, gt_masks, images, intersection_meter, union_meter, acc_iou_meter, xy_factors
        )
        
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    print(ciou, giou)
    

if __name__ == "__main__":
    main()
