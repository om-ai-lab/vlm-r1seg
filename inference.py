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
import hydra

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
import re
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="/home/zilun/Documents/R1-Seg/checkpoint/Qwen2.5-VL-3B-GRPO-R1SEG")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--reasonseg_json", type=str, default="/home/zilun/Documents/R1-Seg/dataset/reasonseg_test.json")
    # parser.add_argument("--text", type=str, default="the unusal object in the image")
    # parser.add_argument("--image_path", type=str, default="./assets/test_image.png")
    # parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output.png")
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


def mask_iou_reward(completions, solution, image):
    def calculate_miou(pred_mask, true_mask, num_classes) -> object:
        """
        计算给定预测掩膜和真实标签掩膜的mIoU。

        参数:
            pred_mask (numpy.ndarray): 预测的分割掩膜，形状为[H, W]，其中H是高度，W是宽度。
            true_mask (numpy.ndarray): 真实标签的分割掩膜，形状与pred_mask相同。
            num_classes (int): 类别的数量。

        返回:
            float: 平均交并比(mIoU)。
        """
        ious = []
        for cls in range(1, num_classes):  # 假设0是背景类，不计入计算
            pred_inds = pred_mask == cls
            target_inds = true_mask == cls

            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()

            if union == 0:
                ious.append(float('nan'))  # 如果这个类别在图像中不存在，则跳过
            else:
                ious.append(intersection / union)
        return np.nanmean(ious)  # 忽略NaN值计算平均

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol, img in zip(contents, solution, image):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)),
                            int(bbox_match.group(4))]
                    input_box = np.array(bbox)
                    masks = predict_sam2(img, input_box, None, None)
                    ground_truth_mask = np.array(Image.open(sol).convert('L')) // 255
                    reward = calculate_miou(masks[0], ground_truth_mask, 2)
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards



def main():
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

    # segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    # checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    # segmentation_model = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox. " \
        "Compare the difference between objects and find the most closely matched one. " \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format. " \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"

    for image_path, mask_path, text in tqdm(zip(image_path_list, mask_path_list, text_list)):
        image = Image.open(image_path)
        original_width, original_height = image.size
        resize_size = 840
        x_factor, y_factor = original_width / resize_size, original_height / resize_size

        messages = []
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
                                                     Answer="{'bbox': [10,100,200,210]}")
                }
            ]
        }]
        print(message)
        messages.append(message)

        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # pdb.set_trace()
        image_inputs, video_inputs = process_vision_info(messages)
        # pdb.set_trace()
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
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        solution = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mious = mask_iou_reward(output_text, solution, image)

        # bbox, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
        #
        # print("Thinking process: ", think)
        #
        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     segmentation_model.set_image(image)
        #     masks, scores, _ = segmentation_model.predict(
        #         # point_coords=points,
        #         # point_labels=[1, 1],
        #         box=bbox
        #     )
        #     sorted_ind = np.argsort(scores)[::-1]
        #     masks = masks[sorted_ind]
        #
        # mask = masks[0].astype(bool)


if __name__ == "__main__":
    main()
