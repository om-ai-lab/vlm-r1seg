from datasets import load_dataset, load_from_disk
import json
from tqdm import tqdm
from PIL import Image
import os


# 定义一个函数来增加新键
def add_new_key(data, id2data):
    image_id = data["id"].split("_")[1]
    mask_path = id2data[image_id]
    mask = Image.open(mask_path)
    data["mask"] = mask
    # image = data["image"]
    # image.save("tmp_image.png")
    # mask.save("tmp_mask.png")
    return data


def transform_maskgt_json(mask_json_path):
    maskgt_json = json.load(open(mask_json_path, "r"))
    id2data = dict()
    for data in tqdm(maskgt_json):
        image_id = data["image"].split("/")[-1].split(".")[0]
        mask_path = data["solution"]
        id2data[image_id] = mask_path
    return id2data


def main():
    data_path = "/training/zilun/dataset/refCOCOg_9k_840"
    mask_json_path = "/training/zilun/dataset/vlmr1_refcocog_9k_840.json"
    id2data = transform_maskgt_json(mask_json_path)
    # data_path = "/training/zilun/dataset/ReasonSeg_test"
    save_data_dir = "/training/zilun/dataset/refCOCOg_9k_840_mask/data"
    dataset = load_dataset(data_path)['train']
    dataset = dataset.map(lambda data: add_new_key(data, id2data), num_proc=1)
    os.makedirs(save_data_dir, exist_ok=True)
    num_shards = 10  # set number of files to save (e.g. try to have files smaller than 5GB)
    for shard_idx in range(num_shards):
        shard = dataset.shard(index=shard_idx, num_shards=num_shards)
        shard.to_parquet(f"{save_data_dir}/train-{shard_idx:05d}-of-00010.parquet")

    
if __name__ == "__main__":
    main()
    # dataset = load_dataset("/training/zilun/dataset/ReasonSeg_test")["test"]
    dataset = load_dataset("/training/zilun/dataset/refCOCOg_9k_840_mask")["train"]

    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
        if i == 10:
            break
