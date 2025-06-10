
# -*- coding: utf-8 -*-

import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
xml_dir = '/zhdd/dataset/DAOD/night_sunny/VOC2007/Annotations/'  # VOC xml �ļ���
img_dir = '/zhdd/dataset/DAOD/night_sunny/VOC2007/JPEGImages/'       # ��Ӧ��ͼ���ļ���
output_json = '/zhdd/dataset/DAOD/night_sunny/VOC2007/all.json'
#train_list_path = '/zhdd/dataset/DAOD/daytime_clear/VOC2007/ImageSets/Main/train.txt'
test_list_path = '/zhdd/dataset/DAOD/night_sunny/VOC2007/ImageSets/Main/train.txt'
categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "truck"},
    {"id": 3, "name": "person"},
    {"id": 4, "name": "rider"},
    {"id": 5, "name": "bike"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "motor"},
]
# �����ռ���Ϣ
# categories = [
#     {"id": 1, "name": "chair"},
#     {"id": 2, "name": "car"},
#     {"id": 3, "name": "horse"},
#     {"id": 4, "name": "person"},
#     {"id": 5, "name": "bicycle"},
#     {"id": 6, "name": "cat"},
#     {"id": 7, "name": "dog"},
#     {"id": 8, "name": "train"},
#     {"id": 9, "name": "aeroplane"},
#     {"id": 10, "name": "diningtable"},
#     {"id": 11, "name": "tvmonitor"},
#     {"id": 12, "name": "bird"},
#     {"id": 13, "name": "bottle"},
#     {"id": 14, "name": "motorbike"},
#     {"id": 15, "name": "pottedplant"},
#     {"id": 16, "name": "boat"},
#     {"id": 17, "name": "sofa"},
#     {"id": 18, "name": "sheep"},
#     {"id": 19, "name": "cow"},
#     {"id": 20, "name": "bus"}
# ]
# 生成类别名字到id的映射
category_set = {cat["name"]: cat["id"] for cat in categories}
 
images = []
annotations = []
image_set = {}

image_id = 1
annotation_id = 1

def add_category(name):
    if name in category_set:
        return category_set[name]
    else:
        raise ValueError(f"❗出现了未定义的类别: {name}")

def convert_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax - xmin, ymax - ymin]
image_ids = []
# ��ȡ train.txt �е�ͼ�� ID��������չ����
for list_path in [test_list_path]:
    with open(list_path, 'r') as f:
        lines = f.readlines()
        image_ids.extend([line.strip().replace('.jpg', '').replace('.png', '') for line in lines])

for img_name in tqdm(image_ids):
    xml_file = os.path.join(xml_dir, img_name + '.xml')
    img_file = os.path.join(img_dir, img_name + '.jpg')

    if not os.path.exists(xml_file):
        print(f"Missing XML: {xml_file}")
        continue
    if not os.path.exists(img_file):
        print(f"Missing image: {img_file}")
        continue

    tree = ET.parse(xml_file)
    root = tree.getroot()
 
    img = Image.open(img_file)
    width, height = img.size

    images.append({
        "id": image_id,
        "file_name": img_name + '.jpg',
        "width": width,
        "height": height
    })
    image_set[img_name] = image_id

    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = add_category(name)

        bndbox = obj.find('bndbox')
        bbox = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text)
        ]
        coco_bbox = convert_bbox(bbox)

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": coco_bbox,
            "area": coco_bbox[2] * coco_bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1

    image_id += 1

# ����Ϊ COCO ��ʽ
coco_json = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, 'w') as f:
    json.dump(coco_json, f, indent=4)

print(f"complete")

