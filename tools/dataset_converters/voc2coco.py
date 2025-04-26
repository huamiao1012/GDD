
# -*- coding: utf-8 -*-

import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
xml_dir = '/zhdd/dataset/DAOD/clipart/Annotations/'  # VOC xml �ļ���
img_dir = '/zhdd/dataset/DAOD/clipart/JPEGImages/'       # ��Ӧ��ͼ���ļ���
output_json = '/zhdd/dataset/DAOD/clipart/train.json'
train_list_path = '/zhdd/dataset/DAOD/clipart/ImageSets/Main/train.txt'
test_list_path = '/zhdd/dataset/DAOD/clipart/ImageSets/Main/test.txt'

# �����ռ���Ϣ
category_set = {}
categories = []
images = []
annotations = []
image_set = {}

image_id = 1
annotation_id = 1

def add_category(name):
    if name not in category_set:
        new_id = len(category_set) + 1
        category_set[name] = new_id
        categories.append({'id': new_id, 'name': name})
    return category_set[name]

def convert_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax - xmin, ymax - ymin]
image_ids = []
# ��ȡ train.txt �е�ͼ�� ID��������չ����
for list_path in [train_list_path, test_list_path]:
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

