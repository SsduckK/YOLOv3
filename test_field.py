import os.path as op
import json
import cv2

from glob import glob

total_path = "/mnt/intHDD/cityscapes"
image_path = op.join(total_path, "leftImg8bit", "train")
annotation_file = op.join(total_path, "annotations", "instancesonly_filtered_gtFine_train.json")

with open(annotation_file) as f:
    data = json.load(f)

images = data["images"]
category_anno = data["categories"]
annotations = data["annotations"]


def get_image_annotations(ann, id):
    image_anno = []
    for anno in ann:
        if anno["image_id"] == id:
            image_anno.append(anno)
    return image_anno


def convert_id2cat(category_id, category_anno):
    for i in category_anno:
        if i["id"] == category_id:
            return i["name"]


def make_cate_bbox(image, annotations, category_anno):
    dst_img = image.copy()
    dictioinarized = {}
    object_index = 0
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        category = convert_id2cat(category_id, category_anno)
        dst_img = cv2.rectangle(dst_img, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0))
        dst_img = cv2.putText(dst_img, category, (int(bbox[0]), int(bbox[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 255, 0), 1)
        dictioinarized[object_index] = [category, int(bbox[0]), int(bbox[1]),
                                        int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), 1]
        object_index += 1
    return dst_img, dictioinarized


for image in images:
    image_name = image["file_name"].replace("png", "json").split("/")[-1]
    print(image_name)
    img = cv2.imread(op.join(image_path, image["file_name"]))
    image_id = image["id"]
    image_annotations = get_image_annotations(annotations, image_id)
    img, dict_file = make_cate_bbox(img, image_annotations, category_anno)
    cv2.imshow("img", img)
    cv2.waitKey()
    # with open("/mnt/intHDD/cityscapes/annotations_json/" + image_name, 'w') as json_file:
    #     json.dump(dict_file, json_file, indent=4)
