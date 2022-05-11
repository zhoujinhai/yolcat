import cv2
from PIL import Image
import json
import os
import glob
import numpy as np
import copy
import base64


def encode_base64(file):
    with open(file, "rb") as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        base64_str = str(base64_data, "utf-8")
        return base64_str


def mask_to_json(img_path, mask_path, save_dir):
    print("Deal: ", img_path, mask_path)
    pil_img = Image.open(mask_path)
    img_numpy = np.array(pil_img)
    max_n = np.max(img_numpy)

    cnts = []
    for idx in range(max_n):
        mask = copy.deepcopy(img_numpy)
        mask[mask != idx + 1] = 0
        mask[mask >= 1] = 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_id = 0
        max_len = 0
        for cnt_id, cnt in enumerate(contours):
            if len(cnt) > max_len:
                max_len = len(cnt)
                max_id = cnt_id
        cnts.append(contours[max_id][::8])  # step 10

    file_name = os.path.basename(img_path)
    base64_data = encode_base64(img_path)
    json_data = {"version": "3.16.2", "flags": {}, "shapes": [], "lineColor": [0, 255, 0, 128], "fillColor": [255, 0, 0, 128],
                 "imagePath": file_name, "imageData": base64_data}

    shapes = []
    for idx, cnt in enumerate(cnts):
        shape = dict()
        shape["label"] = "teeth"
        shape["line_color"] = None
        shape["fill_color"] = None
        points = []
        for pt in cnt:
            x = pt[0][0]
            y = pt[0][1]
            points.append([float(x), float(y)])
        shape["points"] = points
        shape["shape_type"] = "polygon"
        shape["flags"] = {}

        shapes.append(shape)
    json_data.update({"shapes": shapes})

    basename = os.path.splitext(file_name)[0]
    save_path = os.path.join(save_dir, basename + ".json")
    with open(save_path, 'w') as wf:
        json.dump(json_data, wf)


if __name__ == "__main__":
    img_dir = r"E:\data\SplitTooth\img\err"
    mask_dir = r"E:\data\SplitTooth\img\results"
    save_dir = r"E:\data\SplitTooth\img\ErrJson"
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))

    for img_path in img_paths:
        basename = os.path.basename(img_path)
        file_name = os.path.splitext(basename)[0]
        mask_path = os.path.join(mask_dir, file_name + "_masks.png")
        if os.path.isfile(mask_path) and os.path.isfile(img_path):
            print(img_path, mask_path)
            mask_to_json(img_path, mask_path, save_dir)








