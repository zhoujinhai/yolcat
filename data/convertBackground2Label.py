import cv2
from PIL import Image
import json
import os
import glob
import numpy as np
import copy
import base64
import shutil


def encode_base64(file):
    with open(file, "rb") as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        base64_str = str(base64_data, "utf-8")
        return base64_str


def modify_json_file(json_file, img_path, save_dir):
    with open(json_file, "r") as f:
        json_data = json.load(f)
        print("deal", json_file, json_data.keys())

        file_name = os.path.splitext(os.path.basename(img_path))[0]
        base64_data = encode_base64(img_path)

        json_data.update({"imagePath": file_name + "_black"})
        json_data.update({"imageData": base64_data})

        save_path = os.path.join(save_dir, file_name + "_black.json")
        with open(save_path, 'w') as wf:
            json.dump(json_data, wf)


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
        if len(contours) > 0:
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
    save_dir = r"E:\data\SplitTooth\img\MeshScan\blackJson"
    json_img_dir = r"E:\data\SplitTooth\img\MeshScan\adjustJson"
    black_img_dir = r"E:\data\SplitTooth\img\MeshScan\adjustBlack"
    img_paths = glob.glob(os.path.join(json_img_dir, "*.png"))

    for img_path in img_paths:
        basename = os.path.basename(img_path)
        file_name = os.path.splitext(basename)[0]
        black_img_path = os.path.join(black_img_dir, file_name + ".png")
        img_save_path = os.path.join(save_dir, file_name + "_black.png")
        json_path = os.path.join(json_img_dir, file_name + ".json")
        if os.path.isfile(black_img_path) and os.path.isfile(json_path):
            modify_json_file(json_path, black_img_path, save_dir)
            shutil.copyfile(black_img_path, img_save_path)








