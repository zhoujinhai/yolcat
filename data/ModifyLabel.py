import os
import glob
import json


JSON_DIR = r"\\10.99.11.210\Backups(d)\MeshCNN\splitData\MeshTrain\test"
SAVE_DIR = r"\\10.99.11.210\Backups(d)\MeshCNN\splitData\MeshTrain\testSave"


if __name__ == "__main__":
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    for json_file in json_files:
        file_name = os.path.splitext(os.path.basename(json_file))[0]
        print(file_name)
        with open(json_file, "r") as f:
            json_data = json.load(f)
            print(json_data.keys())
            shapes = json_data["shapes"]
            for idx, shape in enumerate(shapes):
                label = shape["label"]
                shape.update({"label": idx+1})
            save_path = os.path.join(SAVE_DIR, file_name + ".json")
            with open(save_path, 'w') as wf:
                json.dump(json_data, wf)