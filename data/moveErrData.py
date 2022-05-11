import os
import shutil


err_txt_path = r"E:\code\Server223\yolact\results\err.txt"
FIND_DIR = [r"E:\data\SplitTooth\img\MeshTest", r"E:\data\SplitTooth\img\MeshTrain"]
SAVE_DIR = r"E:\data\SplitTooth\img\err"


if __name__ == "__main__":
    with open(err_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.find("top") != -1:
                for find_dir in FIND_DIR:
                    file_path = os.path.join(find_dir, line + ".png")
                    if os.path.isfile(file_path):
                        save_path = os.path.join(SAVE_DIR, line + ".png")
                        shutil.copyfile(file_path, save_path)
