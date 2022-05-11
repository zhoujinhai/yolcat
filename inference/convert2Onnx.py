import torch
from yolact_ import Yolact
from data import set_cfg
import os
import cv2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_path = "../weights/20200428_batch4/tooth_754_80000.pth"
    set_cfg("yolact_resnet50_tooth_config")

    net = Yolact()
    net.load_weights(weight_path)
    net.eval()
    net.to(device)

    weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    onnx_model_path = weight_name + ".onnx"

    inputs = torch.randn(1, 3, 550, 550).to(device)
    print("convert net to ", onnx_model_path, " ... ")
    torch.onnx.export(
        net,
        (inputs,),
        onnx_model_path,
        verbose=True,
        input_names=["img"],
        output_names=["loc", "conf", "mask", "proto"],
        opset_version=12
    )
    print("converted successed!")

    try:
        dnn_net = cv2.dnn.readNet(onnx_model_path)
        print("cv read onnx successed!")
    except:
        print("cv read onnx failed!")

