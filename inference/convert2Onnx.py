import torch
from yolact_ import Yolact
from data import set_cfg
import copy
import os
import cv2
import time


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv


class IdentityModule(torch.nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x


def fuse_model(m):
    # inp = torch.randn(1, 3, 550, 550).to(device)
    # s = time.time()
    # o_output = m(inp)
    # for _ in range(100):
    #     o_output = m(inp)
    # print("Original time: ", time.time() - s)

    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, torch.nn.BatchNorm2d):
            # bc = fuse(c, child)
            bc = fuse_conv_bn_eval(c, child)
            m._modules[cn] = bc
            m._modules[name] = IdentityModule()
            c = None
        elif isinstance(child, torch.nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_model(child)


def benchmark(model, inp, iters=10):
    model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str((time.time()-begin)/iters)


def test_diff(o_output, f_output):
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert(o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", torch.nn.MSELoss()(o_output, f_output).item())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_path = "../weights/20220613_batch8/tooth_272_40000.pth"
    set_cfg("yolact_resnet50_tooth_config")

    net = Yolact()
    net.load_weights(weight_path)
    net.eval()
    net.to(device)

    inputs = torch.randn(1, 3, 550, 550).to(device)
    ori_out = net(inputs)
    print("origin time: ****: ", benchmark(net, inputs))

    # fusing convolution with batch norm
    fuse_model(net)
    after_out = net(inputs)
    print("after time: ****: ", benchmark(net, inputs))

    test_diff(ori_out[2], after_out[2])  # mask

    # print(net)

    # weight_name = os.path.splitext(os.path.basename(weight_path))[0]
    # onnx_model_path = weight_name + "_.onnx"
    #
    # inputs = torch.randn(1, 3, 550, 550).to(device)
    # print("convert net to ", onnx_model_path, " ... ")
    # torch.onnx.export(
    #     net,
    #     (inputs,),
    #     onnx_model_path,
    #     verbose=True,
    #     input_names=["img"],
    #     output_names=["loc", "conf", "mask", "proto"],
    #     opset_version=12
    # )
    # print("converted successed!")
    #
    # try:
    #     dnn_net = cv2.dnn.readNet(onnx_model_path)
    #     print("cv read onnx successed!")
    # except:
    #     print("cv read onnx failed!")

