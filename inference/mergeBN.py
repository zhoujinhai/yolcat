import torch
import torch.nn as nn
import torchvision as tv


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_conv_and_bn(conv, bn):
    # init
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    # # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fused_conv.weight = nn.Parameter(torch.mm(w_bn, w_conv).view(fused_conv.weight.size()))
    # # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias = nn.Parameter(torch.matmul(w_bn, b_conv) + b_bn)

    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            # bc = fuse(c, child)
            bc = fuse_conv_and_bn(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m):
    p = torch.randn([1, 3, 224, 224])
    import time
    s = time.time()
    o_output = m(p)
    for _ in range(100):
        o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    for _ in range(100):
        f_output = m(p)
    print("Fused time: ", time.time() - s)

    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert(o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


def test_layer():
    p = torch.randn([1, 3, 112, 112])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


if __name__ == "__main__":

    m = tv.models.resnet18(True)
    # print(list(m.named_modules()))
    m.eval()

    # inputs = torch.randn(1, 3, 224, 224)
    # torch.onnx.export(
    #     m,
    #     (inputs,),
    #     "./before.onnx",
    #     verbose=True
    # )
    # print("converted successed!")

    print("Layer level test: ")
    test_layer()

    print("============================")
    print("Module level test: ")
    test_net(m)
    # print(m)
    # print(list(m.named_modules()))
    # torch.onnx.export(
    #     m,
    #     (inputs,),
    #     "./after.onnx",
    #     verbose=True
    # )
    # print("converted successed!")