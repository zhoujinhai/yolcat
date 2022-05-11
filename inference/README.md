### 细节

#### 1、维度信息

```python
net = Yolact()
print("net: ", net)
net.train()
x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
y = net(x)

# FPN后输入预测网络的P3---p7特征层
for p in net.prediction_layers:
    print(p.last_conv_size)

# 输出维度
for k, a in y.items():
    print(k + ': ', a.size(), torch.sum(a))
 
# out
"""
(69, 69)
(35, 35)
(18, 18)
(9, 9)
(5, 5)

loc:  torch.Size([1, 19248, 4]) tensor(-5334.4219, grad_fn=<SumBackward0>)
conf:  torch.Size([1, 19248, 81]) tensor(-1648.6711, grad_fn=<SumBackward0>)
mask:  torch.Size([1, 19248, 32]) tensor(-24752.4023, grad_fn=<SumBackward0>)
priors:  torch.Size([19248, 4]) tensor(21850.8438)
proto:  torch.Size([1, 138, 138, 32]) tensor(59765.8398, grad_fn=<SumBackward0>)
segm:  torch.Size([1, 80, 69, 69]) tensor(15391.8555, grad_fn=<SumBackward0>)
"""    
```

#### 2、候选框生成

参考`yolact.py`中`mask_priors`函数

```python
def make_priors(self, conv_h, conv_w, device):
    """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
```

#### 3 、 掩膜解析

参考`layers/output_utils.py`中`postprocess`函数

mask乘上相应系数，再缩放回原图大小

```python
def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    ...
    proto_data = dets['proto']
    masks = proto_data @ masks.t()
    masks = cfg.mask_proto_mask_activation(masks)
    ...
    
    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()
    
    # Scale masks up to the full image
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)
```

#### 4、结果解析

参考`yolact.py`文件中`Detect`函数 以及 `eval.py`中`prep_display`函数

```python
# yolact.py
# For use in evaluation
self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
                     conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

# prep_display.py
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
```

