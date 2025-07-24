import os
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import CocoDataset

def visualize(model_cfg, checkpoint, img_path, output_dir, score_thr=0.5):
    """ 生成带检测框的可视化结果 """
    # 初始化模型
    model = init_detector(model_cfg, checkpoint, device='cuda:0')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理单张图片或整个目录
    if os.path.isdir(img_path):
        img_list = [os.path.join(img_path, fn) for fn in os.listdir(img_path) 
                   if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_list = [img_path]
    
    # 逐张推理并保存结果
    for img_file in img_list:
        result = inference_detector(model, img_file)
        out_file = os.path.join(output_dir, os.path.basename(img_file))
        model.show_result(
            img_file, 
            result, 
            score_thr=score_thr,
            show=False, 
            out_file=out_file
        )
        print(f'Saved visualization: {out_file}')

if __name__ == '__main__':
    # 示例调用 (实际路径需替换)
    visualize(
        'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py',
        'work_dirs/mask_rcnn/latest.pth',
        'data/VOCdevkit/VOC2007/JPEGImages/',
        'visualization/'
    )