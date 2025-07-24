import argparse
import os
import torch
import mmcv
from mmdet.apis import init_detector, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    # 创建输出目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 构建模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # 分布式训练设置 (支持单机多卡)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 启动训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,  # 单机训练
        validate=True,       # 每个epoch后验证
        timestamp=None
    )
    
    # 保存最终权重
    final_path = os.path.join(cfg.work_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')

if __name__ == '__main__':
    main()