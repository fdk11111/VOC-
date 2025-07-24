_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 模型配置 (20类VOC物体+背景)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=21),  # VOC 20类 + 背景
        mask_head=dict(num_classes=21))
)

# 优化器配置 (匹配实验参数)
optimizer = dict(
    type='SGD',
    lr=0.0025,  # 实验报告指定值
    momentum=0.9,
    weight_decay=0.0001)

# 训练时长配置
runner = dict(type='EpochBasedRunner', max_epochs=12)  # 12 epochs

# 学习率策略 (每8epoch下降10倍)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])