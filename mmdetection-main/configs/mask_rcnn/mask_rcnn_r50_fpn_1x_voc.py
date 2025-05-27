_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ----------------------- 数据集配置 -----------------------
dataset_type = 'CocoDataset'
data_root = ''  # 移除末尾斜杠

metainfo = dict(
    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
)

# 训练数据加载器
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=f'data/VOCdevkit/annotations/voc07_train.json',
        data_root=data_root,
        data_prefix=dict(img=''),  
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
    )
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=f'data/VOCdevkit/annotations/voc07_val.json',
        data_root=data_root,
        data_prefix=dict(img=''),  # 关键修改：移除VOCdevkit
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='PackDetInputs')
        ],
        test_mode=True
    )
)

# 测试数据加载器
test_dataloader = val_dataloader

# ----------------------- 评估器配置 -----------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'data/VOCdevkit/annotations/voc07_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
)

test_evaluator = val_evaluator

# ----------------------- 训练配置 -----------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# 模型配置：修改类别数为20
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# 加载预训练模型
load_from = 'checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

vis_backends = [
    dict(type='LocalVisBackend'),  # 保存本地可视化结果
    dict(type='TensorboardVisBackend'),  # 启用TensorBoard
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_config = dict(
    interval=50,  # 每50个iter记录一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),  # 关键：启用TensorBoard
    ]
)