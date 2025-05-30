_base_ = [
    '../_base_/datasets/voc_sparse.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

num_stages = 6
num_proposals = 100

model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,  # 加上pad_mask支持mask训练
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        # init_cfg=None
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5  # Mask R-CNN一般用5个输出层，支持mask
    ),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]
    
        # 增加mask分支
        #mask_roi_extractor=dict(
        #    type='SingleRoIExtractor',
        #    roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
        #    out_channels=256,
        #    featmap_strides=[4, 8, 16, 32]),
        #mask_head=dict(
        #    type='FCNMaskHead',
        #    num_convs=4,
        #    in_channels=256,
        #    conv_out_channels=256,
        #    num_classes=20,  
        #    loss_mask=dict(
        #        type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28  # 加上mask_size以支持mask训练
            ) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=num_proposals,
            mask_thr_binary=0.5  # 分割的二值化阈值
        ))
)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2))

# ---------- 可视化配置 ----------
vis_backends = [
    dict(type='LocalVisBackend'),  # 保存本地可视化结果
    dict(type='TensorboardVisBackend'),  # 启用TensorBoard
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)  # 增大滑动窗口

log_config = dict(
    interval=10,  # 每10个iter记录一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# 评估配置
evaluation = dict(
    interval=1,  # 每1个epoch评估一次验证集
    metric=['bbox', 'segm'],
    save_best='coco/segm_mAP',
    classwise=True,
    do_final_eval=False
)

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),  # 与log_config保持一致
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/segm_mAP', out_dir='./work_dirs/Untuned_sparse-rcnn-mask/models'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        test_out_dir='vis_results'
    ),
)