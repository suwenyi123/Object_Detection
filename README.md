# 基于VOC数据训练的Mask R-CNN和Sparse R-CNN目标检测模型
## 官方开源库
`https://github.com/open-mmlab/mmdetection`
## 项目简介
利用MMdetection的开源框架Mask R-CNN和Sparse R-CNN，用VOC2007数据集来进行训练，实现目标检测和实例分割功能
## 安装与依赖
- `python==3.8`
- `cuda==12.6`
- 除mmcv的各类库的安装参考官方文档链接：`https://mmdetection.readthedocs.io/en/latest/get_started.html`
- mmcv按照官方的命令安装会安装最新版，出现报错，建议指定版本`mmcv=2.1.0`（或2.0.0）
## 数据集准备
下载VOC数据集并且使用MMdetection开源脚本将数据转化为COCO格式
## 模型训练
### 配置文件
- Mask：
`mmdetection-main/configs/mask_rcnn/Untuned_mask_rcnn_r50_fpn_1x_voc.py`
- Sparse：
`mmdetection-main/configs/sparse_rcnn/Untuned_sparse-rcnn_r50_fpn_1x_voc.py`
### 模型文件
修改Sparse模型使其增加分割功能，主要修改的模型文件：
`mmdetection-main/mmdet/models/roi_heads/sparse_roi_head_with_mask.py`
`mmdetection-main/mmdet/models/roi_heads/__init__.py`
`mmdetection-main/mmdet/models/roi_heads/sparse_roi_head.py`
### 训练脚本
#### 单GPU
- Mask：
`python tools/train.py configs/mask_rcnn/Untuned_mask_rcnn_r50_fpn_1x_voc.py --work-dir work_dirs/Untuned_mask-rcnn_voc`

文件夹`work_dirs/Untuned_mask-rcnn_voc`用于保存每次训练的日志文件，方便之后的可视化
- Sparse：
`python tools/train.py configs/sparse_rcnn/Untuned_sparse_rcnn_r50_fpn_1x_voc.py --work-dir work_dirs/Untuned_sparse-rcnn_voc` 

文件夹`work_dirs/Untuned_sparse-rcnn_voc`用于保存每次训练的日志文件，方便之后的可视化
#### 多GPU
使用脚本：`tools/dist_train.sh`
### 测试脚本
#### 单GPU
- Mask:
`python tools/test.py configs/mask_rcnn/Untuned_mask_rcnn_r50_fpn_1x_voc.py work_dirs/Untuned_mask-rcnn_voc/best_coco_segm_mAP_epoch_11.pth --out work_dirs/test_Untuned_mask-rcnn_voc`

文件夹`work_dirs/test_Untuned_mask-rcnn_voc`用于保存每次训练的日志文件，方便之后的可视化

- Sparse:
`python tools/test.py configs/sparse_rcnn/Untuned_sparse_rcnn_r50_fpn_1x_voc.py /work_dirs/Untuned_sparse-rcnn-mask/models/Untuned_sparse-rcnn_voc/epoch_36.pth --out work_dirs/test_Untuned_sparse-rcnn_voc`

文件夹`work_dirs/Untuned_mask-rcnn_voc`用于保存每次训练的日志文件，方便之后的可视化

#### 多GPU
使用脚本：`tools/dist_test.sh`

## 可视化结果
使用tensorboard进行acc，loss等指标的可视化
`tensorboard --logdir work_dirs/.../时间戳(如20250528_043603) --port=6006`
