from mmdet.models.roi_heads import SparseRoIHead
from mmdet.structures.bbox import bbox2roi
from mmdet.registry import MODELS
import torch


@MODELS.register_module()
class SparseRoIHeadWithMask(SparseRoIHead):
    def _mask_forward(self, stage, x, rois, attn_feats= None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
        mask_preds = mask_head(mask_feats)
        return dict(mask_preds=mask_preds)


    def mask_loss(self, stage, x, bbox_results, batch_gt_instances, rcnn_train_cfg):
        attn_feats = bbox_results['attn_feats']
        sampling_results = bbox_results['sampling_results']

        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        attn_feats = torch.cat([
            feats[res.pos_inds]
            for feats, res in zip(attn_feats, sampling_results)
        ])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_head = self.mask_head[stage]
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        mask_results.update(mask_loss_and_target)


        return mask_results
