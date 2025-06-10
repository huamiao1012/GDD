_base_ = [
    '../../_base_/models/deformable_detr.py',
    '../../_base_/dg_setting/detr_20k.py',
    '../../_base_/datasets/dwd/dwd.py'
]

detector = _base_.model
detector.bbox_head.num_classes = 7
#detector.backbone.dift_config.scheduler_timesteps = [100 * 5, 80 * 5, 60 * 5, 40 * 5, 20 * 5]
train_cfg = dict(val_interval=2000)