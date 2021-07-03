_base_ = [
    '../_base_/datasets/semkitti-20class.py',
    '../_base_/models/pointnet2_ssg.py',
    '../_base_/schedules/seg_cosine_200e.py', '../_base_/default_runtime.py'
]
workflow = [('train', 1)]

# dataset settings
# in this setting, we only use xyz as network input
# so we need to re-write all the data pipeline
dataset_type = 'SemanticKITTIDataset'
data_root = './data/semkitti/dataset/sequences/'
class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=False,
        load_dim=4,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        semkitti=True,
        seg_3d_dtype='uint32'),
    dict(
        type='SemKittiClassMapping',
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml"),
    dict(type='IndoorPointSample', num_points=80000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=False,
        load_dim=4,
        use_dim=[0, 1, 2]),
    dict(
        # a wrapper in order to successfully call test function
        # actually we don't perform test-time-aug
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=False,
        load_dim=4,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        semkitti=True,
        seg_3d_dtype='uint32'),
    dict(
        type='SemKittiClassMapping',
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml"),
    # dict(type='IndoorPointSample', num_points=16384),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        test_mode=False,
        ignore_index=0,
        pipeline=train_pipeline,
        classes=class_names,
        imageset='train'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        test_mode=False,
        ignore_index=0,
        pipeline=test_pipeline,
        classes=class_names,
        imageset='val'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        test_mode=True,
        ignore_index=0,
        pipeline=test_pipeline,
        classes=class_names,
        imageset='test'))

evaluation = dict(pipeline=eval_pipeline, interval=4)

# model settings
model = dict(
    backbone=dict(in_channels=3),  # only [xyz]
    decode_head=dict(
        num_classes=20,
        ignore_index=0,
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        # loss_decode=dict(class_weight=[
        #     2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
        #     4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
        #     5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
        #     5.3954206, 4.6971426
        # ])
        ),
    test_cfg=dict(
        num_points=80000,
        use_normalized_coord=False,
        mode='whole',
        batch_size=8))

# runtime settings
checkpoint_config = dict(interval=5)
