# dataset settings
dataset_type = 'SemanticKITTIDataset'
data_root = './data/semkitti/'
class_names = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        load_dim=3,
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        load_dim=3,
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
        use_color=True,
        load_dim=3,
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
    dict(
        type='DefaultFormatBundle3D',
        with_label=False,
        class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        pipeline=train_pipeline,
        classes=class_names,
        imageset='train'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        pipeline=train_pipeline,
        classes=class_names,
        imageset='val'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        pipeline=train_pipeline,
        classes=class_names,
        imageset='test'))

evaluation = dict(pipeline=eval_pipeline)
