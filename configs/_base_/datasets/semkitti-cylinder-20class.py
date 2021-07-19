# dataset settings
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
    dict(
        type='ToCylinderDataset',
        grid_size=[480, 360, 32]),
    dict(type='Cylinder3DFormatBundle'),
    dict(type='Collect3D', keys=['voxel_feat', 'pts_semantic_mask', 'grid_ind'])
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
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        semkitti=True,
        seg_3d_dtype='uint32'),
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
                type='ToCylinderDataset',
                grid_size=[480, 360, 32]),
            dict(type='Cylinder3DFormatBundle'),
            dict(type='Collect3D', keys=['voxel_feat', 'grid_ind', 'pts_semantic_mask'])
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
    dict(
        type='ToCylinderDataset',
        grid_size=[480, 360, 32]),
    dict(type='Cylinder3DFormatBundle'),
    dict(type='Collect3D', keys=['voxel_feat', 'grid_ind', 'pts_semantic_mask'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        test_mode=False,
        ignore_index=0,
        pipeline=train_pipeline,
        classes=class_names,
        load_interval=200,
        imageset='train'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        label_mapping="/mmdetection3d-dev/data/semkitti/label-mapping.yaml",
        test_mode=False,
        ignore_index=0,
        pipeline=eval_pipeline,
        classes=class_names,
        load_interval=160,
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

# eval settings
evaluation = dict(pipeline=eval_pipeline, interval=1)