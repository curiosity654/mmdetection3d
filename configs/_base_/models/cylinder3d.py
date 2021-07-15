# model settings
model = dict(
    type='Cylinder3D',
    backbone=dict(
        type='CylinderFeatureGenerator',
        grid_size=[480, 360, 32]),
    decode_head=dict(
        type='Cylinder3DHead',
        train_batch_size=1,
        eval_batch_size=1,
        output_shape=[480, 360, 32]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())