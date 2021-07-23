_base_ = [
    '../_base_/datasets/semkitti-cylinder-sig-20class.py',
    '../_base_/models/cylinder3d.py',
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

# runtime settings
checkpoint_config = dict(interval=1)

log_config = dict( 
    interval=25, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='WandbLoggerHook', init_kwargs=dict(project='Cylinder3d')), 
    ])

runner = dict(type='EpochBasedRunner', max_epochs=50)