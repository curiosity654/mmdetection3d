import numpy as np
import tempfile
import os
import warnings
from os import path as osp

from torch.utils.data.dataset import Dataset

from mmdet3d.core import show_seg_result
from mmdet.datasets import DATASETS
from mmseg.datasets import DATASETS as SEG_DATASETS
from .pipelines import Compose

from .utils import extract_result_dict, get_loading_pipeline

def absoluteFilePaths(directory):
    res = []
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            res.append(os.path.abspath(os.path.join(dirpath, f)))
            # yield os.path.abspath(os.path.join(dirpath, f))
    return res
@DATASETS.register_module()
@SEG_DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    r"""SemanticKITTI Dataset.

    This class serves as the API for experiments on the SemanticKITTI Dataset
    Please refer to <http://www.semantic-kitti.org/dataset.html>`_
    for data downloading

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    CLASSES = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign')
    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]

    def __init__(self,
                 data_root,
                 label_mapping,
                 test_mode=False,
                 return_ref=False,
                 pipeline=None,
                 classes=None,
                 load_interval=1,
                 palette=None,
                 ignore_index=None,
                 imageset='train'):

        # super().__init__(
        #     data_root=data_root,
        #     ann_file=None,
        #     pipeline=pipeline,
        #     classes=classes,
        #     palette=palette,
        #     modality=None,
        #     test_mode=test_mode,
        #     #TODO add ignore index
        #     ignore_index=None,
        #     #TODO scene idx
        #     scene_idxs=None)

        import yaml
        self.data_root=data_root
        self.return_ref=return_ref
        self.imageset=imageset
        self.pipeline = pipeline
        with open(label_mapping, 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)

        self.learning_map = self.semkittiyaml['learning_map']
        self.test_mode = test_mode
        self.load_interval = load_interval

        if imageset == 'train':
            split = self.semkittiyaml['split']['train']
        elif imageset == 'val':
            split = self.semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = self.semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        # import ipdb
        # ipdb.set_trace()

        self.data_infos = []
        for i_folder in split:
            directory = '/'.join([data_root, str(i_folder).zfill(2), 'velodyne'])
            for dirpath, _, filenames in os.walk(directory):
                filenames.sort()
                for f in filenames:
                    data_info = dict(
                        pts_path = os.path.abspath(os.path.join(dirpath, f))
                    )
                    self.data_infos.append(data_info)

        self.data_infos = self.data_infos[::self.load_interval]
    
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.label2cat = {
            i: cat_name
            for i, cat_name in enumerate(self.CLASSES)
        }
        self.ignore_index = ignore_index

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_infos)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        pts_filename = self.data_infos[index]['pts_path']

        input_dict = dict(
            pts_filename=pts_filename,
            file_name=pts_filename
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api

        pts_semantic_mask_path = self.data_infos[index]['pts_path'].replace('velodyne', 'labels')[:-3] + 'label'

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
        """
        results['img_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['bbox3d_fields'] = []

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            pts_path = self.data_infos[i]
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)
        while True:
            data = self.prepare_train_data(index)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
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
            dict(type='IndoorPointSample', num_points=16384),
            dict(type='DefaultFormatBundle3D', class_names=self.CLASSES),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict] | None): Input pipeline. If None is given, \
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in semantic segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import seg_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._get_pipeline(pipeline)
        pred_sem_masks = [result['semantic_mask'] for result in results]
        gt_sem_masks = [
            self._extract_data(
                i, load_pipeline, 'pts_semantic_mask', load_annos=True)
            for i in range(len(self.data_infos))
        ]
        ret_dict = seg_eval(
            gt_sem_masks,
            pred_sem_masks,
            self.label2cat,
            self.ignore_index,
            logger=logger)

        if show:
            self.show(pred_sem_masks, out_dir, pipeline=pipeline)

        return ret_dict

        # raw_data = np.fromfile(self.data_infos[index], dtype=np.float32).reshape((-1, 4))
        # if self.imageset == 'test':
        #     annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        # else:
        #     annotated_data = np.fromfile(self.data_infos[index].replace('velodyne', 'labels')[:-3] + 'label',
        #                                  dtype=np.uint32).reshape((-1, 1))
        #     annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        #     annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        # data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        # if self.return_ref:
        #     data_tuple += (raw_data[:, 3],)
        # return data_tuple

    # def get_scene_idxs(self, scene_idxs):
    #     """Compute scene_idxs for data sampling.

    #     We sample more times for scenes with more points.
    #     """
    #     # when testing, we load one whole scene every time
    #     if not self.test_mode and scene_idxs is None:
    #         raise NotImplementedError(
    #             'please provide re-sampled scene indexes for training')

    #     return super().get_scene_idxs(scene_idxs)

    # def format_results(self, results, txtfile_prefix=None):
    #     r"""Format the results to txt file. Refer to `ScanNet documentation
    #     <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

    #     Args:
    #         outputs (list[dict]): Testing results of the dataset.
    #         txtfile_prefix (str | None): The prefix of saved files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.

    #     Returns:
    #         tuple: (outputs, tmp_dir), outputs is the detection results,
    #             tmp_dir is the temporal directory created for saving submission
    #             files when ``submission_prefix`` is not specified.
    #     """
    #     import mmcv

    #     if txtfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         txtfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None
    #     mmcv.mkdir_or_exist(txtfile_prefix)

    #     # need to map network output to original label idx
    #     pred2label = np.zeros(len(self.VALID_CLASS_IDS)).astype(np.int)
    #     for original_label, output_idx in self.label_map.items():
    #         if output_idx != self.ignore_index:
    #             pred2label[output_idx] = original_label

    #     outputs = []
    #     for i, result in enumerate(results):
    #         info = self.data_infos[i]
    #         sample_idx = info['point_cloud']['lidar_idx']
    #         pred_sem_mask = result['semantic_mask'].numpy().astype(np.int)
    #         pred_label = pred2label[pred_sem_mask]
    #         curr_file = f'{txtfile_prefix}/{sample_idx}.txt'
    #         np.savetxt(curr_file, pred_label, fmt='%d')
    #         outputs.append(dict(seg_mask=pred_label))

    #     return outputs, tmp_dir