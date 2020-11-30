# -*- coding: utf-8 -*-

from os.path import join
import pandas as pd
import math


class Config(object):
    """
    define a class to store parameters,
    """

    def __init__(self):
        # common information
        self.pm = 'gaussian'
        self.feature_path = "Feature_Path"
        self.train_split_set = 'val'
        # self.test_split_set = 'val'
        self.test_split_set = 'test'
        self.window_size = 512
        self.batch_size = 64
        self.input_steps = 128
        self.num_classes = 21
        self.class_real = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33,
                           36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
        self.layers_name = ['AL1', 'AL2', 'AL3']
        self.scale = {'AL1': 1. / 32, 'AL2': 1. / 16, 'AL3': 1. / 8}
        self.num_anchors = {'AL1': 32, 'AL2': 16, 'AL3': 8}
        self.aspect_ratios = {'AL1': [1],
                              'AL2': [1],
                              'AL3': [1]}
        self.num_dbox = {'AL1': 1, 'AL2': 1, 'AL3': 1}

        self.learning_rates = [0.0001] * 30 + \
            [0.00001] * 4  # the 31th epoch is crucial
        self.training_epochs = len(self.learning_rates)
        self.p_class = 1
        self.p_loc = 10
        self.p_conf = 10
        self.negative_ratio = 1
        self.seed = 1129

        self.nms_threshold = 0.2
        # when process results, remove confident negative anchors by previous
        self.filter_neg_threshold = 0.98
        # when process results, remove confident low overlap (conf) anchors by previous
        self.filter_conf_threshold = 0.1
        # used in load_data.py window_data function to choose window
        self.overlap_ratio_threshold = 0.9

        self.save_predict_result = True
        self.initialize = True
        # True: train from scratch
        # (or delete the corresponding params files in models_dir);
        # False: restore from pretrained model
        self.steps = 30  # defined by steps

        self.outdf_columns = ['video_name', 'start', 'conf', 'xmin', 'xmax', 'score_0', 'score_1', 'score_2',
                              'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'score_8',
                              'score_9', 'score_10', 'score_11', 'score_12', 'score_13', 'score_14',
                              'score_15', 'score_16', 'score_17', 'score_18', 'score_19', 'score_20']


def get_anno_ath(split_set):
    return join('data', 'thumos14', split_set)


def get_anno_df(anno_path, split_set):
    return pd.read_csv(join(anno_path, 'thumos14_' + split_set + '_annotation.csv'))


def get_data_x_path(feature_path, split_set, mode, data_x_type):
    return join(feature_path, split_set, mode + 'DataX' + data_x_type)


def get_models_dir(mode, pretrain_dataset, method):
    return join('./model/', mode + '_' + pretrain_dataset + '_' + method)


def get_predict_result_path(mode, pretrain_dataset, method):
    return join('./result/', 'predict_' + mode + '_' + pretrain_dataset + '_' + method + '.csv')
