# -*- coding: utf-8 -*-


import sys
import time
from os.path import join

import tensorflow as tf

from config import Config, get_models_dir
from operation import *
from load_data import get_test_data, get_train_data, get_val_data
from tensorboardX import SummaryWriter

####################################### PARAMETERS ########################################

stage = sys.argv[1]  # train/test/fuse/train_test_fuse
pretrain_dataset = sys.argv[2]  # UCF101/KnetV3
mode = sys.argv[3]  # temporal/spatial
method = sys.argv[4]
method_temporal = sys.argv[5]  # used for final result fusing

if (mode == 'spatial' and pretrain_dataset == 'Anet') or pretrain_dataset == 'KnetV3':
    feature_dim = 2048
else:
    feature_dim = 1024


models_dir = get_models_dir(mode, pretrain_dataset, method)
models_file_prefix = join(models_dir, 'RBC-ep')
test_checkpoint_file = join(models_dir, 'RBC-ep-30')
predict_file = get_predict_result_path(mode, pretrain_dataset, method)


def get_predict_result_path(mode, pretrain_dataset, method):
    return join('./result/', 'RBC_' + mode + '_' + pretrain_dataset + '_' + method + '.csv')

######################################### TRAIN ##########################################


def train_operation(X, Y_label, Y_bbox, Index, LR, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    MALs = main_anchor_layer(net)

    # --------------------------- Main Stream -----------------------------
    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    full_mainAnc_BM_x = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_w = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_labels = tf.reshape(
        tf.constant([], dtype=tf.int32), [bsz, -1, ncls])
    full_mainAnc_BM_scores = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_BM_scores_d = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        num_anch = config.num_anchors[ln]
        mainAnc_rx, mainAnc_rw = anchor_box_default(config, ln)
        with tf.variable_scope("sigma_layer" + ln + 'mainStream'):
            sigma = tf.layers.conv1d(inputs=MALs[i], filters=config.num_dbox[ln], activation=tf.nn.sigmoid,
                                     kernel_size=3, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=5))
            sigma = tf.reshape(
                sigma, [config.batch_size, config.num_anchors[ln]*config.num_dbox[ln]])
        mainAnc_rx_ori, mainAnc_rw_ori = anchor_box_sigma(
            sigma, config, ln, mainAnc_rx, mainAnc_rw)
        features = gaussian_pooling(
            MALs[i], num_anch, mainAnc_rx_ori, mainAnc_rw_ori)
        Regs = mulReg_predict_layer(config, features, ln, 'mainStream')
        mainAnc_rx, mainAnc_rw = anchor_box_adjust(
            Regs, config, ln, mainAnc_rx_ori, mainAnc_rw_ori)
        features = gaussian_pooling(MALs[i], num_anch, mainAnc_rx, mainAnc_rw)
        Clss = mulCls_predict_layer(config, features, ln, 'mainStream')

        # --------------------------- Main Stream -----------------------------
        num_classes = Clss.get_shape().as_list()[-1] - 1
        mainAnc_conf = Clss[:, :, -1]
        mainAnc_class = Clss[:, :, :num_classes]

        [mainAnc_BM_x, mainAnc_BM_w, mainAnc_BM_labels_false, mainAnc_BM_scores_d] = \
            anchor_bboxes_encode(num_classes, mainAnc_rx_ori, mainAnc_rw_ori,
                                 Y_label, Y_bbox, Index, config, ln)

        [mainAnc_BM_x_false, mainAnc_BM_w_false, mainAnc_BM_labels, mainAnc_BM_scores] = \
            anchor_bboxes_encode(num_classes, mainAnc_rx, mainAnc_rw,
                                 Y_label, Y_bbox, Index, config, ln)

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat(
            [full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat(
            [full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat(
            [full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat(
            [full_mainAnc_xmax, mainAnc_xmax], axis=1)

        full_mainAnc_BM_x = tf.concat(
            [full_mainAnc_BM_x, mainAnc_BM_x], axis=1)
        full_mainAnc_BM_w = tf.concat(
            [full_mainAnc_BM_w, mainAnc_BM_w], axis=1)
        full_mainAnc_BM_labels = tf.concat(
            [full_mainAnc_BM_labels, mainAnc_BM_labels], axis=1)
        full_mainAnc_BM_scores = tf.concat(
            [full_mainAnc_BM_scores, mainAnc_BM_scores], axis=1)
        full_mainAnc_BM_scores_d = tf.concat(
            [full_mainAnc_BM_scores_d, mainAnc_BM_scores_d], axis=1)

    main_class_loss, main_loc_loss, main_conf_loss, sigma_loss = \
        loss_function(full_mainAnc_class, full_mainAnc_conf,
                      full_mainAnc_xmin, full_mainAnc_xmax,
                      full_mainAnc_BM_x, full_mainAnc_BM_w,
                      full_mainAnc_BM_labels, full_mainAnc_BM_scores,
                      full_mainAnc_BM_scores_d, config)

    loss = main_class_loss + config.p_loc * \
        main_loc_loss + config.p_conf * main_conf_loss + sigma_loss

    trainable_variables = get_trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(
        loss, var_list=trainable_variables)

    return optimizer, loss, trainable_variables


def train_main(config):
    bsz = config.batch_size

    tf.set_random_seed(config.seed)
    X = tf.placeholder(tf.float32, shape=(
        bsz, config.input_steps, feature_dim))
    Y_label = tf.placeholder(tf.int32, [None, config.num_classes])
    Y_bbox = tf.placeholder(tf.float32, [None, 3])
    Index = tf.placeholder(tf.int32, [bsz + 1])
    LR = tf.placeholder(tf.float32)

    optimizer, loss, trainable_variables = \
        train_operation(X, Y_label, Y_bbox, Index, LR, config)

    model_saver = tf.train.Saver(
        var_list=trainable_variables, max_to_keep=config.training_epochs)
    tensor_log_dir = './logs/' + 'RBC_' + mode + '_0'
    count = 0
    while os.path.exists(tensor_log_dir):
        tensor_log_dir = tensor_log_dir[:-2] + '_' + str(count)
        count += 1
    writer = SummaryWriter(log_dir=tensor_log_dir)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=tf_config)

    tf.global_variables_initializer().run()

    # initialize parameters or restore from previous model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if os.listdir(models_dir) == [] or config.initialize:
        init_epoch = 0
        print("Initializing Network")
    else:
        init_epoch = int(config.steps)
        restore_checkpoint_file = join(
            models_dir, 'RBC-ep-' + str(config.steps - 1))
        model_saver.restore(sess, restore_checkpoint_file)

    batch_train_dataX, batch_train_gt_label, batch_train_gt_info, batch_train_index = \
        get_train_data(config, mode, pretrain_dataset, True)
    num_batch_train = len(batch_train_dataX)

    for epoch in range(init_epoch, config.training_epochs):
        loss_info = []
        for idx in range(num_batch_train):
            feed_dict = {X: batch_train_dataX[idx],
                         Y_label: batch_train_gt_label[idx],
                         Y_bbox: batch_train_gt_info[idx],
                         Index: batch_train_index[idx],
                         LR: config.learning_rates[epoch]}
            _, out_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

            step_count_train = epoch * num_batch_train + idx
            writer.add_scalar('batch_loss_' + mode, out_loss, step_count_train)
            loss_info.append(out_loss)

        print("Training epoch ", epoch, " loss: ", np.mean(loss_info))
        writer.add_scalar('epoch_loss_' + mode, np.mean(loss_info), epoch)

        model_saver.save(sess, models_file_prefix, global_step=epoch)


########################################### TEST ############################################

def test_operation(X, config):
    bsz = config.batch_size
    ncls = config.num_classes

    net = base_feature_network(X)
    MALs = main_anchor_layer(net)

    full_mainAnc_class = tf.reshape(tf.constant([]), [bsz, -1, ncls])
    full_mainAnc_conf = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmin = tf.reshape(tf.constant([]), [bsz, -1])
    full_mainAnc_xmax = tf.reshape(tf.constant([]), [bsz, -1])

    for i, ln in enumerate(config.layers_name):
        num_anch = config.num_anchors[ln]
        mainAnc_rx, mainAnc_rw = anchor_box_default(config, ln)
        with tf.variable_scope("sigma_layer" + ln + 'mainStream'):
            sigma = tf.layers.conv1d(inputs=MALs[i], filters=config.num_dbox[ln], activation=tf.nn.sigmoid,
                                     kernel_size=3, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=5))
            sigma = tf.reshape(
                sigma, [config.batch_size, config.num_anchors[ln]*config.num_dbox[ln]])
        mainAnc_rx_ori, mainAnc_rw_ori = anchor_box_sigma(
            sigma, config, ln, mainAnc_rx, mainAnc_rw)
        features = gaussian_pooling(
            MALs[i], num_anch, mainAnc_rx_ori, mainAnc_rw_ori)
        Regs = mulReg_predict_layer(config, features, ln, 'mainStream')
        mainAnc_rx, mainAnc_rw = anchor_box_adjust(
            Regs, config, ln, mainAnc_rx_ori, mainAnc_rw_ori)
        features = gaussian_pooling(MALs[i], num_anch, mainAnc_rx, mainAnc_rw)
        Clss = mulCls_predict_layer(config, features, ln, 'mainStream')

        num_classes = Clss.get_shape().as_list()[-1] - 1
        mainAnc_conf = Clss[:, :, -1]
        mainAnc_class = Clss[:, :, :num_classes]

        mainAnc_xmin = mainAnc_rx - mainAnc_rw / 2
        mainAnc_xmax = mainAnc_rx + mainAnc_rw / 2

        full_mainAnc_class = tf.concat(
            [full_mainAnc_class, mainAnc_class], axis=1)
        full_mainAnc_conf = tf.concat(
            [full_mainAnc_conf, mainAnc_conf], axis=1)
        full_mainAnc_xmin = tf.concat(
            [full_mainAnc_xmin, mainAnc_xmin], axis=1)
        full_mainAnc_xmax = tf.concat(
            [full_mainAnc_xmax, mainAnc_xmax], axis=1)

    full_mainAnc_class = tf.nn.softmax(full_mainAnc_class, dim=-1)
    return full_mainAnc_class, full_mainAnc_conf, full_mainAnc_xmin, full_mainAnc_xmax


def test_main(config):
    batch_dataX, batch_winInfo = get_test_data(config, mode, pretrain_dataset)

    X = tf.placeholder(tf.float32, shape=(
        config.batch_size, config.input_steps, feature_dim))

    anchors_class, anchors_conf, anchors_xmin, anchors_xmax = test_operation(
        X, config)

    model_saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, test_checkpoint_file)

    batch_result_class = []
    batch_result_conf = []
    batch_result_xmin = []
    batch_result_xmax = []

    num_batch = len(batch_dataX)
    for idx in range(num_batch):
        out_anchors_class, out_anchors_conf, out_anchors_xmin, out_anchors_xmax = \
            sess.run([anchors_class, anchors_conf, anchors_xmin, anchors_xmax],
                     feed_dict={X: batch_dataX[idx]})
        batch_result_class.append(out_anchors_class)
        batch_result_conf.append(out_anchors_conf)
        batch_result_xmin.append(out_anchors_xmin * config.window_size)
        batch_result_xmax.append(out_anchors_xmax * config.window_size)

    outDf = pd.DataFrame(columns=config.outdf_columns)

    for i in range(num_batch):
        tmpDf = result_process(batch_winInfo, batch_result_class, batch_result_conf,
                               batch_result_xmin, batch_result_xmax, config, i)

        outDf = pd.concat([outDf, tmpDf])
    if config.save_predict_result:
        outDf.to_csv(predict_file, index=False)
    return outDf


if __name__ == "__main__":
    config = Config()
    start_time = time.time()
    elapsed_time = 0
    if stage == 'train':
        train_main(config)
        elapsed_time = time.time() - start_time
    elif stage == 'test':
        df = test_main(config)
        elapsed_time = time.time() - start_time
        final_result_process(stage, pretrain_dataset,
                             config, mode, method, '', df)
    elif stage == 'fuse':
        final_result_process(stage, pretrain_dataset,
                             config, mode, method, method_temporal)
        elapsed_time = time.time() - start_time
    elif stage == 'train_test_fuse':
        train_main(config)
        elapsed_time = time.time() - start_time
        tf.reset_default_graph()
        df = test_main(config)
        final_result_process(stage, pretrain_dataset,
                             config, mode, method, '', df)
    else:
        print("No stage", stage,
              "Please choose a stage from train/test/fuse/train_test_fuse.")
    print("Elapsed time:", elapsed_time, "start time:", start_time)
