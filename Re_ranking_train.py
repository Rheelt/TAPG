# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

import Re_ranking_load_data as Re_ranking_load_data
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def re_ranking_loss(anchors_iou, match_iou, config):
    # iou regressor
    u_hmask = tf.cast(match_iou > 0.7, dtype=tf.float32)
    u_mmask = tf.cast(tf.logical_and(match_iou <= 0.7, match_iou > 0.3), dtype=tf.float32)
    u_lmask = tf.cast(match_iou <= 0.3, dtype=tf.float32)

    num_h = tf.reduce_sum(u_hmask)
    num_m = tf.reduce_sum(u_mmask)
    num_l = tf.reduce_sum(u_lmask)

    r_m = config.u_ratio_m * num_h / (num_m)
    r_m = tf.minimum(r_m, 1)
    u_smmask = tf.random_uniform([tf.shape(u_hmask)[0]], dtype=tf.float32)
    u_smmask = u_smmask * u_mmask
    u_smmask = tf.cast(u_smmask > (1. - r_m), dtype=tf.float32)

    r_l = config.u_ratio_l * num_h / (num_l)
    r_l = tf.minimum(r_l, 1)
    u_slmask = tf.random_uniform([tf.shape(u_hmask)[0]], dtype=tf.float32)
    u_slmask = u_slmask * u_lmask
    u_slmask = tf.cast(u_slmask > (1. - r_l), dtype=tf.float32)

    iou_weights = u_hmask + u_smmask + u_slmask
    iou_loss = abs_smooth(match_iou - anchors_iou)
    iou_loss = tf.losses.compute_weighted_loss(iou_loss, iou_weights)

    num_iou = [tf.reduce_sum(u_hmask), tf.reduce_sum(u_smmask), tf.reduce_sum(u_slmask)]

    loss = {'iou_loss': iou_loss, 'num_iou': num_iou}

    return loss


def re_ranking_train(X_feature_action, X_feature_start, X_feature_end, Y_iou, label_ph,
              offset_ph, keep_prob, LR, config):
    X = tf.concat((X_feature_start, X_feature_action, X_feature_end), axis=1)
    X = tf.nn.l2_normalize(X, dim=1)
    net = tf.layers.dense(X, units=512, activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, units=3, activation=None,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    net_iou = net[:, 0]
    net_offset_pred = net[:, 1:]
    net_iou = tf.nn.sigmoid(net_iou)

    anchors_iou = tf.reshape(net_iou, [-1])

    loss = re_ranking_loss(anchors_iou, Y_iou, config)

    label_tmp = tf.to_float(tf.reshape(label_ph, [config.batch_size, 1]))
    label_for_reg = tf.concat([label_tmp, label_tmp], axis=1)
    regress_loss = tf.div(tf.reduce_sum(tf.multiply(tf.abs(tf.subtract(net_offset_pred, offset_ph)), label_for_reg)),
                          tf.reduce_sum(label_tmp))

    trainable_variables = tf.trainable_variables()

    latent_l2 = 0.000025 * sum(tf.nn.l2_loss(tf_var) for tf_var in trainable_variables)
    latent_cost = 10 * loss["iou_loss"] + latent_l2 + regress_loss
    latent_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(latent_cost,
                                                                         var_list=trainable_variables)
    loss["l2"] = latent_l2
    loss["reg"] = regress_loss
    return latent_optimizer, loss, anchors_iou


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # common information
        self.training_epochs = 20

        self.input_steps = 256
        self.learning_rates = [0.005] * 10 + [0.001] * 10 + [0.0005] * 10 + [0.0001] * 10
        # self.learning_rates = np.logspace(-4, -5, num=30)
        self.training_epochs = len(self.learning_rates)
        self.batch_size = 400
        self.u_ratio_m = 1
        self.u_ratio_l = 2
        self.lambda_reg = 2.0


if __name__ == "__main__":
    config = Config()

    X_feature_action = tf.placeholder(tf.float32, [config.batch_size, 1024])
    X_feature_start = tf.placeholder(tf.float32, [config.batch_size, 1024])
    X_feature_end = tf.placeholder(tf.float32, [config.batch_size, 1024])
    Y_iou = tf.placeholder(tf.float32, [config.batch_size])
    label_ph = tf.placeholder(tf.int32, shape=(config.batch_size))
    offset_ph = tf.placeholder(tf.float32, shape=(config.batch_size, 2))
    LR = tf.placeholder(tf.float32)
    dropout_ratio = tf.placeholder(tf.float32)
    latent_optimizer, loss, prop_score = re_ranking_train(X_feature_action, X_feature_start, X_feature_end, Y_iou, label_ph,
                                                   offset_ph, dropout_ratio, LR, config)

    model_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()

    fullDataTrain = Re_ranking_load_data.getFullData("Val")
    numPropsTrain = len(fullDataTrain["iou_list"])
    fullDataTest = Re_ranking_load_data.getFullData("Test")
    numPropsTest = len(fullDataTest["iou_list"])

    train_info = {"iou_loss": [], "l2": [], "reg": []}
    val_info = {"iou_loss": [], "l2": [], "reg": []}

    for epoch in range(0, config.training_epochs):
        ## TRAIN ##
        batch_prop_list = Re_ranking_load_data.getBatchList(numPropsTrain, config.batch_size, shuffle=True)
        mini_info = {"iou_loss": [], "l2": [], "reg": []}
        for batch_props in batch_prop_list:
            batch_feature_action, batch_feature_start, batch_feature_end, batch_iou_list, batch_ioa_list, batch_offset, batch_label = \
                Re_ranking_load_data.getBatchData(
                    fullDataTrain, batch_props)
            _, out_loss, out_score, out_alpha = sess.run([latent_optimizer, loss, prop_score, config.alpha],
                                                         feed_dict={X_feature_action: batch_feature_action,
                                                                    X_feature_start: batch_feature_start,
                                                                    X_feature_end: batch_feature_end,
                                                                    Y_iou: batch_iou_list,
                                                                    dropout_ratio: 0.5,
                                                                    offset_ph: batch_offset,
                                                                    label_ph: batch_label,
                                                                    LR: config.learning_rates[epoch]})
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])
            mini_info["reg"].append(out_loss["reg"])

        train_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        train_info["l2"].append(np.mean(mini_info["l2"]))
        train_info["reg"].append(np.mean(mini_info["reg"]))
        print 'TRAIN' + str(epoch) + '   ' + str(train_info['iou_loss']) + '\n' + str(train_info["reg"]) + '\n' + str(
            train_info['l2'])
        model_saver.save(sess, "./models/re_ranking_model_epoch", global_step=epoch)

        batch_prop_list = Re_ranking_load_data.getBatchList(numPropsTest, config.batch_size, shuffle=False)
        mini_info = {"iou_loss": [], "l2": [], "reg": []}
        for batch_props in batch_prop_list:
            batch_feature_action, batch_feature_start, batch_feature_end, batch_iou_list, batch_ioa_list, batch_offset, batch_label = Re_ranking_load_data.getBatchData(
                fullDataTest, batch_props)
            out_loss = sess.run(loss, feed_dict={X_feature_action: batch_feature_action,
                                                 X_feature_start: batch_feature_start,
                                                 X_feature_end: batch_feature_end,
                                                 Y_iou: batch_iou_list,
                                                 offset_ph: batch_offset,
                                                 label_ph: batch_label,
                                                 dropout_ratio: 1.0,
                                                 LR: config.learning_rates[epoch]})
            mini_info["iou_loss"].append(out_loss["iou_loss"])
            mini_info["l2"].append(out_loss["l2"])
            mini_info["reg"].append(out_loss["reg"])

        val_info["iou_loss"].append(np.mean(mini_info["iou_loss"]))
        val_info["l2"].append(np.mean(mini_info["l2"]))
        val_info["reg"].append(np.mean(mini_info["reg"]))
        print 'VAL' + str(epoch) + '   ' + str(val_info['iou_loss']) + '\n' + str(val_info["reg"]) + '\n' + str(
            val_info['l2'])
