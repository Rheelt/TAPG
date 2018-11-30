# -*- coding: utf-8 -*-

import random
import scipy
import pandas
import numpy
import numpy as np
import os
import glob

'''Constant Variable'''
ctx_num = 2
unit_feature_size = 1024
unit_size = 6.0


def getBatchList(numProps, batch_size, shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    propList = range(numProps)
    batch_start_list = [i * batch_size for i in range(len(propList) / batch_size)]
    batch_start_list.append(len(propList) - batch_size)
    if shuffle == True:
        random.shuffle(propList)
    batch_prop_list = []
    for bstart in batch_start_list:
        batch_prop_list.append(propList[bstart:(bstart + batch_size)])
    return batch_prop_list


def prop_dict_data(prop_dict):
    prop_name_list = prop_dict.keys()

    batch_feature_action = []
    batch_feature_start = []
    batch_feature_end = []

    batch_iou_list = []
    batch_ioa_list = []

    offset = []
    label = []
    for prop_name in prop_name_list:
        batch_feature_action.append(prop_dict[prop_name]["feature_action"])
        batch_feature_start.append(prop_dict[prop_name]["feature_left"])
        batch_feature_end.append(prop_dict[prop_name]["feature_right"])
        batch_iou_list.extend(list(prop_dict[prop_name]["match_iou"]))
        batch_ioa_list.extend(list(prop_dict[prop_name]["match_ioa"]))
        offset.append(prop_dict[prop_name]["offset"])
        label.append(prop_dict[prop_name]["label"])

    batch_feature_action = numpy.concatenate(batch_feature_action)
    batch_feature_start = numpy.concatenate(batch_feature_start)
    batch_feature_end = numpy.concatenate(batch_feature_end)
    offset = numpy.concatenate(offset)
    label = numpy.concatenate(label)

    batch_iou_list = numpy.array(batch_iou_list)
    batch_ioa_list = numpy.array(batch_ioa_list)
    fullData = {"feature_action": batch_feature_action, "feature_start": batch_feature_start,
                "feature_end": batch_feature_end,
                "iou_list": batch_iou_list, "ioa_list": batch_ioa_list, "offset": offset, "label": label}
    return fullData


def getVideoFeature(videoname, subset):
    appearance_path = '/feature/THUMOS_14_two_stream_feature/{}_appearance/'.format(subset)
    denseflow_path = '/feature/THUMOS_14_two_stream_feature/{}_denseflow/'.format(subset)
    rgb_file_list = glob.glob(appearance_path + videoname + '*')
    flow_file_list = glob.glob(denseflow_path + videoname + '*')

    swin_start = 1.0
    window_size = 6.0
    rgb_feature = []
    for ii in range(len(rgb_file_list)):
        swin_end = swin_start + window_size
        if os.path.exists(appearance_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
            rgb_feat = np.load(
                appearance_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            rgb_feature.append(rgb_feat)
        swin_start = swin_end

    swin_start = 1.0
    flow_feature = []
    for ii in range(len(flow_file_list)):
        swin_end = swin_start + window_size
        if os.path.exists(denseflow_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
            flow_feat = np.load(
                denseflow_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            flow_feature.append(flow_feat)
        swin_start = swin_end

    return rgb_feature, flow_feature


def getVideoProposalData(videoName):
    adf = pandas.read_csv("./outputs/tapg_results_epoch11/" + videoName + ".csv")
    snippets = adf.frame.values[:]
    score_action = adf.action.values[:]
    rnndf = pandas.read_csv("./outputs/tapg_results_epoch11/" + videoName + "_rnn.csv")
    rnn_feature = rnndf.values[:, 1:]
    pdf = pandas.read_csv("./outputs/tapg_results_epoch11/" + videoName + ".csv")
    candidate_proposals = pdf.values[:]
    len_snippets = snippets.shape[0]
    num_candidate_proposals = candidate_proposals.shape[0]
    label = np.zeros([num_candidate_proposals], dtype=np.int32)
    offset = np.zeros([num_candidate_proposals, 2], dtype=np.float32)
    feature_left = []
    feature_right = []
    feature_action = []
    for idx, pre_proposal in enumerate(candidate_proposals):
        proposal_f_start = pre_proposal[0]
        proposal_f_end = pre_proposal[1]
        match_iou = pre_proposal[4]
        gt_f_start = pre_proposal[6]
        gt_f_end = pre_proposal[7]
        round_gt_start = np.round(gt_f_start / unit_size) * unit_size + 1
        round_gt_end = np.round(gt_f_end / unit_size) * unit_size + 1
        snippets_start_idx = snippets.tolist().index(proposal_f_start)
        snippets_end_idx = snippets.tolist().index(proposal_f_end)
        '''Get the central features'''
        central_action_score = score_action[snippets_start_idx:snippets_end_idx + 1]
        central_rnn_feature = rnn_feature[snippets_start_idx:snippets_end_idx + 1, :]
        central_action_score_sum = np.sum(central_action_score)
        central_action_score_reg = central_action_score / central_action_score_sum
        central_action_score_reg = np.reshape(central_action_score_reg, [-1, 1])
        central_rnn_feature = central_rnn_feature * central_action_score_reg
        # pool_central_feat = np.sum(central_rnn_feature, axis=0)
        pool_central_feat = np.mean(central_rnn_feature, axis=0)
        feature_action.append(pool_central_feat)
        '''Get the left features'''
        left_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
        left_action_score = np.zeros([0, 1], dtype=np.float32)
        count = 0
        context_ext = False
        current_pos = snippets_start_idx
        while count < ctx_num:
            if current_pos >= 0:
                tmp_feat = rnn_feature[current_pos, :]
                left_feat = np.vstack((left_feat, tmp_feat))
                left_action_score = np.vstack((left_action_score, score_action[current_pos]))
                context_ext = True
            current_pos -= 1
            count += 1
        count = 0
        current_pos = snippets_start_idx
        while count < ctx_num:
            current_pos += 1
            if current_pos < len_snippets:
                tmp_feat = rnn_feature[current_pos, :]
                left_feat = np.vstack((left_feat, tmp_feat))
                left_action_score = np.vstack((left_action_score, score_action[current_pos]))
                context_ext = True
            count += 1
        if context_ext:
            left_action_score_sum = np.sum(left_action_score)
            left_action_score_reg = left_action_score / left_action_score_sum
            left_action_score_reg = np.reshape(left_action_score_reg, [-1, 1])
            left_feat = left_feat * left_action_score_reg
            # pool_left_feat = np.sum(left_feat, axis=0)
            pool_left_feat = np.mean(left_feat, axis=0)
        else:
            pool_left_feat = np.zeros([unit_feature_size], dtype=np.float32)
        feature_left.append(pool_left_feat)
        '''Get the right features'''
        right_feat = np.zeros([0, unit_feature_size], dtype=np.float32)
        right_action_score = np.zeros([0, 1], dtype=np.float32)
        count = 0
        context_ext = False
        current_pos = snippets_end_idx
        while count < ctx_num:
            if current_pos >= 0:
                tmp_feat = rnn_feature[current_pos, :]
                right_feat = np.vstack((right_feat, tmp_feat))
                right_action_score = np.vstack((right_action_score, score_action[current_pos]))
                context_ext = True
            current_pos -= 1
            count += 1
        count = 0
        current_pos = snippets_end_idx
        while count < ctx_num:
            current_pos += 1
            if current_pos < len_snippets:
                tmp_feat = rnn_feature[current_pos, :]
                right_feat = np.vstack((right_feat, tmp_feat))
                right_action_score = np.vstack((right_action_score, score_action[current_pos]))
                context_ext = True
            count += 1
        if context_ext:
            right_action_score_sum = np.sum(right_action_score)
            right_action_score_reg = right_action_score / right_action_score_sum
            right_action_score_reg = np.reshape(right_action_score_reg, [-1, 1])
            right_feat = right_feat * right_action_score_reg
            # pool_right_feat = np.sum(right_feat, axis=0)
            pool_right_feat = np.mean(right_feat, axis=0)
        else:
            pool_right_feat = np.zeros([unit_feature_size], dtype=np.float32)
        feature_right.append(pool_right_feat)
        if match_iou >= 0.6:
            offset[idx, 0] = (round_gt_start - proposal_f_start) / unit_size
            offset[idx, 1] = (round_gt_end - proposal_f_end) / unit_size
            label[idx] = 1
        else:
            offset[idx, 0] = 0
            offset[idx, 1] = 0
            label[idx] = 0
    prop_dict = {"match_iou": pdf.match_iou.values[:], "match_ioa": pdf.match_ioa.values[:],
                 "xmin": pdf.xmin.values[:], "xmax": pdf.xmax.values[:], "xmin_score": pdf.xmin_score.values[:],
                 "xmax_score": pdf.xmax_score.values[:],
                 "feature_left": numpy.array(feature_left), "feature_right": numpy.array(feature_right),
                 "feature_action": numpy.array(feature_action), "offset": offset, "label": label}

    return prop_dict


def getBatchData(fullData, batch_props):
    batch_feature_action = fullData["feature_action"][batch_props]
    batch_feature_start = fullData["feature_start"][batch_props]
    batch_feature_end = fullData["feature_end"][batch_props]
    batch_iou_list = fullData["iou_list"][batch_props]
    batch_ioa_list = fullData["ioa_list"][batch_props]
    batch_offset = fullData["offset"][batch_props]
    batch_label = fullData["label"][batch_props]
    return batch_feature_action, batch_feature_start, batch_feature_end, batch_iou_list, batch_ioa_list, batch_offset, batch_label


def getFullData(dataSet, flag_test=False):
    annoDf = pandas.read_csv("./data/thumos_14_annotations/" + dataSet + "_Annotation.csv")
    videoNameList = list(set(annoDf.video.values[:]))
    VideoData = {}
    for videoName in videoNameList:
        prop_dict = getVideoProposalData(videoName)
        VideoData[videoName] = prop_dict
        print len(VideoData.keys())

    if flag_test == False:
        fullData = prop_dict_data(VideoData)
        return fullData
    else:
        return VideoData
