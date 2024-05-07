import os
import torch
import numpy as np
import pandas as pd
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision,BinaryAUROC
from NTXentLoss import NTXentLoss
from model import MIPE
from utils import *

if __name__ == '__main__':
    file_name = "../data/dataset/alldata.pkl"
    data_PECAN = pd.read_pickle(file_name)
    K = 5
    for kfold in range(K):
        train_data_PECAN, val_data_PECAN, test_data_PECAN = get_k_fold_data(K, kfold, data_PECAN)
        model = MIPE().to(device)
        loss_NTXent = NTXentLoss()
        loss_BCE = torch.nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        epoch = 800
        model.train()
        Loss_val = []
        AUROC_val = []
        torch.cuda.empty_cache()
        for e in range(epoch):
            result_loss_train_all = 0.0
            train_auprc = 0.0
            train_auroc = 0.0
            train_precision = 0.0
            train_recall = 0.0
            train_mcc = 0.0
            ab_train_auprc = 0.0
            ab_train_auroc = 0.0
            ab_train_precision = 0.0
            ab_train_recall = 0.0
            ab_train_mcc = 0.0
            model.train()
            for i in range(len(train_data_PECAN)):
                ag_node_attr, ag_edge_ind, ag_targets = torch.tensor(train_data_PECAN[i]["vertex_AG"],dtype=torch.float).to(device), torch.tensor(train_data_PECAN[i]["edge_AG"], dtype=torch.long).to(device), torch.tensor(train_data_PECAN[i]["label_AG"],dtype=torch.float).to(device)
                ab_node_attr, ab_edge_ind, ab_targets = torch.tensor(train_data_PECAN[i]["vertex_AB"],dtype=torch.float).to(device), torch.tensor(train_data_PECAN[i]["edge_AB"], dtype=torch.long).to(device), torch.tensor(train_data_PECAN[i]["label_AB"],dtype=torch.float).to(device)
                ag_esm = torch.unsqueeze(torch.tensor(train_data_PECAN[i]["ESM1b_AG"]).to(device).t(), dim=0)
                ab_esm = torch.unsqueeze(torch.tensor(train_data_PECAN[i]["AbLang_AB"]).to(device).t(), dim=0)
                ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)# PLM
                ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                ag_edge_ind, ab_edge_ind = CreateGearnetGraph(train_data_PECAN[i])# Edge
                agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, ag_esm, ab_esm, False, ag_targets,ab_targets, i]
                outputs = model(*agab)
                edge_label = np.zeros((ag_targets.shape[0], ab_targets.shape[0]))
                inter_edge_index = (train_data_PECAN[i]["edge_AGAB"])
                for edge_ind in range(len(inter_edge_index[0])):
                    edge_label[inter_edge_index[0][edge_ind], inter_edge_index[1][edge_ind]] = 1.0
                edge_label = torch.tensor(edge_label, dtype=torch.float32).to(device)
                # multimodalloss
                ag_out_seq = outputs[10]
                ag_out_strc = outputs[11]
                ab_out_seq = outputs[12]
                ab_out_strc = outputs[13]
                multimodal_loss_ag_intra_seq = loss_NTXent(outputs[4], outputs[4])
                multimodal_loss_ag_intra_struc = loss_NTXent(outputs[8], outputs[8])
                multimodal_loss_ab_intra_seq = loss_NTXent(outputs[6], outputs[6])
                multimodal_loss_ab_intra_strc = loss_NTXent(outputs[9], outputs[9])
                tag_ag = []
                tag_ab = []
                for tag_idx in range(len(train_data_PECAN[i]["label_AG"])):
                    if ((train_data_PECAN[i]["label_AG"][tag_idx] and ag_out_seq[tag_idx]) or (
                            train_data_PECAN[i]["label_AG"][tag_idx] and ag_out_strc[tag_idx])):
                        tag_ag.append(1.)
                    else:
                        tag_ag.append(0.)
                for tag_idx in range(len(train_data_PECAN[i]["label_AB"])):
                    if ((train_data_PECAN[i]["label_AB"][tag_idx] and ab_out_seq[tag_idx]) or (
                            train_data_PECAN[i]["label_AB"][tag_idx] and ab_out_strc[tag_idx])):
                        tag_ab.append(1.)
                    else:
                        tag_ab.append(0.)
                cos_ag = consine_inter(outputs[4], outputs[8])
                cos_ab = consine_inter(outputs[6], outputs[9])
                cos_ag_pos = torch.mul(torch.tensor(tag_ag).to(device), cos_ag.to(device))
                cos_ab_pos = torch.mul(torch.tensor(tag_ab).to(device), cos_ab.to(device))
                multimodal_loss_ag_inter = torch.div(torch.sum(cos_ag_pos), torch.sum(cos_ag))
                multimodal_loss_ab_inter = torch.div(torch.sum(cos_ab_pos), torch.sum(cos_ab))
                multimodal_loss = 0.06 * (multimodal_loss_ag_inter + multimodal_loss_ab_inter + multimodal_loss_ag_intra_seq + multimodal_loss_ag_intra_struc + multimodal_loss_ab_intra_seq + multimodal_loss_ab_intra_strc + loss_NTXent(outputs[4], outputs[8]) + loss_NTXent(outputs[6], outputs[9]))
                result_loss = 5 * loss_BCE((outputs[0]).squeeze(dim=1), ag_targets) + 5 * loss_BCE((outputs[1]).squeeze(dim=1), ab_targets) + 1 * multimodal_loss + 10 * (loss_BCE(outputs[2], edge_label)) + 10 * (loss_BCE(outputs[3].t(), edge_label))
                if ((i) % 16 == 0):
                    result_loss_batch = result_loss
                else:
                    result_loss_batch = result_loss + result_loss_batch
                if ((i + 1) % 16 == 0 or i == len(train_data_PECAN) - 1):
                    optim.zero_grad()
                    result_loss_batch.backward()
                    optim.step()
                result_loss_i = float(result_loss.item())
                result_loss_train_all = result_loss_train_all + result_loss_i
                # ag
                output_ag = torch.flatten(outputs[0]) if i==0 else torch.cat((output_ag,torch.flatten(outputs[0])),dim=0)
                target_ag = ag_targets.long() if i==0 else torch.cat((target_ag, ag_targets.long()), dim=0)
                output_ab = torch.flatten(outputs[1]) if i==0 else torch.cat((output_ab, torch.flatten(outputs[1])), dim=0)
                target_ab = ab_targets.long() if i==0 else torch.cat((target_ab, ab_targets.long()), dim=0)
                # train_auprc_i, train_auroc_i, train_precision_i, train_recall_i, train_mcc_i = evalution_prot(torch.flatten(outputs[0]), ag_targets.long())
                # train_auprc = train_auprc_i + train_auprc
                # train_auroc = train_auroc_i + train_auroc
                # train_precision = train_precision_i + train_precision
                # train_recall = train_recall_i + train_recall
                # train_mcc = train_mcc_i + train_mcc
                # # ab
                # ab_train_auprc_i, ab_train_auroc_i, ab_train_precision_i, ab_train_recall_i, ab_train_mcc_i = evalution_prot(
                #     torch.flatten(outputs[1]), ab_targets.long())
                # ab_train_auprc = ab_train_auprc_i + ab_train_auprc
                # ab_train_auroc = ab_train_auroc_i + ab_train_auroc
                # ab_train_precision = ab_train_precision_i + ab_train_precision
                # ab_train_recall = ab_train_recall_i + ab_train_recall
                # ab_train_mcc = ab_train_mcc_i + ab_train_mcc

                # del [outputs, ag_targets, ab_targets, train_auprc_i, train_auroc_i, train_precision_i,
                #      train_recall_i,
                #      train_mcc_i, ab_train_auprc_i, ab_train_auroc_i, ab_train_precision_i, ab_train_recall_i,
                #      ab_train_mcc_i]

            #
            model.eval()
            val_total_loss = 0
            val_total_loss_all = 0
            right_number = 0

            val_auprc = 0.0
            val_auroc = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_mcc = 0.0
            ab_val_auprc = 0.0
            ab_val_auroc = 0.0
            ab_val_precision = 0.0
            ab_val_recall = 0.0
            ab_val_mcc = 0.0

            with torch.no_grad():
                for j in range(len(val_data_PECAN)):
                    ag_node_attr, ag_edge_ind, ag_targets = torch.tensor(val_data_PECAN[j]["vertex_AG"],dtype=torch.float), torch.tensor(val_data_PECAN[j]["edge_AG"], dtype=torch.long), torch.tensor(val_data_PECAN[j]["label_AG"],dtype=torch.float)
                    ab_node_attr, ab_edge_ind, ab_targets = torch.tensor(val_data_PECAN[j]["vertex_AB"],dtype=torch.float), torch.tensor(val_data_PECAN[j]["edge_AB"], dtype=torch.long), torch.tensor(val_data_PECAN[j]["label_AB"],dtype=torch.float)
                    ag_node_attr = ag_node_attr.to(device)
                    ag_edge_ind = ag_edge_ind.to(device)
                    ag_targets = ag_targets.to(device)
                    ab_node_attr = ab_node_attr.to(device)
                    ab_edge_ind = ab_edge_ind.to(device)
                    ab_targets = ab_targets.to(device)
                    # PLM
                    ag_esm = torch.tensor(val_data_PECAN[j]["ESM1b_AG"]).to(device)
                    ag_esm = ag_esm.t()
                    ag_esm = torch.unsqueeze(ag_esm, dim=0)
                    ab_esm = torch.tensor(val_data_PECAN[j]["AbLang_AB"]).to(device)
                    ab_esm = ab_esm.t()
                    ab_esm = torch.unsqueeze(ab_esm, dim=0)

                    ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)
                    ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                    # Edge
                    ag_edge_ind, ab_edge_ind = CreateGearnetGraph(val_data_PECAN[j])
                    agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, ag_esm, ab_esm, False, ag_targets,ab_targets, i]
                    outputs = model(*agab)
                    edge_label = np.zeros((ag_targets.shape[0], ab_targets.shape[0]))
                    inter_edge_index = (val_data_PECAN[j]["edge_AGAB"])
                    for edge_ind in range(len(inter_edge_index[0])):
                        edge_label[inter_edge_index[0][edge_ind], inter_edge_index[1][edge_ind]] = 1.0
                    edge_label = torch.tensor(edge_label, dtype=torch.float32).to(device)
                    ag_out_seq = outputs[10]
                    ag_out_strc = outputs[11]
                    ab_out_seq = outputs[12]
                    ab_out_strc = outputs[13]
                    multimodal_loss_ag_intra_seq = loss_NTXent(outputs[4], outputs[4])
                    multimodal_loss_ag_intra_struc = loss_NTXent(outputs[8], outputs[8])
                    multimodal_loss_ab_intra_seq = loss_NTXent(outputs[6], outputs[6])
                    multimodal_loss_ab_intra_strc = loss_NTXent(outputs[9], outputs[9])
                    tag_ag = []
                    tag_ab = []
                    for tag_idx in range(len(val_data_PECAN[j]["label_AG"])):
                        if ((val_data_PECAN[j]["label_AG"][tag_idx] and ag_out_seq[tag_idx]) or (
                                val_data_PECAN[j]["label_AG"][tag_idx] and ag_out_strc[tag_idx])):
                            tag_ag.append(1.)
                        else:
                            tag_ag.append(0.)
                    for tag_idx in range(len(val_data_PECAN[j]["label_AB"])):
                        if ((val_data_PECAN[j]["label_AB"][tag_idx] and ab_out_seq[tag_idx]) or (
                                val_data_PECAN[j]["label_AB"][tag_idx] and ab_out_strc[tag_idx])):
                            tag_ab.append(1.)
                        else:
                            tag_ab.append(0.)
                    cos_ag = consine_inter(outputs[4], outputs[8])
                    cos_ab = consine_inter(outputs[6], outputs[9])
                    cos_ag_pos = torch.mul(torch.tensor(tag_ag).to(device), cos_ag.to(device))
                    cos_ab_pos = torch.mul(torch.tensor(tag_ab).to(device), cos_ab.to(device))
                    multimodal_loss_ag_inter = torch.div(torch.sum(cos_ag_pos), torch.sum(cos_ag))
                    multimodal_loss_ab_inter = torch.div(torch.sum(cos_ab_pos), torch.sum(cos_ab))
                    multimodal_loss = 0.06 * (multimodal_loss_ag_inter + multimodal_loss_ab_inter + multimodal_loss_ag_intra_seq + multimodal_loss_ag_intra_struc + multimodal_loss_ab_intra_seq + multimodal_loss_ab_intra_strc + loss_NTXent(outputs[4], outputs[8]) + loss_NTXent(outputs[6], outputs[9]))
                    val_result_loss = 5 * loss_BCE((outputs[0]).squeeze(dim=1), ag_targets) + 5 * loss_BCE((outputs[1]).squeeze(dim=1), ab_targets) + 1 * multimodal_loss + 10 * (loss_BCE(outputs[2], edge_label)) + 10 * (loss_BCE(outputs[3].t(), edge_label))
                    # loss
                    val_result_loss_j = float(val_result_loss.item())
                    val_total_loss_all = val_total_loss_all + val_result_loss_j

                    output_ag_val = torch.flatten(outputs[0]) if j == 0 else torch.cat((output_ag_val, torch.flatten(outputs[0])), dim=0)
                    target_ag_val = ag_targets.long() if j == 0 else torch.cat((target_ag_val, ag_targets.long()), dim=0)
                    output_ab_val = torch.flatten(outputs[1]) if j == 0 else torch.cat((output_ab_val, torch.flatten(outputs[1])), dim=0)
                    target_ab_val = ab_targets.long() if j == 0 else torch.cat((target_ab_val, ab_targets.long()), dim=0)
                    # # evalution
                    # # ag
                    # val_auprc_j, val_auroc_j, val_precision_j, val_recall_j, val_mcc_j = evalution_prot(
                    #     torch.flatten(outputs[0]), ag_targets.long())
                    # val_auprc = val_auprc_j + val_auprc
                    # val_auroc = val_auroc_j + val_auroc
                    # val_precision = val_precision_j + val_precision
                    # val_recall = val_recall_j + val_recall
                    # val_mcc = val_mcc_j + val_mcc
                    # # ab
                    # ab_val_auprc_j, ab_val_auroc_j, ab_val_precision_j, ab_val_recall_j, ab_val_mcc_j = evalution_prot(
                    #     torch.flatten(outputs[1]), ab_targets.long())
                    # ab_val_auprc = ab_val_auprc_j + ab_val_auprc
                    # ab_val_auroc = ab_val_auroc_j + ab_val_auroc
                    # ab_val_precision = ab_val_precision_j + ab_val_precision
                    # ab_val_recall = ab_val_recall_j + ab_val_recall
                    # ab_val_mcc = ab_val_mcc_j + ab_val_mcc
                    #
                    # del [outputs, ag_targets, ab_targets, val_auprc_j, val_auroc_j, val_precision_j, val_recall_j,
                    #      val_mcc_j, ab_val_auprc_j, ab_val_auroc_j, ab_val_precision_j, ab_val_recall_j,
                    #      ab_val_mcc_j, edge_label]
            if ((e + 1) % 10 == 0):
                # evaluate
                auprc_ag, auroc_ag, mcc_ag = evalution_prot(output_ag,target_ag)
                auprc_ab, auroc_ab, mcc_ab = evalution_prot(output_ab,target_ab)
                val_auprc_ag, val_auroc_ag, val_mcc_ag = evalution_prot(output_ag_val, target_ag_val)
                val_auprc_ab, val_auroc_ab, val_mcc_ab = evalution_prot(output_ab_val,target_ab_val)
                print("===============================")
                print(val_auroc_ag)
                print(val_auprc_ag)
                print(val_mcc_ag)
                print(val_auroc_ab)
                print(val_auprc_ab)
                print(val_mcc_ab)
                torch.save(model, "output_files/modelsave/model_k{}_{}".format(kfold, (e + 1)))
                Loss_val.append(val_total_loss_all / len(val_data_PECAN))
                AUROC_val.append(val_auroc / len(val_data_PECAN))

        # test
        min_idx = Loss_val.index(min(Loss_val))
        min_idx = (min_idx + 1) * 10
        model_filepath = "output_files/modelsave/model_k" + str(kfold) + "_" + str(min_idx)
        model = torch.load(model_filepath)
        model.eval()
        test_auprc = 0.0
        test_auroc = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_mcc = 0.0
        ab_test_auprc = 0.0
        ab_test_auroc = 0.0
        ab_test_precision = 0.0
        ab_test_recall = 0.0
        ab_test_mcc = 0.0

        ag_h1_all = []
        ab_h1_all = []

        with torch.no_grad():
                for j in range(len(test_data_PECAN)):
                    ag_node_attr, ag_edge_ind, ag_targets = torch.tensor(test_data_PECAN[j]["vertex_AG"],dtype=torch.float),torch.tensor(test_data_PECAN[j]["edge_AG"],dtype=torch.long),torch.tensor(test_data_PECAN[j]["label_AG"],dtype=torch.float)
                    ab_node_attr, ab_edge_ind, ab_targets = torch.tensor(test_data_PECAN[j]["vertex_AB"],dtype=torch.float),torch.tensor(test_data_PECAN[j]["edge_AB"],dtype=torch.long),torch.tensor(test_data_PECAN[j]["label_AB"],dtype=torch.float)
                    ag_node_attr = ag_node_attr.to(device)
                    ag_edge_ind = ag_edge_ind.to(device)
                    ag_targets = ag_targets.to(device)
                    ab_node_attr = ab_node_attr.to(device)
                    ab_edge_ind = ab_edge_ind.to(device)
                    ab_targets = ab_targets.to(device)
                    # PLM
                    ag_esm = torch.tensor(test_data_PECAN[j]["ESM1b_AG"]).to(device)
                    ag_esm = ag_esm.t()
                    ag_esm = torch.unsqueeze(ag_esm, dim=0)
                    ab_esm = torch.tensor(test_data_PECAN[j]["AbLang_AB"]).to(device)
                    ab_esm = ab_esm.t()
                    ab_esm = torch.unsqueeze(ab_esm, dim=0)
                    ag_node_attr = torch.unsqueeze(ag_node_attr, dim=0)
                    ab_node_attr = torch.unsqueeze(ab_node_attr, dim=0)
                    # Edge
                    ag_edge_ind, ab_edge_ind = CreateGearnetGraph(test_data_PECAN[j])
                    agab = [ag_node_attr, ag_edge_ind, ab_node_attr, ab_edge_ind, True, ag_targets, ab_targets, j]
                    outputs = model(*agab)
                    # evalution
                    output_ag_test = torch.flatten(outputs[0]) if j == 0 else torch.cat((output_ag_test, torch.flatten(outputs[0])), dim=0)
                    target_ag_test = ag_targets.long() if j == 0 else torch.cat((target_ag_test, ag_targets.long()), dim=0)
                    output_ab_test = torch.flatten(outputs[1]) if j == 0 else torch.cat((output_ab_test, torch.flatten(outputs[1])), dim=0)
                    target_ab_test = ab_targets.long() if j == 0 else torch.cat((target_ab_test, ab_targets.long()), dim=0)
                    test_auprc_ag, test_auroc_ag, test_mcc_ag = evalution_prot(output_ag_test, target_ag_test)
                    test_auprc_ab, test_auroc_ab, test_mcc_ab = evalution_prot(output_ab_test, target_ab_test)
                    # test_auprc_j, test_auroc_j, test_precision_j, test_recall_j, test_mcc_j = evalution_prot(torch.flatten(outputs[0]), ag_targets.long())
                    # test_auprc = test_auprc_j + test_auprc
                    # test_auroc = test_auroc_j + test_auroc
                    # test_precision = test_precision_j + test_precision
                    # test_recall = test_recall_j + test_recall
                    # test_mcc = test_mcc_j + test_mcc
                    # ab_test_auprc_j, ab_test_auroc_j, ab_test_precision_j, ab_test_recall_j, ab_test_mcc_j = evalution_prot(torch.flatten(outputs[1]), ab_targets.long())
                    # ab_test_auprc = ab_test_auprc_j + ab_test_auprc
                    # ab_test_auroc = ab_test_auroc_j + ab_test_auroc
                    # ab_test_precision = ab_test_precision_j + ab_test_precision
                    # ab_test_recall = ab_test_recall_j + ab_test_recall
                    # ab_test_mcc = ab_test_mcc_j + ab_test_mcc

                # print("==========================================================================")
                # print("Ag test AUROC:{}".format(test_auroc / len(test_data_PECAN)))
                # print("Ag test AUPRC:{}".format(test_auprc / len(test_data_PECAN)))
                # print("Ag test MCC:{}".format(test_mcc / len(test_data_PECAN)))
                # print("Ab test AUROC:{}".format(ab_test_auroc / len(test_data_PECAN)))
                # print("Ab test AUPRC:{}".format(ab_test_auprc / len(test_data_PECAN)))
                # print("Ab test MCC:{}".format(ab_test_mcc / len(test_data_PECAN)))
                print("==========================================================================")
                print("Ag test AUROC:{}".format(test_auprc_ag))
                print("Ag test AUPRC:{}".format(test_auroc_ag))
                print("Ag test MCC:{}".format(test_mcc_ag))
                print("Ab test AUROC:{}".format(test_auprc_ab))
                print("Ab test AUPRC:{}".format(test_auroc_ab))
                print("Ab test MCC:{}".format(test_mcc_ab))