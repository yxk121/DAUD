from tqdm import tqdm
import argparse
import random
import datetime
from pathlib import Path
import h5py
import os, psutil
import re
import sys
import torch
import numpy as np
import torch.nn as nn
import threading
from multiprocessing import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from slm import init_roberta
from layers import Classifier, TransformerModel, MLP, CoAttention, CrossAttention
from Preliminary.disentanglers import encode_content, HierarchicalDisentangler
# from Preliminary.llm_agents import load_data, run_llm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup  # AdamW,
from Preliminary.parallel_request import run_script, evaluate_test_results
from Preliminary.ann_similar_users import NewsSearch
from toolkits import (save_dict, read_dict, dict_statistics, save_ndjson, read_ndjson,
                      read_ndjson_clean_save,
                      news_dataloader, get_text_data_batch, compute_metrics_print,
                      save_embedding_h5, read_embedding_h5, EarlyStopping)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

random.seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_access_lock = Lock()


def load_data_domain(mode, start_no, root_path="D:/WorkSpace/workspace/LLMFND/", get_subset=False, selected_user_num=500):
    base_path = root_path + "Results/Prelim/"
    if mode in ["politifact", "gossipcop"]:
        veracity_dict_path = base_path + "veracity_dict_1_fake"
        news_dict_path = base_path + "news_dict"
        u_at_dict_path = base_path + "u_at_dict"
        user_type_path = base_path + "user_type_dict"
        comments_dict_path = base_path + "comments_dict"
    elif mode == "covid":
        veracity_dict_path = base_path + "veracity_dict_1_fake_c"
        news_dict_path = base_path + "news_dict_c"
        u_at_dict_path = base_path + "u_at_dict_c"
        user_type_path = base_path + "user_type_dict_c"
        comments_dict_path = base_path + "comments_dict_c"
    else:
        raise ValueError("mode must be 'politifact', 'gossipcop', or 'covid'")

    save_path_load = root_path + f"Results/domain_aware_agents/{mode.upper()}/"
    if get_subset:
        save_path_load += f"{selected_user_num}/"
    else:
        save_path_load += f"all/"
    if not os.path.exists(save_path_load):
        os.makedirs(save_path_load)

    nidx_u_dict_path = save_path_load + "nidx_u_dict"
    uc_dict_path = save_path_load + "uc_dict"
    news_id_idx_dict_path = save_path_load + "news_id_idx_dict"
    news_idx_content_dict_path = save_path_load + "news_idx_content_dict"

    veracity_dict = read_dict(veracity_dict_path)
    news_dict_all = read_dict(news_dict_path)
    u_at_dict_all = read_dict(u_at_dict_path)
    user_type_dict = read_dict(user_type_path)
    comments_dict = read_dict(comments_dict_path)

    if mode in ["politifact", "gossipcop"]:
        news_ids = [nid for nid in news_dict_all.keys() if mode in nid]
    else:  # covid
        news_ids = list(news_dict_all.keys())

    domain_idx_dict = {}
    for i, news_id in enumerate(news_ids):
        domain_idx_dict[i+start_no] = veracity_dict[news_id]

    if os.path.exists(news_idx_content_dict_path + '.json') and os.path.exists(news_id_idx_dict_path + '.json'):
        news_idx_content_dict = read_dict(news_idx_content_dict_path)
        news_id_idx_dict = read_dict(news_id_idx_dict_path)
    else:
        news_idx_content_dict, news_id_idx_dict = {}, {}
        for i, news_id in enumerate(news_ids):
            news_idx_content_dict[str(i+start_no)] = news_dict_all[news_id]
            news_id_idx_dict[news_id] = str(i+start_no)
        save_dict(news_idx_content_dict_path, news_idx_content_dict)
        save_dict(news_id_idx_dict_path, news_id_idx_dict)

    nidx_u_dict, uc_dict, u_at_dict = {}, {}, {}
    all_user_ids = []
    if os.path.exists(uc_dict_path + '.json') and os.path.exists(nidx_u_dict_path + '.json'):
        uc_dict = read_dict(uc_dict_path)
        nidx_u_dict = read_dict(nidx_u_dict_path)
        u_at_dict = {uid: [rec for rec in u_at_dict_all[uid]
                           if rec[0] in news_id_idx_dict]
                     for uid in u_at_dict_all if uid in uc_dict}
        all_user_ids = list(u_at_dict.keys())
    else:
        for user_id, records in u_at_dict_all.items():
            filtered_records = []
            for news_domain_id, comment_id in records:
                if news_domain_id in news_id_idx_dict:
                    news_idx = news_id_idx_dict[news_domain_id]
                    nidx_u_dict.setdefault(news_idx, []).append(user_id)
                    uc_dict.setdefault(user_id, []).append(comment_id)
                    filtered_records.append([news_domain_id, comment_id])
            if filtered_records:
                u_at_dict[user_id] = filtered_records
                all_user_ids.append(user_id)
        save_dict(uc_dict_path, uc_dict)
        save_dict(nidx_u_dict_path, nidx_u_dict)

    if get_subset:
        subset_save_path = save_path_load + f"{selected_user_num}/"
        if not os.path.exists(subset_save_path):
            os.makedirs(subset_save_path)
        user_subset = random.sample(all_user_ids, min(selected_user_num, len(all_user_ids)))
        nidx_u_dict_subset, uc_dict_subset, u_at_dict_subset = {}, {}, {}
        news_subset = set()
        for user_id in user_subset:
            for news_domain_id, comment_id in u_at_dict[user_id]:
                news_idx = news_id_idx_dict[news_domain_id]
                news_subset.add(news_idx)
                nidx_u_dict_subset.setdefault(news_idx, []).append(user_id)
                uc_dict_subset.setdefault(user_id, []).append(comment_id)
                u_at_dict_subset.setdefault(user_id, []).append([news_domain_id, comment_id])

        print(f"[{mode} - Subset] Num of news: {len(news_subset)}, users: {len(user_subset)}")
        return (domain_idx_dict, news_idx_content_dict, nidx_u_dict_subset,
                uc_dict_subset, news_id_idx_dict, user_type_dict, comments_dict, u_at_dict_subset)

    else:
        print(f"[{mode}] Num of news: {len(nidx_u_dict)}, users: {len(all_user_ids)}")
        return (domain_idx_dict, news_idx_content_dict, nidx_u_dict, uc_dict,
                news_id_idx_dict, user_type_dict, comments_dict, u_at_dict)


# Losses of HDs
def hd_loss_calc(hd_output, veracity_labels, domain_labels, recon_ref, calc_lvl):
    if calc_lvl == "news":  # news level: news, news feature
        (vr_logits, vi_logits, dsh_logits, dsf_logits,
         vr_feature, vi_feature, dsh_feature, dsf_feature) = hd_output
    elif calc_lvl == "user" or calc_lvl == "comment":
        (vr_logits_list, vi_logits_list, dsh_logits_list, dsf_logits_list,
         vr_feature_list, vi_feature_list, dsh_feature_list, dsf_feature_list,
         recon_ref_list) = [], [], [], [], [], [], [], [], []
        if calc_lvl == "user":
            def get_mean(x): return x.mean(dim=1)
        elif calc_lvl == "comment":
            def get_mean(x): return x.mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown calc_lvl: {calc_lvl}")
        for hd_output_each_news in hd_output:
            try:
                tp, ts = hd_output_each_news
                vr_logits_list.append(get_mean(tp[0].unsqueeze(0)))
                vi_logits_list.append(get_mean(tp[1].unsqueeze(0)))
                dsh_logits_list.append(get_mean(tp[2].unsqueeze(0)))
                dsf_logits_list.append(get_mean(tp[3].unsqueeze(0)))
                vr_feature_list.append(tp[4])
                vi_feature_list.append(tp[5])
                dsh_feature_list.append(tp[6])
                dsf_feature_list.append(tp[7])
                recon_ref_list.append(ts)
            except Exception as e:
                print(f"\n[Error] in hd_loss_calc for {calc_lvl}: {e}")
        vr_logits = torch.cat(vr_logits_list, dim=0)
        vi_logits= torch.cat(vi_logits_list, dim=0)
        dsh_logits = torch.cat(dsh_logits_list, dim=0)
        dsf_logits = torch.cat(dsf_logits_list, dim=0)
        vr_feature = torch.cat(vr_feature_list, dim=0)
        vi_feature = torch.cat(vi_feature_list, dim=0)
        dsh_feature = torch.cat(dsh_feature_list, dim=0)
        dsf_feature = torch.cat(dsf_feature_list, dim=0)
        recon_ref = torch.cat(recon_ref_list, dim=0)
    else:
        raise ValueError(f"Unknown calc_lvl: {calc_lvl}")
    cls_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss(reduction='none')
    # [2.1]: Veracity Loss
    vr_loss = cls_criterion(vr_logits, veracity_labels)
    ad_vi_loss = torch.clamp(1 - cls_criterion(vi_logits, veracity_labels), min=0)
    v_loss = vr_loss + ad_vi_loss
    # [2.2]: Domain Loss
    dsf_loss = cls_criterion(dsf_logits, domain_labels)
    ad_dsh_loss = torch.clamp(1 - cls_criterion(dsh_logits, domain_labels), min=0)
    d_loss = dsf_loss + ad_dsh_loss
    if calc_lvl == "comment":
        vr_feature = vr_feature.view(-1, vr_feature.size(2), vr_feature.size(3))
        vi_feature = vi_feature.view(-1, vi_feature.size(2), vi_feature.size(3))
        dsh_feature = dsh_feature.view(-1, dsh_feature.size(2), dsh_feature.size(3))
        dsf_feature = dsf_feature.view(-1, dsf_feature.size(2), dsf_feature.size(3))
        recon_ref = recon_ref.view(-1, recon_ref.size(2), recon_ref.size(3))
    content_reconstruct = vr_feature + vi_feature
    whole_r1_loss = recon_criterion(content_reconstruct, recon_ref)
    # news / news feature (32,256,16); user memory (320,256,16); comments / hst news (user * comment,256,16)
    per_sample_r1_loss = whole_r1_loss.mean(dim=(1, 2))
    r1_loss = per_sample_r1_loss.mean(dim=0)
    vr_reconstruct = dsh_feature + dsf_feature
    whole_r2_loss = recon_criterion(vr_reconstruct, vr_feature)
    per_sample_r2_loss = whole_r2_loss.mean(dim=(1, 2))
    r2_loss = per_sample_r2_loss.mean(dim=0)
    return v_loss, d_loss, r1_loss, r2_loss


def run(mod_list, lm_list, plain_inputs, augment_inputs, setting_inputs, path_inputs, get_subset=False, model_type="train"):
    user_memory_dict, news_features_dict, flag_list = augment_inputs
    pre_training_hd, engagement_augment_mode, comment_embed_mode = flag_list

    (step, news_batch, news_content_batch, veracity_batch, domain_batch,
     nu_dict, uc_dict, comments_dict,
     u_at_dict, news_id_idx_dict, un_dict_4batch, news_idx_content_dict) = plain_inputs

    max_related_users_num, max_comments_num, selected_user_num, neighbor_num, run_mode, cml, nml, uml, neg_pos_weight = setting_inputs

    (news_embedding_path, news_features_embedding_path, comment_embedding_path,
     hst_news_embedding_path, hc_embedding_path) = path_inputs

    if "user replaced" in run_mode:
        (hd, v_classifier, comments_mlp, comments_tf, users_mlp, users_tf,
         news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
         user_memory_mlp, user_memory_tf,
         before_target_mlp, news_output_mlp, input_proj,
         memory_encode_mlp) = mod_list
    elif "news augmented" in run_mode and "user augmented" in run_mode:
        (hd, v_classifier, comments_mlp, comments_tf, users_mlp, users_tf,
         news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
         user_memory_mlp, user_memory_tf,
         before_target_mlp, news_output_mlp, input_proj,
         memory_encode_mlp, user_encode_mlp, news_attn, user_attn) = mod_list
    elif "news augmented" in run_mode:
        (hd, v_classifier, comments_mlp, comments_tf, users_mlp, users_tf,
         news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
         user_memory_mlp, user_memory_tf,
         before_target_mlp, news_output_mlp, input_proj,
         user_encode_mlp, news_attn) = mod_list
    elif "user augmented" in run_mode:
        (hd, v_classifier, comments_mlp, comments_tf, users_mlp, users_tf,
         news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
         user_memory_mlp, user_memory_tf,
         before_target_mlp, news_output_mlp, input_proj,
         memory_encode_mlp, user_encode_mlp, user_attn) = mod_list
    else:
        (hd, v_classifier, comments_mlp, comments_tf, users_mlp, users_tf,
         news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
         user_memory_mlp, user_memory_tf,
         before_target_mlp, news_output_mlp, input_proj,
         user_encode_mlp) = mod_list

    (news_input_proj, comment_input_proj, hst_news_input_proj, news_feature_input_proj,
     user_memory_input_proj) = input_proj

    hd_news, hd_news_feature, hd_user_profile, hd_comment = hd
    if not pre_training_hd:
        hd_news.zero_grad()
        hd_news_feature.zero_grad()
        hd_user_profile.zero_grad()
        hd_comment.zero_grad()

    news_embeddings_file_path = news_embedding_path + f"embedding/{model_type}/{step}"
    news_mask_file_path = news_embedding_path + f"mask/{model_type}/{step}"
    news_features_embeddings_file_path = news_features_embedding_path + f"embedding/{model_type}/{step}"
    news_features_mask_file_path = news_features_embedding_path + f"mask/{model_type}/{step}"

    try:
        news_embeddings_batch = read_embedding_h5("embedding", news_embeddings_file_path).to(DEVICE)
        news_mask_batch = read_embedding_h5("mask", news_mask_file_path).to(DEVICE)
        news_embeddings_batch_p = news_input_proj(news_embeddings_batch)
        # ==== 先用hd处理news content ===
        news_mask_batch = news_mask_batch.unsqueeze(-1)
        # dsh_logits1 (32, 2)
        hd_news_output = hd_news(news_embeddings_batch_p, news_mask_batch)
        news_embeddings_batch_p = hd_news_output[6]  # dsh_feature (32,256,16)
        # ==============================
        if "news replaced" in run_mode or "news augmented" in run_mode:
            news_features_embeddings_batch = read_embedding_h5("embedding",
                                                               news_features_embeddings_file_path).to(DEVICE)
            news_features_mask_batch = read_embedding_h5("mask",
                                                         news_features_mask_file_path).to(DEVICE)
            news_features_embeddings_batch_p = news_feature_input_proj(news_features_embeddings_batch)
            # ==== 先用hd处理news feature ===
            news_features_mask_batch = news_features_mask_batch.unsqueeze(-1)
            # dsh_logits2 (32, 2)
            hd_n_feature_output = hd_news_feature(news_features_embeddings_batch_p, news_features_mask_batch)
            news_features_embeddings_batch_p = hd_n_feature_output[6]  # dsh_feature (32,256,16)
            # =============================
            if "news replaced" in run_mode:  
                news_embeddings_batch = news_features_embeddings_batch_p
                news_mask_batch = news_features_mask_batch.unsqueeze(-1)
            elif "news augmented" in run_mode:  
                news_attended = news_attn(news_embeddings_batch_p, news_features_embeddings_batch_p,
                                          news_mask_batch.squeeze(-1), news_features_mask_batch.squeeze(-1))
                news_embeddings_batch = news_attended
                news_mask_batch = None 
        else: 
            news_embeddings_batch = news_embeddings_batch_p
            news_mask_batch = news_mask_batch.unsqueeze(-1)

    except FileNotFoundError:
        news_embeddings_batch, news_mask_batch = encode_content(lm_list, news_content_batch, max_length=nml)
        save_embedding_h5(news_embeddings_file_path, news_embeddings_batch, "embedding")
        save_embedding_h5(news_mask_file_path, news_mask_batch, "mask")
        # news_embeddings_batch_p (32,256,16)
        news_embeddings_batch_p = news_input_proj(news_embeddings_batch)
        news_mask_batch = news_mask_batch.unsqueeze(-1)
        # dsh_logits1 (32, 2)
        hd_news_output = hd_news(news_embeddings_batch_p, news_mask_batch)
        news_embeddings_batch_p = hd_news_output[6]  # dsh_feature (32,256,16)
        if "news replaced" in run_mode or "news augmented" in run_mode:
            new_features_list = []
            for news_idx in news_batch:
                news_idx = str(news_idx.item())
                each_content = news_features_dict[news_idx]
                new_features_list.append(each_content)
            news_features_embeddings_batch, news_features_mask_batch = encode_content(lm_list, new_features_list,
                                                                                      max_length=nml)
            save_embedding_h5(news_features_embeddings_file_path, news_features_embeddings_batch, "embedding")
            save_embedding_h5(news_features_mask_file_path, news_features_mask_batch, "mask")
            # news_features_embeddings_batch_p (32,256,16)
            news_features_embeddings_batch_p = news_feature_input_proj(news_features_embeddings_batch)
            news_features_mask_batch = news_features_mask_batch.unsqueeze(-1)
            # dsh_logits2 (32, 2)
            hd_n_feature_output = hd_news_feature(news_features_embeddings_batch_p, news_features_mask_batch)
            news_features_embeddings_batch_p = hd_n_feature_output[6]  # dsh_feature (32,256,16)
            if "news replaced" in run_mode:
                news_embeddings_batch = news_features_embeddings_batch_p
                news_mask_batch = news_features_mask_batch.unsqueeze(-1)
            elif "news augmented" in run_mode: 
                news_attended = news_attn(news_embeddings_batch_p, news_features_embeddings_batch_p,
                                          news_mask_batch.squeeze(-1), news_features_mask_batch.squeeze(-1))
                news_embeddings_batch = news_attended
                news_mask_batch = None
        else: 
            news_embeddings_batch = news_embeddings_batch_p
            news_mask_batch = news_mask_batch.unsqueeze(-1)

    news_content_feature_batch = news_embedding_mlp(news_embeddings_batch)
    news_content_feature_batch = news_content_feature_batch.view(news_content_feature_batch.size(0), -1)
    news_content_batch = news_output_mlp(news_content_feature_batch)

    user_memory_all_news, user_comment_all_news, user_enhanced_all_news = [], [], []
    hc_embeddings_file_path = hc_embedding_path + f"embedding/{model_type}/{step}"
    hc_mask_file_path = hc_embedding_path + f"mask/{model_type}/{step}"
    comment_corres_rows_file_path = hc_embedding_path + f"corres_rows_dict/{model_type}/{step}" 

    with file_access_lock:
        try:
            all_hc_embeddings = read_embedding_h5("embedding", hc_embeddings_file_path).to(DEVICE)
            all_hc_mask = read_embedding_h5("mask", hc_mask_file_path).to(DEVICE)
            corres_rows = read_dict(comment_corres_rows_file_path)
            comment_loaded = True
        except FileNotFoundError:
            all_hc_embeddings, all_hc_mask = [], []
            corres_rows = {}
            comment_loaded = False
        hd_u_profile_output_all_news, hd_comment_output_all_news, hd_hst_news_output_all_news = [], [], []
        news_which_has_profile, news_which_has_users = [], []
        for i_news, news_idx in enumerate(news_batch):
            news_idx = str(news_idx.item())
            related_users = list(set(nu_dict.get(news_idx, [])))

            if "user replaced" in run_mode or "user augmented" in run_mode:
                user_memory_text_all_users = []
                if len(related_users) == 0:
                    raise ValueError(f"News {news_idx} has no related users.")
                else:
                    if len(related_users) > max_related_users_num:
                        related_users = related_users[:max_related_users_num]
                    for user_id in related_users:
                        try:
                            user_memory_text_per_user = user_memory_dict[user_id]
                            user_memory_text_all_users.append(user_memory_text_per_user)
                        except KeyError:
                            user_memory_text_per_user = "This user has no memory." 
                            user_memory_text_all_users.append(user_memory_text_per_user)
                    user_memory_encoded_all_users, user_memory_mask_all_users = encode_content(lm_list,
                                                                                               user_memory_text_all_users,
                                                                                               max_length=uml)
                    user_memory_encoded_all_users_p = user_memory_input_proj(user_memory_encoded_all_users)
                    user_memory_encode_out_dim = user_memory_input_proj.output_size
                    max_sequence_length = user_memory_encoded_all_users.size(1)
                    user_memory_padding = torch.zeros(max_related_users_num - len(user_memory_text_all_users),
                                                      max_sequence_length, user_memory_encode_out_dim).to(DEVICE)
                    num_users, seq_len = user_memory_mask_all_users.shape
                    user_mask_padding = torch.zeros(max_related_users_num - num_users, seq_len).to(DEVICE)

                    user_memory_all_users = torch.cat((user_memory_encoded_all_users_p, user_memory_padding), dim=0)
                    user_memory_padding_mask = torch.cat((user_memory_mask_all_users, user_mask_padding), dim=0)
                    user_memory_padding_mask = user_memory_padding_mask.unsqueeze(-1)
                    # dsh_logits3 (max_related_users_num, 2)
                    hd_u_profile_output = hd_user_profile(user_memory_all_users, user_memory_padding_mask)
                    user_memory_all_users = hd_u_profile_output[6]
                    hd_u_profile_output_all_news.append([hd_u_profile_output, user_memory_all_users])
                    news_which_has_profile.append(i_news)

                    if "user replaced" in run_mode:
                        user_memory_features = user_memory_tf(user_memory_mlp(user_memory_all_users),
                                                              user_memory_padding_mask.squeeze(-1))
                        user_memory_features = user_memory_features.reshape(user_memory_features.size(0), -1)
                        user_memory_features_encoded = memory_encode_mlp(user_memory_features)
                        user_memory_all_news.append(user_memory_features_encoded.unsqueeze(0))  # (1, 10, 2560)

            if "user replaced" not in run_mode:
                user_padding_flag = False
                only_one_padding_user = 0
                if len(related_users) == 0:
                    comment_features = torch.zeros(1, max_comments_num, cml * hst_news_input_proj.output_size).to(DEVICE)
                    comment_padding_mask_2d = torch.zeros(1, max_comments_num).to(DEVICE)
                    engagement_features = comment_features
                    engagement_padding_mask_2d = comment_padding_mask_2d
                else:
                    if len(related_users) > max_related_users_num:
                        related_users = related_users[:max_related_users_num]
                        user_padding_flag = True
                    all_comments, comment_owner_indices, user_id_list_ordered = [], [], []
                    all_hst_news_articles, hst_owner_indices, all_comments_from_news = [], [], []
                    num_valid_users = len(related_users)
                    for user_id in related_users:
                        hst_news_idx_raw = un_dict_4batch.get(user_id, [])
                        if len(hst_news_idx_raw) == 0:
                            num_valid_users -= 1
                            continue
                        hst_news_idx_tensor = torch.tensor(hst_news_idx_raw) 
                        hst_news_idx = hst_news_idx_tensor[:max_comments_num]
                        hst_news_articles = get_text_data_batch(hst_news_idx, news_idx_content_dict)
                        hst_owner_indices.extend([user_id] * len(hst_news_articles))
                        all_hst_news_articles.extend(hst_news_articles)
                        # coresponding comments
                        at_list = u_at_dict.get(user_id, [])
                        at_dict4batch = {}
                        for a, t in at_list:
                            nidx = news_id_idx_dict[a]
                            if nidx in at_dict4batch:
                                at_dict4batch[nidx].append(t)
                            else:
                                at_dict4batch[nidx] = [t]
                        comment_id_from_news, comment_from_news = [], []
                        for hn in hst_news_idx:
                            cores_comment_id = at_dict4batch.get(str(hn.item()), [])
                            comment_id_from_news.append(cores_comment_id)
                            c4n = ""
                            for cid in cores_comment_id:
                                try:
                                    c = comments_dict[cid].split("###")[0]
                                except Exception:
                                    c = comments_dict[cid]
                                c4n += c + ";"
                            comment_from_news.append(c4n)
                        user_id_list_ordered.append(user_id)
                        comment_owner_indices.extend([user_id] * len(comment_from_news))
                        all_comments_from_news.extend(comment_from_news)
                        all_comments = all_comments_from_news

                    if num_valid_users < len(related_users):
                        user_padding_flag = False
                    if num_valid_users > 0:
                        if comment_loaded:
                            if news_idx not in corres_rows:
                                hc_embeddings = torch.zeros((1, 2 * cml, 768)).to(DEVICE)
                                hc_mask = torch.zeros((1, 2 * cml)).to(DEVICE)
                            else:
                                corres_row = corres_rows[news_idx]
                                hc_embeddings = all_hc_embeddings[corres_row[0]:corres_row[1]]
                                hc_mask = all_hc_mask[corres_row[0]:corres_row[1]]
                            hst_news_embeddings, comment_embeddings = torch.split(hc_embeddings, cml, dim=1)
                            hst_news_mask, comment_mask = torch.split(hc_mask, cml, dim=1)
                        else:
                            hst_news_embeddings, hst_news_mask = encode_content(lm_list, all_hst_news_articles,
                                                                                max_length=cml)
                            comment_embeddings, comment_mask = encode_content(lm_list, all_comments,
                                                                              max_length=cml)
                            if len(all_hc_embeddings) == 0:
                                corres_rows[news_idx] = (0, len(all_comments))
                            else:
                                total_len = torch.cat(all_hc_embeddings, dim=0).size(0)
                                corres_rows[news_idx] = (total_len,
                                                         total_len + len(all_comments))
                            hc_embeddings = torch.cat((hst_news_embeddings, comment_embeddings), dim=1)
                            hc_mask = torch.cat((hst_news_mask, comment_mask), dim=1)
                            all_hc_embeddings.append(hc_embeddings.cpu())
                            all_hc_mask.append(hc_mask.cpu())
                        hst_news_embeddings_p = hst_news_input_proj(hst_news_embeddings)
                        comment_embeddings_p = comment_input_proj(comment_embeddings)

                        num_users = len(user_id_list_ordered)
                        hidden_dim = comment_embeddings_p.size(1)
                        embedding_dim = comment_embeddings_p.size(2)
                        comment_embedding_tensor = torch.zeros(num_users, max_comments_num,
                                                               hidden_dim, embedding_dim).to(DEVICE)
                        comment_padding_mask = torch.zeros(num_users, max_comments_num, hidden_dim).to(DEVICE)
                        user_idx_map = {uid: idx for idx, uid in enumerate(user_id_list_ordered)}
                        cursor_per_user = defaultdict(int)
                        uidx_tensor1 = []
                        pos_tensor1 = []
                        for emb, mask, uid in zip(comment_embeddings_p, comment_mask, comment_owner_indices):
                            uidx = user_idx_map[uid]
                            pos = cursor_per_user[uid]
                            uidx_tensor1.append(uidx)
                            pos_tensor1.append(pos)
                            if pos < max_comments_num:
                                comment_embedding_tensor[uidx, pos] = emb
                                comment_padding_mask[uidx, pos] = mask
                                cursor_per_user[uid] += 1
                        comment_padding_mask = comment_padding_mask.unsqueeze(-1)

                        hidden_dim2 = hst_news_embeddings_p.size(1)
                        embedding_dim2 = hst_news_embeddings_p.size(2)
                        hst_news_embedding_tensor = torch.zeros(num_users, max_comments_num,
                                                                hidden_dim2, embedding_dim2).to(DEVICE)
                        hst_news_padding_mask = torch.zeros(num_users, max_comments_num, hidden_dim2).to(DEVICE)
                        cursor_per_user2 = defaultdict(int)
                        for emb, mask, uid in zip(hst_news_embeddings_p, hst_news_mask, hst_owner_indices):
                            uidx = user_idx_map[uid]
                            pos = cursor_per_user2[uid]
                            if pos < max_comments_num:
                                hst_news_embedding_tensor[uidx, pos] = emb
                                hst_news_padding_mask[uidx, pos] = mask
                                cursor_per_user2[uid] += 1
                        hst_news_padding_mask = hst_news_padding_mask.unsqueeze(-1)

                        hd_comment_output = hd_comment(comment_embedding_tensor, comment_padding_mask)
                        comments_embedding = hd_comment_output[6]
                        news_which_has_users.append(i_news)
                        hd_comment_output_all_news.append([hd_comment_output, comment_embedding_tensor])
                        
                        hd_hst_news_output = hd_news(hst_news_embedding_tensor, hst_news_padding_mask)
                        hst_news_embedding = hd_hst_news_output[6]
                        hd_hst_news_output_all_news.append([hd_hst_news_output, hst_news_embedding_tensor])
                       
                        comments_embedding_re = comments_embedding.view(-1, comments_embedding.size(2),
                                                                        comments_embedding.size(3))
                        comment_padding_mask_re = comment_padding_mask.view(-1, comment_padding_mask.size(2),
                                                                            comment_padding_mask.size(3))
                        hst_news_embedding_re = hst_news_embedding.view(-1, hst_news_embedding.size(2),
                                                                        hst_news_embedding.size(3))
                        hst_news_padding_mask_re = hst_news_padding_mask.view(-1, hst_news_padding_mask.size(2),
                                                                              hst_news_padding_mask.size(3))
                        hc_features_re = hc_attn(hst_news_embedding_re, comments_embedding_re,
                                                 hst_news_padding_mask_re.squeeze(-1),
                                                 comment_padding_mask_re.squeeze(-1))
                        hc_features = hc_features_re.view(num_users, max_comments_num,
                                                          hc_features_re.size(1), hc_features_re.size(2))
                        hst_news_padding_mask = hst_news_padding_mask_re.view(num_users, max_comments_num,
                                                                              hst_news_padding_mask_re.size(1),
                                                                              hst_news_padding_mask_re.size(2))
                        
                        engagement_features = hc_features.view(num_users, max_comments_num, -1)
                        hst_news_padding_mask_2d = hst_news_padding_mask.squeeze(-1).any(dim=-1).float()
                        engagement_padding_mask_2d = hst_news_padding_mask_2d
                    else:
                        comment_features = torch.zeros(1, max_comments_num, cml * hst_news_input_proj.output_size).to(DEVICE)
                        comment_padding_mask_2d = torch.zeros(1, max_comments_num).to(DEVICE)
                        engagement_features = comment_features
                        engagement_padding_mask_2d = comment_padding_mask_2d

                if user_padding_flag:
                    user_padding_mask = engagement_padding_mask_2d
                else:
                    user_out_dim = cml * hst_news_input_proj.output_size
                    a = engagement_features.size(0)
                    user_embedding_padding = torch.zeros(max_related_users_num - a,
                                                         max_comments_num, user_out_dim).to(DEVICE)
                    engagement_features_padding = torch.cat((engagement_features, user_embedding_padding), dim=0)
                    
                    user_padding_mask = torch.cat((engagement_padding_mask_2d,
                                                   torch.zeros(
                                                       max_related_users_num - a,
                                                       max_comments_num).to(DEVICE)), dim=0)
                    engagement_features = engagement_features_padding

                user_features = users_tf(users_mlp(engagement_features), user_padding_mask)
                user_features_encoded = user_encode_mlp(user_features).view(user_features.size(0), -1).unsqueeze(0)
                user_comment_all_news.append(user_features_encoded)
              
                if "user augmented" in run_mode:
                    user_memory_all_users = user_memory_all_users.view(max_related_users_num, -1).unsqueeze(1)
                    user_memory_padding_mask_flat = user_memory_padding_mask.squeeze(-1).any(dim=-1).float().unsqueeze(-1)
                    engagement_features = engagement_features.view(max_related_users_num, -1).unsqueeze(1)
                    user_padding_mask_flat = user_padding_mask.any(dim=-1).float().unsqueeze(-1)
                    engagement_enhanced_features = user_attn(user_memory_all_users, engagement_features,
                                                             user_memory_padding_mask_flat, user_padding_mask_flat)
                    engagement_enhanced_features = engagement_enhanced_features.squeeze(1).view(max_related_users_num,
                                                                                                max_comments_num, -1)
                    user_enhanced_features = users_tf(users_mlp(engagement_enhanced_features), user_padding_mask)
                    user_enhanced_features_encoded = user_encode_mlp(
                        user_enhanced_features).view(user_enhanced_features.size(0), -1).unsqueeze(0)
                    user_enhanced_all_news.append(user_enhanced_features_encoded)

    if not comment_loaded:
        save_embedding_h5(hc_embeddings_file_path, torch.cat(all_hc_embeddings, dim=0), "embedding")
        save_embedding_h5(hc_mask_file_path, torch.cat(all_hc_mask, dim=0), "mask")
        save_dict(comment_corres_rows_file_path, corres_rows)

    if "user replaced" in run_mode:
        user_features_all_news = torch.cat(user_memory_all_news, dim=0)
    elif "user augmented" in run_mode:
        user_features_all_news = torch.cat(user_enhanced_all_news, dim=0)
    else: 
        user_features_all_news = torch.cat(user_comment_all_news, dim=0)

    user_features_batch = user_feature_mlp(user_features_all_news).view(user_features_all_news.size(0), -1)
    user_features_batch2 = before_target_mlp(user_features_batch)
    target_embeddings = target_mlp(news_content_batch + user_features_batch2)
    target_logits = v_classifier(target_embeddings)

    # ================================
    #         Loss Calculation
    # ================================
    classification_criterion = nn.CrossEntropyLoss()
    main_veracity_loss = classification_criterion(target_logits, veracity_batch)

    if model_type == "train":
        hd_news_output_all_news = hd_news_output  # news embedding
        hd_news_losses = hd_loss_calc(hd_news_output_all_news, veracity_batch, domain_batch,
                                      news_embeddings_batch_p, "news")
        v_loss, d_loss, r1_loss, r2_loss = hd_news_losses
        loss_info = "news"
        if "news replaced" in run_mode or "news augmented" in run_mode:
            hd_n_feature_output_all_news = hd_n_feature_output  # news feature
            hd_n_feature_losses = hd_loss_calc(hd_n_feature_output_all_news, veracity_batch, domain_batch,
                                               news_features_embeddings_batch_p, "news")
            v_loss += hd_n_feature_losses[0]
            d_loss += hd_n_feature_losses[1]
            r1_loss += hd_n_feature_losses[2]
            r2_loss += hd_n_feature_losses[3]
            loss_info += "+news_feature"
        if "user replaced" in run_mode or "user augmented" in run_mode:
            hd_u_profile_losses = hd_loss_calc(hd_u_profile_output_all_news, veracity_batch[news_which_has_profile],
                                               domain_batch[news_which_has_profile], None,
                                               "user")
            v_loss += hd_u_profile_losses[0]
            d_loss += hd_u_profile_losses[1]
            r1_loss += hd_u_profile_losses[2]
            r2_loss += hd_u_profile_losses[3]
            loss_info += "+user_memory"
        if "user replaced" not in run_mode:
            hd_comment_losses = hd_loss_calc(hd_comment_output_all_news, veracity_batch[news_which_has_users],
                                             domain_batch[news_which_has_users], None, "comment")
            v_loss += hd_comment_losses[0]
            d_loss += hd_comment_losses[1]
            r1_loss += hd_comment_losses[2]
            r2_loss += hd_comment_losses[3]
            loss_info += "+comments"
            hd_hst_news_losses = hd_loss_calc(hd_hst_news_output_all_news, veracity_batch[news_which_has_users],
                                              domain_batch[news_which_has_users], None, "comment")
            v_loss += hd_hst_news_losses[0]
            d_loss += hd_hst_news_losses[1]
            r1_loss += hd_hst_news_losses[2]
            r2_loss += hd_hst_news_losses[3]
            loss_info += "+hst_news"

        loss_info += "."
        return main_veracity_loss, v_loss, d_loss, r1_loss, r2_loss
    elif model_type == "test":
        return target_logits, main_veracity_loss


def domain_aware_agents4cdfnd(
        root_path, selected_user_num, get_subset, is_one_round, domain_list, early_stop_delta, run_mode, optuna_flag,
        pre_training_hd, engagement_augment_mode, comment_embed_mode, dglr_proj_size, clf_proj_size,
        neighbor_num, max_augmented_user, max_related_users_num, max_comments_num,
        input_proj_out, comment_max_length, news_max_length, user_max_length, loss_weight1,
        tf_heads, tf_layers, tf_hid_amplify,
        mlp_comment_reduce, mlp_user_reduce, mlp_news_reduce, mlp_user_feature_output,
        mlp1_mode, mlp1_dropout, mlp2_mode, mlp2_dropout, mlp3_mode, mlp3_dropout, mlp4_mode, mlp4_dropout,
        hc_attention_type, hc_attn_dropout, hc_out_dropout, hc_attn_layer_norm,
        news_attention_type, user_attention_type, news_attn_dropout, news_out_dropout,
        user_attn_dropout, user_out_dropout, news_attn_layer_norm, user_attn_layer_norm,
        classifier_hidden, classifier_dropout,
        lr
        ):
    save_path = root_path + "Results/domain_aware_agents/"
    file_path = root_path + "Results/llm_agents/"
    domain_aware_agents_path = root_path + "Results/domain_aware_agents/"
    if get_subset:
        subset_info = str(selected_user_num)
    else:
        subset_info = "all_users"
    news_embedding_path = file_path + f"news/{subset_info}_{news_max_length}/"
    news_features_embedding_path = file_path + f"news_features/{subset_info}_{news_max_length}/"
    if "engagement augmented" in run_mode:
        if engagement_augment_mode == "both":
            mode_info1 = "augmented_comments_fn"  # from news
            mode_info2 = "augmented_hst_news_fn"
            mode_info3 = "augmented_hc_fn"
        else:
            mode_info1 = f"augmented_comments_fn_{engagement_augment_mode}"
            mode_info2 = f"augmented_hst_news_fn_{engagement_augment_mode}"
            mode_info3 = f"augmented_hc_fn_{engagement_augment_mode}"
        if len(domain_list) == 1:
            domain_info = domain_list[0]
        elif len(domain_list) == 2:
            domain_info = domain_list[0] + "_to_" + domain_list[1]
        elif len(domain_list) == 3:
            domain_info = domain_list[0][0].upper() + "&" + domain_list[1][0].upper() + "_to_" + domain_list[2][0].upper()
        else:
            raise ValueError("More Than 2 Domains Not Supported!")
        comment_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info1}/{domain_info}/"
                f"{subset_info}_{max_related_users_num}_{max_comments_num}_{comment_max_length}/")
        hst_news_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info2}/{domain_info}/"
                f"{subset_info}_{max_related_users_num}_{max_comments_num}_{comment_max_length}/")
        hc_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info3}/{domain_info}/"
                f"{subset_info}_{max_related_users_num}_{max_comments_num}_{comment_max_length}/")
    else:
        mode_info1 = "comments_fn"
        mode_info2 = "hst_news_fn"
        mode_info3 = "hc_fn"
        if len(domain_list) == 1:
            domain_info = domain_list[0]
        elif len(domain_list) == 2:
            domain_info = domain_list[0] + "_to_" + domain_list[1]
        elif len(domain_list) == 3:
            domain_info = domain_list[0][0].upper() + "&" + domain_list[1][0].upper() + "_to_" + domain_list[2][0].upper()
        else:
            raise ValueError("More Than 2 Domains Not Supported!")
        comment_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info1}/{domain_info}/{subset_info}_{comment_max_length}/")
        hst_news_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info2}/{domain_info}/{subset_info}_{comment_max_length}/")
        hc_embedding_path = (
                domain_aware_agents_path +
                f"{mode_info3}/{domain_info}/{subset_info}_{comment_max_length}/")

    path_inputs = [news_embedding_path, news_features_embedding_path, comment_embedding_path,
                   hst_news_embedding_path, hc_embedding_path]

    (politifact_idx_dict, p_news_idx_content_dict, p_nidx_u_dict, p_uc_dict, p_news_id_idx_dict,
     p_user_type_dict, p_comments_dict, p_u_at_dict) = load_data_domain("politifact", 0, root_path,
                                                                        get_subset=get_subset,
                                                                        selected_user_num=selected_user_num)
    (gossipcop_idx_dict, g_news_idx_content_dict, g_nidx_u_dict, g_uc_dict, g_news_id_idx_dict,
     g_user_type_dict, g_comments_dict, g_u_at_dict) = load_data_domain("gossipcop", len(politifact_idx_dict), root_path,
                                                                        get_subset=get_subset,
                                                                        selected_user_num=selected_user_num)
    (covid_idx_dict, c_news_idx_content_dict, c_nidx_u_dict, c_uc_dict, c_news_id_idx_dict,
     c_user_type_dict, c_comments_dict, c_u_at_dict) = load_data_domain("covid", len(politifact_idx_dict) + len(gossipcop_idx_dict), root_path)

    politifact_idx_dict_filtered, gossipcop_idx_dict_filtered, covid_idx_dict_filtered = {}, {}, {}
    for n_idx, v in list(politifact_idx_dict.items()):
        if str(n_idx) in p_nidx_u_dict:
            politifact_idx_dict_filtered[n_idx] = v
    for n_idx, v in list(gossipcop_idx_dict.items()):
        if str(n_idx) in g_nidx_u_dict:
            gossipcop_idx_dict_filtered[n_idx] = v
    for n_idx, v in list(covid_idx_dict.items()):
        if str(n_idx) in c_nidx_u_dict:
            covid_idx_dict_filtered[n_idx] = v
    politifact_idx_dict = politifact_idx_dict_filtered
    gossipcop_idx_dict = gossipcop_idx_dict_filtered
    covid_idx_dict = covid_idx_dict_filtered

    news_idx_content_dict = {**p_news_idx_content_dict, **g_news_idx_content_dict, **c_news_idx_content_dict}
    nidx_u_dict = {**p_nidx_u_dict, **g_nidx_u_dict, **c_nidx_u_dict}
    uc_dict = {**p_uc_dict, **g_uc_dict, **c_uc_dict}
    news_id_idx_dict = {**p_news_id_idx_dict, **g_news_id_idx_dict, **c_news_id_idx_dict}
    user_type_dict = {**p_user_type_dict, **g_user_type_dict, **c_user_type_dict}
    comments_dict = {**p_comments_dict, **g_comments_dict, **c_comments_dict}
    u_at_dict = {**p_u_at_dict, **g_u_at_dict, **c_u_at_dict}

    print(f"Politifact News Count in subset: {len(politifact_idx_dict)}, "
          f"Gossipcop News Count in subset: {len(gossipcop_idx_dict)}, "
          f"Covid News Count in subset: {len(covid_idx_dict)}")

    dict_statistics(nidx_u_dict, "[News - Users]")
    dict_statistics(uc_dict, "[User - Comments]")

    # (0) pathes
    news_features_dict_path = save_path + "news_features_dict"
    user_memory_dict_path = save_path + f"{selected_user_num}/user_memory_dict"
    comment_features_dict_path = save_path + f"{selected_user_num}/comment_features_dict"
    test_agent_results_path = save_path + f"{selected_user_num}/test_output.jsonl"

    similarity_threshold = 30
    similar_news_dict_path = save_path + f"{selected_user_num}/similar_news_dict_{similarity_threshold}"

    augmented_nu_dict_path = save_path + f"{selected_user_num}/augmented_nu_dict_{similarity_threshold}"
    augmented_uc_dict_path = save_path + f"{selected_user_num}/augmented_uc_dict_{similarity_threshold}"
    augmented_comments_dict_path = save_path + f"{selected_user_num}/augmented_comments_dict_{similarity_threshold}"
    # best_disentangler_path = root_path + "Results/disentangler/saved_[test]BEST_default/best_disentangler_3d.pt"

    news_features_input_dicts = [news_id_idx_dict, news_idx_content_dict]
    # TODO run_script(save_path, selected_user_num, news_features_input_dicts, mode="news_features")
    news_features_dict = read_ndjson_clean_save(news_features_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson

    user_agent_input_dicts = [u_at_dict, news_id_idx_dict, news_idx_content_dict]
    run_script(save_path, selected_user_num, user_agent_input_dicts, mode="user_agent_train")
    user_memory_dict = read_ndjson_clean_save(user_memory_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson

    run_script(save_path, selected_user_num, user_agent_input_dicts, mode="user_agent_test")
    evaluate_test_results(test_agent_results_path)

    comment_features_input_dicts = [u_at_dict, news_id_idx_dict, news_idx_content_dict, comments_dict]
    run_script(save_path, selected_user_num, comment_features_input_dicts, mode="comment_features")
    comment_features_dict = read_ndjson_clean_save(comment_features_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson

    lm = init_roberta(root_path=str(Path(root_path).parent))
    try:
        similar_news_dict = read_dict(similar_news_dict_path)
    except FileNotFoundError:
        idx_dict_list = [politifact_idx_dict, gossipcop_idx_dict, covid_idx_dict]
        newssearch = NewsSearch(neighbor_num, similarity_threshold,
                                idx_dict_list,
                                news_features_dict, lm, selected_user_num, get_subset,
                                root_path)
        similar_news_dict = newssearch.engagement_augmentation(u_at_dict, news_id_idx_dict)
        save_dict(similar_news_dict_path, similar_news_dict)

    news_user_dict, similar_news_dict_2 = {}, {}
    for user_id in list(similar_news_dict.keys()):
        similar_news_list = similar_news_dict[user_id]
        deleted_news_list = []
        for news_idx in similar_news_list:
            if news_idx not in news_user_dict:
                news_user_dict[news_idx] = []
            if len(news_user_dict[news_idx]) < max_augmented_user:
                news_user_dict[news_idx].append(user_id)
            else:
                deleted_news_list.append(news_idx)

    max_news_per_user = 60
    for news_idx, users in news_user_dict.items():
        for user_id in users:
            if user_id not in similar_news_dict_2:
                similar_news_dict_2[user_id] = []
            if len(similar_news_dict_2[user_id]) >= max_news_per_user:
                continue
            similar_news_dict_2[user_id].append(news_idx)
    similar_news_dict = similar_news_dict_2  # user_id的similar_news_list

    comment_generation_input_dicts = [u_at_dict, news_id_idx_dict, news_idx_content_dict,
                                      similar_news_dict]
    run_script(save_path, selected_user_num, comment_generation_input_dicts, mode="comment_generation")
    augmented_nu_dict = read_ndjson_clean_save(augmented_nu_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson
    augmented_uc_dict = read_ndjson_clean_save(augmented_uc_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson
    augmented_comments_dict = read_ndjson_clean_save(augmented_comments_dict_path)  # first time, read_ndjson_clean_save; otherwise, read_ndjson

    best_metric = 0.0
    for i in range(len(domain_list)):
        train_news_idx_dict, test_news_idx_dict, domain_labels, domain_test_labels = {}, {}, [], []
        if i > 2: 
            break
        if len(domain_list) == 3:
            train_domain1, train_domain2, test_domain = domain_list[0], domain_list[1], domain_list[2]
            domain_map = {
                "politifact": (politifact_idx_dict, 0),
                "gossipcop": (gossipcop_idx_dict, 1),
                "covid": (covid_idx_dict, 2)
            }
            for td in [train_domain1, train_domain2]:
                idx_dict, label = domain_map[td]
                train_news_idx_dict.update(idx_dict)
                domain_labels.extend([label] * len(idx_dict))
            test_news_idx_dict, test_label = domain_map[test_domain]
            domain_test_labels = [test_label] * len(test_news_idx_dict)
            train_domain = f"{train_domain1} & {train_domain2}"
        else: 
            train_domain, test_domain = domain_list[i], domain_list[i]
            test_size = 0.2
            if train_domain == "politifact":
                news_idxs = list(politifact_idx_dict.keys())
                veracity_labels = list(politifact_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs,
                                                                                    veracity_labels,
                                                                                    test_size=test_size,
                                                                                    random_state=42,
                                                                                    shuffle=True,
                                                                                    stratify=veracity_labels)
                train_news_idx_dict = {idx: label for idx, label in zip(train_idxs, train_labels)}
                test_news_idx_dict = {idx: label for idx, label in zip(test_idxs, test_labels)}
                domain_labels = [0] * len(train_news_idx_dict)
                domain_test_labels = [0] * len(test_news_idx_dict)
            elif train_domain == "gossipcop":
                news_idxs = list(gossipcop_idx_dict.keys())
                veracity_labels = list(gossipcop_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs, veracity_labels,
                                                                                    test_size=test_size,
                                                                                    random_state=42,
                                                                                    shuffle=True,
                                                                                    stratify=veracity_labels)
                train_news_idx_dict = {idx: label for idx, label in zip(train_idxs, train_labels)}
                test_news_idx_dict = {idx: label for idx, label in zip(test_idxs, test_labels)}
                domain_labels = [1] * len(train_news_idx_dict)
                domain_test_labels = [1] * len(test_news_idx_dict)
            elif train_domain == "covid":
                news_idxs = list(covid_idx_dict.keys())
                veracity_labels = list(covid_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs, veracity_labels,
                                                                                    test_size=test_size,
                                                                                    random_state=42,
                                                                                    shuffle=True,
                                                                                    stratify=veracity_labels)
                train_news_idx_dict = {idx: label for idx, label in zip(train_idxs, train_labels)}
                test_news_idx_dict = {idx: label for idx, label in zip(test_idxs, test_labels)}
                domain_labels = [2] * len(train_news_idx_dict)
                domain_test_labels = [2] * len(test_news_idx_dict)
        if len(domain_list) == 3:
            info = f"\n[{run_mode.title()}] Train on {train_domain1} & {train_domain2}, Test on {test_domain}"
        else:
            info = f"\n[{run_mode.title()}] Train on {train_domain}, Test on {test_domain}"
        print(info)
        print(f"Train News Count: {len(train_news_idx_dict)}, Test News Count: {len(test_news_idx_dict)}")

        start_running_time = datetime.datetime.now()
        start_running_timestamp = start_running_time.strftime("%Y%m%d_%H%M%S")

        epochs = 100
        batch_size, eps = 32, 1e-8

        x_set, y_set = np.array(list(train_news_idx_dict.keys())), np.array(list(train_news_idx_dict.values()))
        x_test_set, y_test_set = np.array(list(test_news_idx_dict.keys())), np.array(list(test_news_idx_dict.values()))
        train_loader = news_dataloader(x_set, y_set, domain_labels, batch_size)
        test_loader = news_dataloader(x_test_set, y_test_set, domain_test_labels, batch_size)

        y = torch.tensor(y_set)
        neg = (y == 0).sum() 
        pos = (y == 1).sum()  
        print("neg:", neg.item(), "pos:", pos.item())
        neg_pos_weight = torch.tensor([1.0, neg / pos]).float().to(DEVICE)

        converted_shape2, dglr_v_proj, dglr_d_proj = 768, dglr_proj_size, dglr_proj_size
        clf_v_proj, clf_d_proj, n_classes = clf_proj_size, clf_proj_size, 2
        alpha1, alpha2, alpha3, alpha4, alpha5 = 1, 1, 1, 10, 10  # Normalize the loss terms to a similar scale.
        w1, w2, w3, w4, w5 = loss_weight1, 1, 1, 1, 1  # weights for loss terms

        news_input_proj = MLP(converted_shape2, input_proj_out,
                              mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE) 

        comment_input_proj_out = input_proj_out
        comment_input_proj = MLP(converted_shape2, comment_input_proj_out,
                                 mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE) 
        hst_news_input_proj = MLP(converted_shape2, comment_input_proj_out,
                                  mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE)  

        news_feature_input_proj = MLP(converted_shape2, input_proj_out,
                                      mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE) 
        user_memory_input_proj_out = int(comment_input_proj_out * comment_max_length /
                                         user_max_length * max_comments_num)
        user_memory_input_proj = MLP(converted_shape2, user_memory_input_proj_out,
                                     mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE) 

        input_proj = [news_input_proj, comment_input_proj, hst_news_input_proj, news_feature_input_proj,
                      user_memory_input_proj]

        hd_news = HierarchicalDisentangler(input_proj_out, dglr_v_proj, dglr_d_proj,
                                           clf_v_proj, clf_d_proj, n_classes,
                                           news_max_length, comment_max_length).to(DEVICE)
        hd_news_feature = HierarchicalDisentangler(input_proj_out, dglr_v_proj, dglr_d_proj,
                                                   clf_v_proj, clf_d_proj, n_classes,
                                                   news_max_length, comment_max_length).to(DEVICE)
        hd_user_profile = HierarchicalDisentangler(user_memory_input_proj_out, dglr_v_proj, dglr_d_proj,
                                                   clf_v_proj, clf_d_proj, n_classes,
                                                   news_max_length, comment_max_length).to(DEVICE)
        hd_comment = HierarchicalDisentangler(comment_input_proj_out, dglr_v_proj, dglr_d_proj,
                                              clf_v_proj, clf_d_proj, n_classes,
                                              news_max_length, comment_max_length).to(DEVICE)

        best_hd_news_path = root_path + "Results/disentangler/saved_HDs/hd_news.pt"
        best_hd_news_feature_path = root_path + "Results/disentangler/saved_HDs/hd_news_feature.pt"
        best_hd_user_profile_path = root_path + "Results/disentangler/saved_HDs/hd_user_profile.pt"
        best_hd_comment_path = root_path + "Results/disentangler/saved_HDs/hd_comment.pt"
        if not os.path.exists(root_path + "Results/disentangler/saved_HDs/"):
            os.makedirs(root_path + "Results/disentangler/saved_HDs/")

        if not pre_training_hd:
            cp1 = torch.load(best_hd_news_path, map_location=DEVICE)
            state_dict1 = cp1['model_state_dict']
            hd_news.load_state_dict(state_dict1)

            cp2 = torch.load(best_hd_news_feature_path, map_location=DEVICE)
            state_dict2 = cp2['model_state_dict']
            hd_news_feature.load_state_dict(state_dict2)

            cp3 = torch.load(best_hd_user_profile_path, map_location=DEVICE)
            state_dict3 = cp3['model_state_dict']
            hd_user_profile.load_state_dict(state_dict3)

            cp4 = torch.load(best_hd_comment_path, map_location=DEVICE)
            state_dict4 = cp4['model_state_dict']
            hd_comment.load_state_dict(state_dict4)

        hd = [hd_news, hd_news_feature, hd_user_profile, hd_comment]

        project_out6 = int(comment_input_proj_out / mlp_comment_reduce)
        project_in1, project_out1 = (project_out6, project_out6)
        input_channel1, heads1, hidden1, layers_num1 = project_out1, tf_heads, project_out1 * tf_hid_amplify, tf_layers

        comment_encode_mlp = MLP(comment_input_proj_out, project_out6,
                                 mode=mlp2_mode, dropout=mlp2_dropout).to(DEVICE)
        hst_news_encode_mlp = MLP(comment_input_proj_out, project_out6,
                                  mode=mlp2_mode, dropout=mlp2_dropout).to(DEVICE)
        comments_mlp = MLP(project_in1 * comment_max_length, project_out1,
                           mode=mlp3_mode, dropout=mlp3_dropout).to(DEVICE)  
        comments_tf = TransformerModel(d_model=input_channel1, nhead=heads1, d_hid=hidden1, nlayers=layers_num1,
                                       dropout=0.1).to(DEVICE)
        
        project_in2, project_out2 = comment_input_proj_out * comment_max_length, comment_input_proj_out * comment_max_length
        input_channel2, heads2, hidden2, layers_num2 = project_out2, tf_heads, project_out2 * tf_hid_amplify, tf_layers
        project_in7, project_out7 = project_out2, int(project_out2 / mlp_user_reduce)

        users_mlp = MLP(project_in2, project_out2,
                        mode=mlp3_mode, dropout=mlp3_dropout).to(DEVICE)
        user_memory_mlp = MLP(user_memory_input_proj_out, user_memory_input_proj_out,
                              mode=mlp3_mode, dropout=mlp3_dropout).to(DEVICE)
        users_tf = TransformerModel(d_model=input_channel2, nhead=heads2, d_hid=hidden2, nlayers=layers_num2,
                                    dropout=0.1).to(DEVICE)
        user_memory_tf = TransformerModel(d_model=user_memory_input_proj_out, nhead=heads2, d_hid=hidden2,
                                          nlayers=layers_num2, dropout=0.1).to(DEVICE)
        user_encode_mlp = MLP(project_in7, project_out7,
                              mode=mlp2_mode, dropout=mlp2_dropout).to(DEVICE)
        project_out3 = int(input_proj_out / mlp_news_reduce)
        project_in4, project_out4 = max_comments_num * project_out7, mlp_user_feature_output
        project_in8, project_out8 = max_related_users_num * project_out4, project_out3
        project_in5, project_out5 = project_out3, project_out3

        news_embedding_mlp = MLP(input_proj_out, project_out3,
                                 mode=mlp2_mode, dropout=mlp2_dropout).to(DEVICE) 
        news_output_mlp = MLP(news_max_length * project_out3, project_out3,
                              mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE) 
        user_feature_mlp = MLP(project_in4, project_out4,
                               mode=mlp2_mode, dropout=mlp2_dropout, n_layers=3).to(DEVICE)  
        before_target_mlp = MLP(project_in8, project_out8,
                              mode=mlp1_mode, dropout=mlp1_dropout).to(DEVICE)  
        target_mlp = MLP(project_in5, project_out5,
                         mode=mlp4_mode, dropout=mlp4_dropout).to(DEVICE)  

        classifier = Classifier(project_out3, classifier_hidden, 2, dropout=classifier_dropout).to(DEVICE)

        query_dim3, key_dim3, hidden_dim3, out_dim3 = (comment_input_proj_out, comment_input_proj_out,
                                                       comment_input_proj_out, comment_input_proj_out)
        if hc_attention_type == "cross":
            hc_attn = CrossAttention(query_dim3, key_dim3, hidden_dim3, out_dim3,
                                     hc_attn_dropout, hc_out_dropout, hc_attn_layer_norm).to(DEVICE)
        elif hc_attention_type == "co":
            hc_attn = CoAttention(query_dim3, key_dim3, hidden_dim3, out_dim3,
                                  hc_attn_dropout, hc_out_dropout, hc_attn_layer_norm).to(DEVICE)
        else:
            raise ValueError(f"Unknown hc_attention_type: {hc_attention_type}")

        models = [hd, classifier, comments_mlp, comments_tf, users_mlp, users_tf,
                  news_embedding_mlp, user_feature_mlp, target_mlp, comment_encode_mlp, hst_news_encode_mlp, hc_attn,
                  user_memory_mlp, user_memory_tf,
                  before_target_mlp, news_output_mlp, input_proj]

        # model 3: memory feature
        if "user replaced" in run_mode or "user augmented" in run_mode:
            memory_encode_mlp = MLP(project_in7 * max_comments_num, project_out7 * max_comments_num,
                                    mode=mlp2_mode, dropout=mlp2_dropout).to(DEVICE) 
            models.append(memory_encode_mlp)
        if "user replaced" not in run_mode:
            models.append(user_encode_mlp)

        if "engagement augmented" in run_mode:
            if engagement_augment_mode == "user_only":
                for key in augmented_nu_dict.keys():
                    if key in nidx_u_dict:
                        nidx_u_dict[key].extend(augmented_nu_dict[key])
                    else:
                        nidx_u_dict[key] = augmented_nu_dict[key]
            elif engagement_augment_mode == "comment_only":
                for key in augmented_uc_dict.keys():
                    if key in uc_dict:
                        uc_dict[key].extend(augmented_uc_dict[key])
                    else:
                        uc_dict[key] = augmented_uc_dict[key]
            elif engagement_augment_mode == "both":
                for key in augmented_nu_dict.keys():
                    if key in nidx_u_dict:
                        nidx_u_dict[key].extend(augmented_nu_dict[key])
                    else:
                        nidx_u_dict[key] = augmented_nu_dict[key]
                for key in augmented_uc_dict.keys():
                    if key in uc_dict:
                        uc_dict[key].extend(augmented_uc_dict[key])
                    else:
                        uc_dict[key] = augmented_uc_dict[key]
            else:
                raise ValueError(f"Unknown engagement_augment_mode: {engagement_augment_mode}")
            comments_dict.update(augmented_comments_dict)
            dict_statistics(nidx_u_dict, "[Augmented][News - Users]")
            dict_statistics(uc_dict, "[Augmented][User - Comments]")

        # model 4: memory feature
        if "news augmented" in run_mode:
            query_dim1, key_dim1, hidden_dim1, out_dim1 = (input_proj_out,
                                                           input_proj_out,
                                                           input_proj_out,
                                                           input_proj_out) 
            if news_attention_type == "cross":
                news_attn = CrossAttention(query_dim1, key_dim1, hidden_dim1, out_dim1,
                                           news_attn_dropout, news_out_dropout, news_attn_layer_norm).to(DEVICE)
            elif news_attention_type == "co":
                news_attn = CoAttention(query_dim1, key_dim1, hidden_dim1, out_dim1,
                                        news_attn_dropout, news_out_dropout, news_attn_layer_norm).to(DEVICE)
            else:
                raise ValueError(f"Unknown news_attention_type: {news_attention_type}")
            models.append(news_attn)

        if "user augmented" in run_mode:
            query_dim2, key_dim2, hidden_dim2, out_dim2 = (project_in7 * max_comments_num,
                                                           project_in7 * max_comments_num,
                                                           project_in7 * max_comments_num,
                                                           project_in7 * max_comments_num)
            if user_attention_type == "cross":
                user_attn = CrossAttention(query_dim2, key_dim2, hidden_dim2, out_dim2,
                                           user_attn_dropout, user_out_dropout, user_attn_layer_norm).to(DEVICE)
            elif user_attention_type == "co":
                user_attn = CoAttention(query_dim2, key_dim2, hidden_dim2, out_dim2,
                                        user_attn_dropout, user_out_dropout, user_attn_layer_norm).to(DEVICE)
            else:
                raise ValueError(f"Unknown user_attention_type: {user_attention_type}")
            models.append(user_attn)

        flag_list = [pre_training_hd, engagement_augment_mode, comment_embed_mode]
        augment_inputs = [user_memory_dict, news_features_dict, flag_list]
        setting_inputs = [max_related_users_num, max_comments_num, selected_user_num, neighbor_num, run_mode,
                          comment_max_length, news_max_length, user_max_length, neg_pos_weight]

        params = []
        for m in models:
            if isinstance(m, list):
                for sub_m in m:
                    for param in sub_m.parameters():
                        if param.requires_grad:
                            params.append(param)
                            assert not torch.isnan(param).any(), "NaN detected in model parameters!"
            else:
                for param in m.parameters():
                    if param.requires_grad:
                        params.append(param)
                        assert not torch.isnan(param).any(), "NaN detected in model parameters!"
        optimizer = torch.optim.AdamW(params, lr=lr, eps=eps)
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)

        early_stopping = EarlyStopping(patience=10, delta=early_stop_delta)

        best_loss = float("inf")
        for epoch in range(0, epochs):
            # ========================================
            #               Training
            # ========================================
            print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

            total_train_loss = 0
            for m in models:
                if isinstance(m, list):
                    for sub_m in m:
                        sub_m.train()
                else:
                    m.train()
            train_bar = tqdm(enumerate(train_loader))

            for step, batch in train_bar:
                news_batch = batch[0].to(DEVICE)
                veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                content_list = get_text_data_batch(news_batch, news_idx_content_dict)
                plain_inputs = [step, news_batch, content_list, veracity_label, domain_label, nidx_u_dict,
                                uc_dict, comments_dict]

                un_dict_4batch = {}
                for nidx, user_list in nidx_u_dict.items():
                    nidx = int(nidx)
                    news_batch_list_int = [int(x) for x in batch[0].tolist()]
                    test_news_idx_list = list(test_news_idx_dict.keys())
                    if nidx not in test_news_idx_list and nidx not in news_batch_list_int:
                        for uid in user_list:
                            if uid not in un_dict_4batch:
                                un_dict_4batch[uid] = []
                            if nidx not in un_dict_4batch[uid]:
                                un_dict_4batch[uid].append(nidx)
                plain_inputs.extend([u_at_dict, news_id_idx_dict, un_dict_4batch, news_idx_content_dict])

                main_veracity_loss, v_loss, d_loss, r1_loss, r2_loss = run(models, lm, plain_inputs,
                                                                           augment_inputs, setting_inputs,
                                                                           path_inputs, get_subset=get_subset)
                if not pre_training_hd:
                    loss = main_veracity_loss
                else:
                    loss = (alpha2 * v_loss * w2 + alpha3 * d_loss * w3 + alpha4 * r1_loss * w4 + alpha5 * r2_loss * w5)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss = total_train_loss + loss.detach().cpu()
                optimizer.zero_grad()

            train_bar.close()
            avg_train_loss = total_train_loss / len(train_loader)
            print("\n  Average training loss: {}".format(avg_train_loss))

            if pre_training_hd:  
                best_loss1 = save_best_model(best_hd_news_path, hd_news, optimizer,
                                             epoch, best_loss, avg_train_loss)
                best_loss2 = save_best_model(best_hd_news_feature_path, hd_news_feature, optimizer,
                                             epoch, best_loss, avg_train_loss)
                best_loss3 = save_best_model(best_hd_user_profile_path, hd_user_profile, optimizer,
                                             epoch, best_loss, avg_train_loss)
                best_loss4 = save_best_model(best_hd_comment_path, hd_comment, optimizer,
                                             epoch, best_loss, avg_train_loss)
                best_loss = min(best_loss1, best_loss2, best_loss3, best_loss4)

            if ((epoch + 1) % 5 == 0 or (epoch + 1) >= epochs * 0.8) and not pre_training_hd: 
                # ========================================
                #               Testing
                # ========================================
                test_bar = tqdm(enumerate(test_loader))

                (test_target_logits, test_dsh_v_logits, test_dsf_v_logits, test_vrre_logits,
                 test_veracity_label) = [], [], [], [], []
                test_loss_list = []
                for step, batch in test_bar:
                    news_batch = batch[0].to(DEVICE)
                    veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                    domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                    content_list = get_text_data_batch(news_batch, news_idx_content_dict)
                    plain_inputs = [step, news_batch, content_list, veracity_label, domain_label, nidx_u_dict,
                                    uc_dict, comments_dict]

                    un_dict_4batch = {}
                    for nidx, user_list in nidx_u_dict.items():
                        nidx = int(nidx)
                        news_batch_list_int = [int(x) for x in batch[0].tolist()]
                        test_news_idx_list = list(test_news_idx_dict.keys())
                        if nidx not in test_news_idx_list and nidx not in news_batch_list_int:
                            for uid in user_list:
                                if uid not in un_dict_4batch:
                                    un_dict_4batch[uid] = []
                                if nidx not in un_dict_4batch[uid]:
                                    un_dict_4batch[uid].append(nidx)
                    plain_inputs.extend([u_at_dict, news_id_idx_dict, un_dict_4batch, news_idx_content_dict])

                    with torch.no_grad():
                        # (target_logits, dsh_v_logits, dsf_v_logits, vrre_logits, main_veracity_loss, v_loss,
                        #  d_loss, r1_loss, r2_loss) = run(models, lm, plain_inputs, augment_inputs, setting_inputs,
                        #                                  path_inputs, get_subset=get_subset, model_type="test")
                        (target_logits, main_veracity_loss) = run(models, lm, plain_inputs, augment_inputs,
                                                                  setting_inputs, path_inputs,
                                                                  get_subset=get_subset, model_type="test")
                        test_loss = main_veracity_loss
                    test_target_logits.append(target_logits)
                    # test_dsh_v_logits.append(dsh_v_logits)
                    # test_dsf_v_logits.append(dsf_v_logits)
                    # test_vrre_logits.append(vrre_logits)
                    test_veracity_label.append(veracity_label)
                    test_loss_list.append(test_loss.cpu().detach().numpy())

                test_target_logits = torch.cat(test_target_logits, dim=0).cpu()
                # test_dsh_v_logits = torch.cat(test_dsh_v_logits, dim=0).cpu()
                # test_dsf_v_logits = torch.cat(test_dsf_v_logits, dim=0).cpu()
                # test_vrre_logits = torch.cat(test_vrre_logits, dim=0).cpu()
                test_veracity_label = torch.cat(test_veracity_label, dim=0).cpu()

                _, _, _, _, test_macro_f1, _ = compute_metrics_print(test_target_logits, test_veracity_label,
                                                                     f"Veracity Prediction on {test_domain.title()}",
                                                                     has_return=True)

                early_stopping(test_macro_f1)
                if early_stopping.early_stop:
                    print(f"Stopped at epoch:{epoch}, with best_macro_f1:{early_stopping.best_score}")
                    break


        end_running_time = datetime.datetime.now()
        end_running_timestamp = end_running_time.strftime("%Y%m%d_%H%M%S")
        running_time = end_running_time - start_running_time
        print("[ {} -> {} ] Start Time: {}, End Time: {}, Running Time: {}".format(train_domain, test_domain,
                                                                                   start_running_timestamp,
                                                                                   end_running_timestamp, running_time))
        best_metric = early_stopping.best_score
        if is_one_round:
            break

    if optuna_flag:
        return best_metric


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def save_best_model(best_model_path, hd, optimizer, epoch, best_loss, current_loss):
    # Save the best hd
    if current_loss < best_loss:
        best_loss = current_loss
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            saved_loss = checkpoint.get('val_loss', float('inf'))
            if best_loss < saved_loss:
                print(f"\n New best HD ! Old Loss: {saved_loss:.6f}, New Loss: {best_loss:.6f}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': hd.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_loss,
                }, best_model_path)
            else:
                print(
                    f"\n Current Loss {best_loss:.6f} not better than saved Loss {saved_loss:.6f}. HD not saved."
                )
        else:
            print(f"\n Saving first best HD with Loss: {best_loss:.6f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': hd.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }, best_model_path)
    return best_loss


def run_domain_aware_agents():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--root_path")
    parser.add_argument("--selected_user_num", default=10000)
    parser.add_argument("--get_subset", default=True)  
    parser.add_argument("--is_one_round", default=True)
    parser.add_argument("-d", "--domain_list", nargs="+",
                        default=["politifact", "gossipcop", "covid"], help="List of domains")
    parser.add_argument("-delta", "--early_stop_delta", type=float, default=1e-5)
    parser.add_argument("-rn", "--run_name", default="debug", required=True) 
    parser.add_argument("--run_mode", default="news augmented & user augmented & engagement augmented")
    parser.add_argument("--optuna_flag", action="store_true") 
    parser.add_argument("-cfg", "--config_name", required=True)  
    # run_mode:
    # +------------------+-------------------------------+----------------------------------------------+
    # | Run Mode         | News Content Features         | User Engagement Features                     |
    # +------------------+-------------------------------+----------------------------------------------+
    # | "plain"          | news article embeddings       | aggregated comments embeddings               |
    # +------------------+-------------------------------+----------------------------------------------+
    # +------------------+-------------------------------+----------------------------------------------+
    # | "news replaced"  | llm-generated news features   | -- (i.e., same as "plain")                   |
    # +------------------+-------------------------------+----------------------------------------------+
    # | "user replaced"  | --                            | llm-generated user profile                   |
    # +------------------+-------------------------------+----------------------------------------------+
    # +------------------+-------------------------------+----------------------------------------------+
    # | "news augmented" | --                            | --                                           |
    # |                  | + llm-generated news features |                                              |
    # +------------------+-------------------------------+----------------------------------------------+
    # | "user augmented" | --                            | --                                           |
    # |                  |                               | + llm-generated user profile                 |
    # +------------------+-------------------------------+----------------------------------------------+
    # | "engagement      | --                            | --                                           |
    # |       augmented" |                               | + llm-generated comments                     |
    # |                  |                               | + llm-generated user engagements             |
    # +------------------+-------------------------------+----------------------------------------------+
    # disentangler
    parser.add_argument("-ptd",
                        "--pre_training_hd", action="store_true")
    parser.add_argument("-ea_mode",
                        "--engagement_augment_mode") 
    parser.add_argument("-c_mode",
                        "--comment_embed_mode") 
    parser.add_argument("-dp_size", "--dglr_proj_size", type=int) 
    parser.add_argument("-cp_size", "--clf_proj_size", type=int) 
    # augmentation
    parser.add_argument("--neighbor_num", type=int)  
    parser.add_argument("-max_au",
                        "--max_augmented_user", type=int)  
    parser.add_argument("-max_u",
                        "--max_related_users_num", type=int)  
    parser.add_argument("-max_c",
                        "--max_comments_num", type=int)  
    # hyper parameters
    parser.add_argument("-proj", "--input_proj_out", type=int) 
    parser.add_argument("-cml", "--comment_max_length", type=int)  
    parser.add_argument("-nml", "--news_max_length", type=int)
    parser.add_argument("-uml", "--user_max_length", type=int)

    parser.add_argument("-w1", "--loss_weight1", type=int) 
    # transformer
    parser.add_argument("-tf_h", "--tf_heads", type=int) 
    parser.add_argument("-tf_l", "--tf_layers", type=int) 
    parser.add_argument("-tf_a", "--tf_hid_amplify", type=int) 
    # MLP
    parser.add_argument("-mlp_c", "--mlp_comment_reduce", type=int) 
    parser.add_argument("-mlp_n", "--mlp_news_reduce", type=int)  
    parser.add_argument("-mlp_u", "--mlp_user_reduce", type=int)  
    parser.add_argument("-out_uf", "--mlp_user_feature_output", type=int)  
    # MLP mode: 
    parser.add_argument("-mlp1_m", "--mlp1_mode")  
    parser.add_argument("-mlp1_d", "--mlp1_dropout", type=float)  
    parser.add_argument("-mlp2_m", "--mlp2_mode") 
    parser.add_argument("-mlp2_d", "--mlp2_dropout", type=float)  
    parser.add_argument("-mlp3_m", "--mlp3_mode") 
    parser.add_argument("-mlp3_d", "--mlp3_dropout", type=float)  
    parser.add_argument("-mlp4_m", "--mlp4_mode")  
    parser.add_argument("-mlp4_d", "--mlp4_dropout", type=float)  
    # attention
    parser.add_argument("-n_attn", "--news_attention_type")  
    parser.add_argument("-u_attn", "--user_attention_type") 
    parser.add_argument("-n_ad", "--news_attn_dropout", type=float)  
    parser.add_argument("-n_od", "--news_out_dropout", type=float)  
    parser.add_argument("-u_ad", "--user_attn_dropout", type=float)  
    parser.add_argument("-u_od", "--user_out_dropout", type=float)  
    parser.add_argument("-n_aln", "--news_attn_layer_norm", action="store_false", default=True)  
    parser.add_argument("-u_aln", "--user_attn_layer_norm", action="store_false", default=True)  

    parser.add_argument("-hc_attn", "--hc_attention_type", default="co") 
    parser.add_argument("-hc_ad", "--hc_attn_dropout", type=float, default=0.0) 
    parser.add_argument("-hc_od", "--hc_out_dropout", type=float, default=0.0) 
    parser.add_argument("-hc_aln", "--hc_attn_layer_norm", action="store_false", default=True) 
    # classifier
    parser.add_argument("-clf_h", "--classifier_hidden", type=int) 
    parser.add_argument("-clf_d", "--classifier_dropout", type=float) 
    # training
    parser.add_argument("-lr", "--lr", type=float) 

    args = parser.parse_args()
    save_path = args.root_path + "Results/domain_aware_agents/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    terminal_output_save_path = save_path + "terminal_output/"
    if not os.path.exists(terminal_output_save_path):
        os.makedirs(terminal_output_save_path)

    log_file = open(f"{terminal_output_save_path}{args.run_name}.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    param_names = [
        "selected_user_num", "get_subset", "is_one_round", "domain_list", "early_stop_delta", "run_mode", "optuna_flag", 
        "pre_training_hd", "engagement_augment_mode", "comment_embed_mode", "dglr_proj_size", "clf_proj_size",
        "neighbor_num", "max_augmented_user", "max_related_users_num", "max_comments_num",
        "input_proj_out", "comment_max_length", "news_max_length", "user_max_length", "loss_weight1",
        "tf_heads", "tf_layers", "tf_hid_amplify",
        "mlp_comment_reduce", "mlp_user_reduce", "mlp_news_reduce", "mlp_user_feature_output",
        "mlp1_mode", "mlp1_dropout", "mlp2_mode", "mlp2_dropout", "mlp3_mode", "mlp3_dropout", "mlp4_mode", "mlp4_dropout",
        "news_attention_type", "user_attention_type", "news_attn_dropout", "news_out_dropout",
        "user_attn_dropout", "user_out_dropout", "news_attn_layer_norm", "user_attn_layer_norm",
        "hc_attention_type", "hc_attn_dropout", "hc_out_dropout", "hc_attn_layer_norm",
        "classifier_hidden", "classifier_dropout",
        "lr"
    ]
    if args.config_name is not None:
        config_path = args.root_path + "Configs/domain_aware_agents/" + args.config_name
        config_dict = read_dict(config_path)
        for pname in param_names:
            if getattr(args, pname) is None:
                setattr(args, pname, config_dict[pname])

    print("Arguments:", vars(args))
    domain_aware_agents4cdfnd(
        # training
        root_path=args.root_path,
        selected_user_num=args.selected_user_num,
        get_subset=args.get_subset,
        is_one_round=args.is_one_round,
        domain_list=args.domain_list,
        early_stop_delta=args.early_stop_delta,
        run_mode=args.run_mode,
        optuna_flag=args.optuna_flag,
        lr=args.lr,
        # disentangler
        pre_training_hd=args.pre_training_hd,
        engagement_augment_mode=args.engagement_augment_mode,
        comment_embed_mode=args.comment_embed_mode,
        dglr_proj_size=args.dglr_proj_size,
        clf_proj_size=args.clf_proj_size,
        # augmentation
        neighbor_num=args.neighbor_num,
        max_augmented_user=args.max_augmented_user,
        max_related_users_num=args.max_related_users_num,
        max_comments_num=args.max_comments_num,
        # hyper parameters
        input_proj_out=args.input_proj_out,
        comment_max_length=args.comment_max_length,
        news_max_length=args.news_max_length,
        user_max_length=args.user_max_length,
        loss_weight1=args.loss_weight1,
        # transformer
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        tf_hid_amplify=args.tf_hid_amplify,
        # MLP
        mlp_comment_reduce=args.mlp_comment_reduce,
        mlp_user_reduce=args.mlp_user_reduce,
        mlp_news_reduce=args.mlp_news_reduce,
        mlp_user_feature_output=args.mlp_user_feature_output,
        # MLP mode
        mlp1_mode=args.mlp1_mode,
        mlp1_dropout=args.mlp1_dropout,
        mlp2_mode=args.mlp2_mode,
        mlp2_dropout=args.mlp2_dropout,
        mlp3_mode=args.mlp3_mode,
        mlp3_dropout=args.mlp3_dropout,
        mlp4_mode=args.mlp4_mode,
        mlp4_dropout=args.mlp4_dropout,
        # attention
        news_attention_type=args.news_attention_type,
        user_attention_type=args.user_attention_type,
        news_attn_dropout=args.news_attn_dropout,
        news_out_dropout=args.news_out_dropout,
        user_attn_dropout=args.user_attn_dropout,
        user_out_dropout=args.user_out_dropout,
        news_attn_layer_norm=args.news_attn_layer_norm,
        user_attn_layer_norm=args.user_attn_layer_norm,
        # (new)
        hc_attention_type=args.hc_attention_type,
        hc_attn_dropout=args.hc_attn_dropout,
        hc_out_dropout=args.hc_out_dropout,
        hc_attn_layer_norm=args.hc_attn_layer_norm,
        # classifier
        classifier_hidden=args.classifier_hidden,
        classifier_dropout=args.classifier_dropout,
    )


if __name__ == '__main__':
    run_domain_aware_agents()

