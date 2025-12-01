import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import random
import torch
from pathlib import Path
import time
import argparse
import datetime
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from layers import Disentangler, Classifier, MLP
from slm import init_roberta
from toolkits import read_dict, dataset_split, news_dataloader, get_text_data_batch
from toolkits import compute_metrics_print, compute_macro_f1, read_embedding_to_batch_npz
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup  

warnings.filterwarnings("ignore")
random.seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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



def load_data(root_path="D:/WorkSpace/workspace/LLMFND/"):
    veracity_dict_1_fake_path = root_path + "Results/Prelim/veracity_dict_1_fake"
    filtered_news_dict_path = root_path + "Results/Prelim/news_dict"

    news_dict = read_dict(filtered_news_dict_path)
    veracity_dict_1_fake = read_dict(veracity_dict_1_fake_path)

    news_id_list = list(news_dict.keys())
    politifact_idx_dict, gossipcop_idx_dict = {}, {} 
    for i, news_id in enumerate(news_id_list):
        veracity_label = veracity_dict_1_fake[news_id]
        if "politifact" in news_id:
            politifact_idx_dict[i] = veracity_label
        elif "gossipcop" in news_id:
            gossipcop_idx_dict[i] = veracity_label

    news_idx_content_dict = {} 
    for i, news_id in enumerate(news_id_list):
        news_idx_content_dict[str(i)] = news_dict[news_id]

    return politifact_idx_dict, gossipcop_idx_dict, news_idx_content_dict


def encode_content(lm_list, content_list, max_length=256, not_training=True):
    tokenizer, lm_mod = lm_list
    content_embedding_list, attention_mask_list = [], []

    if not_training:
        with torch.no_grad():
            for content in content_list:
                content = content.replace("\n", " ")
                encoded_dict = tokenizer.encode_plus(content,
                                                     add_special_tokens=True,
                                                     max_length=max_length,
                                                     truncation=True,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     return_tensors='pt',
                                                     )
                input_ids = encoded_dict['input_ids'].to(DEVICE)
                attention_mask = encoded_dict['attention_mask'].to(DEVICE)
                content_embedding = lm_mod(input_ids, attention_mask=attention_mask)[0]
                content_embedding_list.append(content_embedding)  # (batch_size, hidden_dim)
                attention_mask_list.append(attention_mask)  # (batch_size, max_length)
    else:
        for content in content_list:
            content = content.replace("\n", " ")
            encoded_dict = tokenizer.encode_plus(content,
                                                 add_special_tokens=True,
                                                 max_length=max_length,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt',
                                                 )
            input_ids = encoded_dict['input_ids'].to(DEVICE)
            attention_mask = encoded_dict['attention_mask'].to(DEVICE)
            content_embedding = lm_mod(input_ids, attention_mask=attention_mask)[0]
            content_embedding_list.append(content_embedding)
            attention_mask_list.append(attention_mask)  # (batch_size, max_length)
    content_embedding = torch.cat(content_embedding_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)

    return content_embedding, attention_mask


class HierarchicalDisentangler(nn.Module):
    def __init__(self, converted_shape2, mask1_hidden_dim, mask2_hidden_dim, veracity_hidden_dim, domain_hidden_dim,
                 n_classes, encode_content_max_length, encode_content_max_length2=None):
        super(HierarchicalDisentangler, self).__init__()
        self.disentangler = Disentangler(converted_shape2, mask1_hidden_dim, mask2_hidden_dim).to(DEVICE)
        self.veracity_classifier = Classifier(converted_shape2, veracity_hidden_dim, n_classes).to(DEVICE)
        self.veracity_discriminator = Classifier(converted_shape2, veracity_hidden_dim, n_classes).to(DEVICE)
        self.domain_classifier = Classifier(converted_shape2, domain_hidden_dim, n_classes).to(DEVICE)
        self.domain_discriminator = Classifier(converted_shape2, domain_hidden_dim, n_classes).to(DEVICE)

        self.vr_proj = nn.Linear(encode_content_max_length, 1).to(DEVICE)
        self.vi_proj = nn.Linear(encode_content_max_length, 1).to(DEVICE)
        self.dsh_proj = nn.Linear(encode_content_max_length, 1).to(DEVICE)
        self.dsf_proj = nn.Linear(encode_content_max_length, 1).to(DEVICE)

        if encode_content_max_length2 is not None:
            self.vr_proj2 = nn.Linear(encode_content_max_length2, 1).to(DEVICE)
            self.vi_proj2 = nn.Linear(encode_content_max_length2, 1).to(DEVICE)
            self.dsh_proj2 = nn.Linear(encode_content_max_length2, 1).to(DEVICE)
            self.dsf_proj2 = nn.Linear(encode_content_max_length2, 1).to(DEVICE)

    def forward(self, content_embedding, sentence_num_padding_mask=None):
        vr_feature, vi_feature, dsh_feature, dsf_feature = self.disentangler(content_embedding,
                                                                             sentence_num_padding_mask)
        if content_embedding.ndim == 3:
            vr_feature_p = self.vr_proj(vr_feature.permute(0, 2, 1)).squeeze(-1)
            vi_feature_p = self.vr_proj(vi_feature.permute(0, 2, 1)).squeeze(-1)
            dsh_feature_p = self.vr_proj(dsh_feature.permute(0, 2, 1)).squeeze(-1)
            dsf_feature_p = self.vr_proj(dsf_feature.permute(0, 2, 1)).squeeze(-1)
            vr_logits = self.veracity_classifier(vr_feature_p)
            vi_logits = self.veracity_discriminator(vi_feature_p)
            dsh_logits = self.domain_discriminator(dsh_feature_p)
            dsf_logits = self.domain_classifier(dsf_feature_p)
            return vr_logits, vi_logits, dsh_logits, dsf_logits, vr_feature, vi_feature, dsh_feature, dsf_feature

        elif content_embedding.ndim == 4:
            try:
                vr_feature_p = self.vr_proj(vr_feature.permute(0, 1, 3, 2)).squeeze(-1)
                vi_feature_p = self.vr_proj(vi_feature.permute(0, 1, 3, 2)).squeeze(-1)
                dsh_feature_p = self.vr_proj(dsh_feature.permute(0, 1, 3, 2)).squeeze(-1)
                dsf_feature_p = self.vr_proj(dsf_feature.permute(0, 1, 3, 2)).squeeze(-1)
            except:
                vr_feature_p = self.vr_proj2(vr_feature.permute(0, 1, 3, 2)).squeeze(-1)
                vi_feature_p = self.vr_proj2(vi_feature.permute(0, 1, 3, 2)).squeeze(-1)
                dsh_feature_p = self.vr_proj2(dsh_feature.permute(0, 1, 3, 2)).squeeze(-1)
                dsf_feature_p = self.vr_proj2(dsf_feature.permute(0, 1, 3, 2)).squeeze(-1)
            vr_logits = self.veracity_classifier(vr_feature_p)
            vi_logits = self.veracity_discriminator(vi_feature_p)
            dsh_logits = self.domain_discriminator(dsh_feature_p)
            dsf_logits = self.domain_classifier(dsf_feature_p)
            return vr_logits, vi_logits, dsh_logits, dsf_logits, vr_feature, vi_feature, dsh_feature, dsf_feature

        else:
            raise ValueError("Unsupported content_embedding shape: {}".format(content_embedding.shape))


def run(mod, lm_list, news_batch, news_content_batch, veracity_batch, domain_batch, model_type="train",
        root_path="D:/WorkSpace/workspace/LLMFND/"):
    selected_user_num = 10000
    file_path = f"{root_path}Results/llm_agents/"
    news_embedding_path = file_path + f"news/{selected_user_num}/"

    news_input_proj, hd = mod
    for m in mod:
        m.zero_grad()

    try:
        news_embedding_raw = read_embedding_to_batch_npz(news_batch, news_embedding_path + "embedding/").to(DEVICE)
        news_mask_raw = read_embedding_to_batch_npz(news_batch, news_embedding_path + "mask/").squeeze(1).to(DEVICE)
    except FileNotFoundError:
        news_embedding_raw, news_mask_raw = encode_content(lm_list, news_content_batch)

    news_embedding = news_input_proj(news_embedding_raw)
    news_mask = news_mask_raw.unsqueeze(-1)  # (batch_size, 1, max_length)

    (vr_logits, vi_logits, dsh_logits, dsf_logits,
     vr_feature, vi_feature, dsh_feature, dsf_feature) = hd(news_embedding, news_mask)

    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss(reduction='none')
    # [1]: Veracity Loss
    vr_loss = classification_criterion(vr_logits, veracity_batch)
    ad_vi_loss = torch.clamp(1 - classification_criterion(vi_logits, veracity_batch), min=0)
    v_loss = vr_loss + ad_vi_loss
    # [2]: Domain Loss
    dsf_loss = classification_criterion(dsf_logits, domain_batch)
    ad_dsh_loss = torch.clamp(1 - classification_criterion(dsh_logits, domain_batch), min=0)
    d_loss = dsf_loss + ad_dsh_loss
    # [3]: Reconstruction Loss of input content_embedding
    content_reconstruct = vr_feature + vi_feature
    whole_r1_loss = reconstruction_criterion(content_reconstruct, news_embedding)
    per_sample_r1_loss = whole_r1_loss.mean(dim=(1, 2))
    r1_loss = per_sample_r1_loss.mean(dim=0)
    # [4]: Reconstruction Loss of veracity-relevant features
    vr_reconstruct = dsh_feature + dsf_feature
    whole_r2_loss = reconstruction_criterion(vr_reconstruct, vr_feature)
    per_sample_r2_loss = whole_r2_loss.mean(dim=(1, 2))
    r2_loss = per_sample_r2_loss.mean(dim=0)
    # [5]: Veracity Prediction Loss: 
    dsh_v_logits = hd.veracity_classifier(hd.dsh_proj(dsh_feature.permute(0, 2, 1)).squeeze(-1))
    vp_loss = classification_criterion(dsh_v_logits, veracity_batch)

    if model_type == "train":
        return [v_loss, d_loss, r1_loss, r2_loss, vp_loss]
    elif model_type == "test":
        v_classifier = hd.veracity_classifier
        dsh_proj, dsf_proj = hd.dsh_proj, hd.dsf_proj
        dsh_feature_p = dsh_proj(dsh_feature.permute(0, 2, 1)).squeeze(-1)
        dsf_feature_p = dsf_proj(dsf_feature.permute(0, 2, 1)).squeeze(-1)
        vr_reconstruct_p = dsh_feature_p + dsf_feature_p

        dsh_v_logits = v_classifier(dsh_feature_p)
        dsf_v_logits = v_classifier(dsf_feature_p)
        vrre_logits = v_classifier(vr_reconstruct_p)
        return dsh_v_logits, dsf_v_logits, vrre_logits, [v_loss, d_loss, r1_loss, r2_loss, vp_loss]


def disentangler4cdfnd():
    politifact_idx_dict, gossipcop_idx_dict, news_idx_content_dict = load_data()
    print("Politifact News Count: {}, Gossipcop News Count: {}".format(len(politifact_idx_dict),
                                                                       len(gossipcop_idx_dict)))

    output_path = "../Results/disentangler/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    domain_list = ["gossipcop", "politifact"]
    is_one_round = True
    for i in range(len(domain_list)):
        train_news_idx_dict, test_news_idx_dict, domain_labels, domain_test_labels = {}, {}, [], []
        if len(domain_list) > 1:
            train_domain, test_domain = domain_list[i], domain_list[1 - i]
            if train_domain == "politifact":
                train_news_idx_dict = politifact_idx_dict
                test_news_idx_dict = gossipcop_idx_dict
                domain_labels = [0] * len(train_news_idx_dict)
                domain_test_labels = [1] * len(test_news_idx_dict)
            elif train_domain == "gossipcop":
                train_news_idx_dict = gossipcop_idx_dict
                test_news_idx_dict = politifact_idx_dict
                domain_labels = [1] * len(train_news_idx_dict)
                domain_test_labels = [0] * len(test_news_idx_dict)
        else:
            train_domain, test_domain = domain_list[i], domain_list[i]
            test_size = 0.2
            if train_domain == "politifact":
                news_idxs = list(politifact_idx_dict.keys())
                veracity_labels = list(politifact_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs, veracity_labels,
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

        info = "Train on {}, Test on {}".format(train_domain, test_domain)
        print(info)
        print("\nTrain News Count: {}, Test News Count: {}".format(len(train_news_idx_dict),
                                                                 len(test_news_idx_dict)))

        start_running_time = datetime.datetime.now()
        start_running_timestamp = start_running_time.strftime("%Y%m%d_%H%M%S")

        epochs, lr = 0, 0
        if train_domain == "politifact":
            epochs = 100
            lr = 5e-4
        elif train_domain == "gossipcop":
            epochs = 20
            lr = 1e-4
        batch_size, eps = 32, 1e-8

        x_set, y_set = list(train_news_idx_dict.keys()), list(train_news_idx_dict.values())
        x_test_set, y_test_set = list(test_news_idx_dict.keys()), list(test_news_idx_dict.values())

        train_loader = news_dataloader(x_set, y_set, domain_labels, batch_size)
        test_loader = news_dataloader(x_test_set, y_test_set, domain_test_labels, batch_size)

        # task 1: hierarchical disentangler for unseen news domain
        converted_shape2, dglr_v_proj, dglr_d_proj = 768, 512, 512  
        clf_v_proj, clf_d_proj, n_classes = 512, 512, 2
        alpha1, alpha2, alpha3, alpha4, alpha5 = 1, 1, 1, 17, 170

        model = HierarchicalDisentangler(converted_shape2, dglr_v_proj, dglr_d_proj,
                                         clf_v_proj, clf_d_proj, n_classes)
        lm = init_roberta()
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, eps=eps)
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)
        training_stats, test_metrics = [], []
        for epoch in range(0, epochs):
            # ========================================
            #               Training
            # ========================================
            print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

            total_train_loss, total_veracity_loss, total_domain_loss, total_r1_loss, total_r2_loss = 0, 0, 0, 0, 0
            model.train()
            train_bar = tqdm(enumerate(train_loader))
            for step, batch in train_bar:
                news_batch = batch[0].to(DEVICE)
                veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                veracity_loss, domain_loss, r1_loss, r2_loss, prediction_loss = run(model, lm, news_batch, content_list,
                                                                                    veracity_label, domain_label)
                loss = (alpha1 * prediction_loss +
                        alpha2 * veracity_loss + alpha3 * domain_loss + alpha4 * r1_loss + alpha5 * r2_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss = total_train_loss + loss.detach().cpu()
                batch_train_losses = {
                    'train_loss': loss.detach().cpu(),
                    'veracity_loss': veracity_loss.detach().cpu(),
                    'domain_loss': domain_loss.detach().cpu(),
                    'v_reconstruct_loss': r1_loss.detach().cpu(),
                    'd_reconstruct_loss': r2_loss.detach().cpu()
                }
                optimizer.zero_grad()
            train_bar.close()
            avg_train_loss = total_train_loss / len(train_loader)
            print("\n  Average training loss: {}".format(avg_train_loss))

            if (epoch + 1) % 5 == 0 or (epoch + 1) >= epochs * 0.8:
                # ========================================
                #               Testing
                # ========================================
                test_bar = tqdm(enumerate(test_loader))

                test_dsh_v_logits, test_dsf_v_logits, test_vrre_logits, test_veracity_label = [], [], [], []
                test_loss_list = []
                for step, batch in test_bar:
                    news_batch = batch[0].to(DEVICE)
                    veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                    domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                    content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                    with torch.no_grad():
                        dsh_v_logits, dsf_v_logits, vrre_logits, losses = run(
                            model, lm, news_batch, content_list, veracity_label, domain_label, model_type="test")
                        veracity_loss, domain_loss, r1_loss, r2_loss, prediction_loss = losses
                        test_loss = (alpha1 * prediction_loss +
                                     alpha2 * veracity_loss + alpha3 * domain_loss + alpha4 * r1_loss + alpha5 * r2_loss)

                    test_dsh_v_logits.append(dsh_v_logits)
                    test_dsf_v_logits.append(dsf_v_logits)
                    test_vrre_logits.append(vrre_logits)

                    test_veracity_label.append(veracity_label)
                    test_loss_list.append(test_loss.cpu().detach().numpy())

                test_dsh_v_logits = torch.cat(test_dsh_v_logits, dim=0).cpu()
                test_dsf_v_logits = torch.cat(test_dsf_v_logits, dim=0).cpu()
                test_vrre_logits = torch.cat(test_vrre_logits, dim=0).cpu()
                test_veracity_label = torch.cat(test_veracity_label, dim=0).cpu()

                compute_metrics_print(test_dsh_v_logits, test_veracity_label,
                                      "Domain-Shared Features on {}".format(test_domain))
                compute_metrics_print(test_dsf_v_logits, test_veracity_label,
                                      "Domain-Specific Features on {}".format(test_domain))
                compute_metrics_print(test_vrre_logits, test_veracity_label,
                                      "Reconstructed VR Features on {}".format(test_domain))
                avg_test_loss = float(np.mean(test_loss_list))

        end_running_time = datetime.datetime.now()
        end_running_timestamp = end_running_time.strftime("%Y%m%d_%H%M%S")
        running_time = end_running_time - start_running_time
        print("[ {} -> {} ] Start Time: {}, End Time: {}, Running Time: {}".format(train_domain, test_domain,
                                                                                   start_running_timestamp,
                                                                                   end_running_timestamp, running_time))
        if is_one_round:
            break


def no_disentangler4cdfnd():
    # [Goal]: Only RoBERTa For comparison
    politifact_idx_dict, gossipcop_idx_dict, news_idx_content_dict = load_data()
    print("Politifact News Count: {}, Gossipcop News Count: {}".format(len(politifact_idx_dict),
                                                                       len(gossipcop_idx_dict)))
    output_path = "../Results/disentangler/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    domain_list = ["gossipcop", "politifact"] 
    is_one_round = True
    for i in range(len(domain_list)):
        train_news_idx_dict, test_news_idx_dict, domain_labels, domain_test_labels = {}, {}, [], []
        if len(domain_list) > 1:
            train_domain, test_domain = domain_list[i], domain_list[1 - i]
            if train_domain == "politifact":
                train_news_idx_dict = politifact_idx_dict
                test_news_idx_dict = gossipcop_idx_dict
                domain_labels = [0] * len(train_news_idx_dict)
                domain_test_labels = [1] * len(test_news_idx_dict)
            elif train_domain == "gossipcop":
                train_news_idx_dict = gossipcop_idx_dict
                test_news_idx_dict = politifact_idx_dict
                domain_labels = [1] * len(train_news_idx_dict)
                domain_test_labels = [0] * len(test_news_idx_dict)
        else:
            train_domain, test_domain = domain_list[i], domain_list[i]
            test_size = 0.2
            if train_domain == "politifact":
                news_idxs = list(politifact_idx_dict.keys())
                veracity_labels = list(politifact_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs, veracity_labels,
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

        info = "Train on {}, Test on {}".format(train_domain, test_domain)
        print(info)
        print("\nTrain News Count: {}, Test News Count: {}".format(len(train_news_idx_dict),
                                                                 len(test_news_idx_dict)))

        start_running_time = datetime.datetime.now()
        start_running_timestamp = start_running_time.strftime("%Y%m%d_%H%M%S")

        epochs, lr = 0, 0
        if train_domain == "politifact":
            epochs = 100
            lr = 5e-4
        elif train_domain == "gossipcop":
            epochs = 20
            lr = 1e-4
        batch_size, eps = 32, 1e-8

        x_set, y_set = list(train_news_idx_dict.keys()), list(train_news_idx_dict.values())
        x_test_set, y_test_set = list(test_news_idx_dict.keys()), list(test_news_idx_dict.values())
        train_loader = news_dataloader(x_set, y_set, domain_labels, batch_size)
        test_loader = news_dataloader(x_test_set, y_test_set, domain_test_labels, batch_size)

        veracity_classifier = Classifier(768, 512, 2).to(DEVICE)
        lm = init_roberta()

        params = [param for param in veracity_classifier.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, eps=eps)
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)
        for epoch in range(0, epochs):
            # ========================================
            #               Training
            # ========================================
            print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

            total_train_loss = 0
            veracity_classifier.train()
            train_bar = tqdm(enumerate(train_loader))
            for step, batch in train_bar:
                news_batch = batch[0].to(DEVICE)
                veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                content_embedding = encode_content(lm, content_list)
                logits = veracity_classifier(content_embedding)
                classification_criterion = nn.CrossEntropyLoss()
                loss = classification_criterion(logits, veracity_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(veracity_classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss = total_train_loss + loss.detach().cpu()
                optimizer.zero_grad()
            train_bar.close()
            avg_train_loss = total_train_loss / len(train_loader)
            print("\n  Average training loss: {}".format(avg_train_loss))

            if (epoch + 1) % 5 == 0 or (epoch + 1) >= epochs * 0.8:
                # ========================================
                #               Testing
                # ========================================
                test_bar = tqdm(enumerate(test_loader))

                test_logits, test_veracity_label = [], []
                test_loss_list = []
                for step, batch in test_bar:
                    news_batch = batch[0].to(DEVICE)
                    veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                    content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                    with torch.no_grad():
                        content_embedding = encode_content(lm, content_list)
                        logits = veracity_classifier(content_embedding)
                        classification_criterion = nn.CrossEntropyLoss()
                        loss = classification_criterion(logits, veracity_label)

                    test_logits.append(logits)
                    test_veracity_label.append(veracity_label)
                    test_loss_list.append(loss.cpu().detach().numpy())

                test_logits = torch.cat(test_logits, dim=0).cpu()
                test_veracity_label = torch.cat(test_veracity_label, dim=0).cpu()

                compute_metrics_print(test_logits, test_veracity_label,
                                      "Test on {}".format(test_domain))

        end_running_time = datetime.datetime.now()
        end_running_timestamp = end_running_time.strftime("%Y%m%d_%H%M%S")
        running_time = end_running_time - start_running_time
        print("[ {} -> {} ] Start Time: {}, End Time: {}, Running Time: {}".format(train_domain, test_domain,
                                                                                   start_running_timestamp,
                                                                                   end_running_timestamp, running_time))
        if is_one_round:
            break


def pre_train_disentangler_save(run_name, root_path, epochs, learning_rate, batch_size, adam_epsilon,
                                input_proj_out, dglr_proj_size, clf_proj_size, optuna_flag):
    # [Goal]: Pre-Train Disentangler, and Save the Best Model
    politifact_idx_dict_origin, gossipcop_idx_dict_origin, news_idx_content_dict = load_data(root_path=root_path)
    print("Politifact News Count: {}, Gossipcop News Count: {}".format(len(politifact_idx_dict_origin),
                                                                       len(gossipcop_idx_dict_origin)))
    test_split_ratio = 0.2
    p_news_idxs, p_veracity_labels = list(politifact_idx_dict_origin.keys()), list(politifact_idx_dict_origin.values())
    p_train_idxs, _, p_train_labels, _ = train_test_split(p_news_idxs, p_veracity_labels,
                                                          test_size=test_split_ratio, random_state=42,
                                                          shuffle=True, stratify=p_veracity_labels)
    politifact_idx_dict = {idx: label for idx, label in zip(p_train_idxs, p_train_labels)}
    g_news_idxs, g_veracity_labels = list(gossipcop_idx_dict_origin.keys()), list(gossipcop_idx_dict_origin.values())
    g_train_idxs, _, g_train_labels, _ = train_test_split(g_news_idxs, g_veracity_labels,
                                                          test_size=test_split_ratio, random_state=42,
                                                          shuffle=True, stratify=g_veracity_labels)
    gossipcop_idx_dict = {idx: label for idx, label in zip(g_train_idxs, g_train_labels)}

    output_path = root_path + "Results/disentangler/"
    best_model_path = output_path + f"saved_{run_name}/best_disentangler_3d.pt"
    folder_path = os.path.dirname(best_model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    best_f1_p, best_f1_g = 0.0, 0.0
    test_size = 0.2
    domain_list = ["gossipcop", "politifact"]  
    is_one_round = False
    is_mixed_train = True
    best_metric = 0.0
    for i in range(len(domain_list)):
        train_news_idx_dict, test_news_idx_dict, domain_labels, domain_test_labels = {}, {}, [], []
        if len(domain_list) > 1:
            if is_mixed_train:
                train_domain, test_domain = "mixed_domain", "mixed_domain"
                p_news_idxs = list(politifact_idx_dict.keys())
                p_veracity_labels = list(politifact_idx_dict.values())
                p_train_idxs, p_test_idxs, p_train_labels, p_test_labels = train_test_split(p_news_idxs,
                                                                                            p_veracity_labels,
                                                                                            test_size=test_size,
                                                                                            random_state=42,
                                                                                            shuffle=True,
                                                                                            stratify=p_veracity_labels)
                p_train_news_idx_dict = {idx: label for idx, label in zip(p_train_idxs, p_train_labels)}
                p_test_news_idx_dict = {idx: label for idx, label in zip(p_test_idxs, p_test_labels)}
                p_domain_labels = [0] * len(p_train_news_idx_dict)
                p_domain_test_labels = [0] * len(p_test_news_idx_dict)

                g_news_idxs = list(gossipcop_idx_dict.keys())
                g_veracity_labels = list(gossipcop_idx_dict.values())
                g_train_idxs, g_test_idxs, g_train_labels, g_test_labels = train_test_split(g_news_idxs,
                                                                                            g_veracity_labels,
                                                                                            test_size=test_size,
                                                                                            random_state=42,
                                                                                            shuffle=True,
                                                                                            stratify=g_veracity_labels)
                g_train_news_idx_dict = {idx: label for idx, label in zip(g_train_idxs, g_train_labels)}
                g_test_news_idx_dict = {idx: label for idx, label in zip(g_test_idxs, g_test_labels)}
                g_domain_labels = [1] * len(g_train_news_idx_dict)
                g_domain_test_labels = [1] * len(g_test_news_idx_dict)

                train_news_idx_dict = {**p_train_news_idx_dict, **g_train_news_idx_dict}
                test_news_idx_dict = {**p_test_news_idx_dict, **g_test_news_idx_dict}
                domain_labels = p_domain_labels + g_domain_labels
                domain_test_labels = p_domain_test_labels + g_domain_test_labels
                split_length = len(p_train_news_idx_dict)
                is_one_round = True
            else:
                train_domain, test_domain = domain_list[i], domain_list[1 - i]
                if train_domain == "politifact":
                    train_news_idx_dict = politifact_idx_dict
                    test_news_idx_dict = gossipcop_idx_dict
                    domain_labels = [0] * len(train_news_idx_dict)
                    domain_test_labels = [1] * len(test_news_idx_dict)
                elif train_domain == "gossipcop":
                    train_news_idx_dict = gossipcop_idx_dict
                    test_news_idx_dict = politifact_idx_dict
                    domain_labels = [1] * len(train_news_idx_dict)
                    domain_test_labels = [0] * len(test_news_idx_dict)
        else:
            train_domain, test_domain = domain_list[i], domain_list[i]
            if train_domain == "politifact":
                news_idxs = list(politifact_idx_dict.keys())
                veracity_labels = list(politifact_idx_dict.values())
                train_idxs, test_idxs, train_labels, test_labels = train_test_split(news_idxs, veracity_labels,
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

        info = "Train on {}, Test on {}".format(train_domain, test_domain)
        print(info)
        print("\nTrain News Count: {}, Test News Count: {}".format(len(train_news_idx_dict),
                                                                   len(test_news_idx_dict)))

        start_running_time = datetime.datetime.now()
        start_running_timestamp = start_running_time.strftime("%Y%m%d_%H%M%S")

        x_set, y_set = list(train_news_idx_dict.keys()), list(train_news_idx_dict.values())
        x_test_set, y_test_set = list(test_news_idx_dict.keys()), list(test_news_idx_dict.values())

        train_loader = news_dataloader(x_set, y_set, domain_labels, batch_size)
        test_loader = news_dataloader(x_test_set, y_test_set, domain_test_labels, batch_size)

        # task 1: hierarchical disentangler for unseen news domain
        converted_shape2, dglr_v_proj, dglr_d_proj = 768, dglr_proj_size, dglr_proj_size
        clf_v_proj, clf_d_proj, n_classes = clf_proj_size, clf_proj_size, 2
        alpha1, alpha2, alpha3, alpha4, alpha5 = 1, 1, 1, 1, 10
        encode_content_max_length = 256

        news_input_proj = MLP(converted_shape2, input_proj_out).to(DEVICE)
        hd = HierarchicalDisentangler(input_proj_out, dglr_v_proj, dglr_d_proj,
                                      clf_v_proj, clf_d_proj, n_classes,
                                      encode_content_max_length)
        models = [news_input_proj, hd]
        lm = init_roberta(root_path=str(Path(root_path).parent))
        params = []
        for m in models:
            for param in m.parameters():
                if param.requires_grad:
                    params.append(param)
                    assert not torch.isnan(param).any(), "NaN detected in model parameters!"
        optimizer = torch.optim.AdamW(params, lr=learning_rate, eps=adam_epsilon)
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)
        for epoch in range(0, epochs):
            # ========================================
            #               Training
            # ========================================
            print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

            total_train_loss, total_veracity_loss, total_domain_loss, total_r1_loss, total_r2_loss = 0, 0, 0, 0, 0
            for m in models:
                m.train()
            train_bar = tqdm(enumerate(train_loader))
            for step, batch in train_bar:
                news_batch = batch[0].to(DEVICE)
                veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                veracity_loss, domain_loss, r1_loss, r2_loss, prediction_loss = run(models, lm, news_batch,
                                                                                    content_list, veracity_label,
                                                                                    domain_label, root_path=root_path)
                loss = (alpha1 * prediction_loss +
                        alpha2 * veracity_loss + alpha3 * domain_loss + alpha4 * r1_loss + alpha5 * r2_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss = total_train_loss + loss.detach().cpu()
                batch_train_losses = {
                    'train_loss': loss.detach().cpu(),
                    'veracity_loss': veracity_loss.detach().cpu(),
                    'domain_loss': domain_loss.detach().cpu(),
                    'v_reconstruct_loss': r1_loss.detach().cpu(),
                    'd_reconstruct_loss': r2_loss.detach().cpu()
                }
                optimizer.zero_grad()
            train_bar.close()
            avg_train_loss = total_train_loss / len(train_loader)
            print("\n  Average training loss: {}".format(avg_train_loss))

            if (epoch + 1) >= epochs * 0.2:
                # ========================================
                #               Testing
                # ========================================
                test_bar = tqdm(enumerate(test_loader))

                test_dsh_v_logits, test_dsf_v_logits, test_vrre_logits, test_veracity_label = [], [], [], []
                test_loss_list = []
                for step, batch in test_bar:
                    news_batch = batch[0].to(DEVICE)
                    veracity_label = batch[1].type(torch.LongTensor).to(DEVICE)
                    domain_label = batch[2].type(torch.LongTensor).to(DEVICE)
                    content_list = get_text_data_batch(news_batch, news_idx_content_dict)

                    with torch.no_grad():
                        dsh_v_logits, dsf_v_logits, vrre_logits, losses = run(models, lm, news_batch, content_list,
                                                                              veracity_label, domain_label,
                                                                              model_type="test", root_path=root_path)
                        veracity_loss, domain_loss, r1_loss, r2_loss, prediction_loss = losses
                        test_loss = (alpha1 * prediction_loss +
                                     alpha2 * veracity_loss + alpha3 * domain_loss +
                                     alpha4 * r1_loss + alpha5 * r2_loss)

                    test_dsh_v_logits.append(dsh_v_logits)
                    test_dsf_v_logits.append(dsf_v_logits)
                    test_vrre_logits.append(vrre_logits)

                    test_veracity_label.append(veracity_label)
                    test_loss_list.append(test_loss.cpu().detach().numpy())

                test_dsh_v_logits = torch.cat(test_dsh_v_logits, dim=0).cpu()
                test_dsf_v_logits = torch.cat(test_dsf_v_logits, dim=0).cpu()
                test_vrre_logits = torch.cat(test_vrre_logits, dim=0).cpu()
                test_veracity_label = torch.cat(test_veracity_label, dim=0).cpu()

                if is_mixed_train:  
                    compute_metrics_print(test_dsh_v_logits[:split_length], test_veracity_label[:split_length],
                                          "Domain-Shared Features on Politifact")
                    compute_metrics_print(test_dsf_v_logits[:split_length], test_veracity_label[:split_length],
                                          "Domain-Specific Features on Politifact")
                    _, _, _, _, test_macro_f1_p, _ = compute_metrics_print(test_vrre_logits[:split_length],
                                                                           test_veracity_label[:split_length],
                                                                           "Reconstructed VR Features on Politifact",
                                                                           has_return=True)

                    compute_metrics_print(test_dsh_v_logits[split_length:], test_veracity_label[split_length:],
                                          "Domain-Shared Features on Gossipcop")
                    compute_metrics_print(test_dsf_v_logits[split_length:], test_veracity_label[split_length:],
                                          "Domain-Specific Features on Gossipcop")
                    _, _, _, _, test_macro_f1_g, _ = compute_metrics_print(test_vrre_logits[split_length:],
                                                                           test_veracity_label[split_length:],
                                                                           "Reconstructed VR Features on Gossipcop",
                                                                           has_return=True)

                    best_f1_g = save_best_model(best_model_path, hd, optimizer, epoch,
                                                best_f1_g, test_macro_f1_p + test_macro_f1_g, "P&G")

                else:
                    compute_metrics_print(test_dsh_v_logits, test_veracity_label,
                                          f"Domain-Shared Features on {test_domain}")
                    compute_metrics_print(test_dsf_v_logits, test_veracity_label,
                                          f"Domain-Specific Features on {test_domain}")
                    _, _, _, _, test_macro_f1, _ = compute_metrics_print(test_vrre_logits, test_veracity_label,
                                                                        f"Reconstructed VR Features on {test_domain}",
                                                                         has_return=True)
                    if test_domain == "politifact":
                        best_f1_p = save_best_model(best_model_path, hd, optimizer, epoch,
                                                    best_f1_p, test_macro_f1, test_domain)
                    elif test_domain == "gossipcop":
                        best_f1_g = save_best_model(best_model_path, hd, optimizer, epoch,
                                                    best_f1_g, test_macro_f1, test_domain)
                    else:
                        raise ValueError(f"Unknown test domain: {test_domain}")

        end_running_time = datetime.datetime.now()
        end_running_timestamp = end_running_time.strftime("%Y%m%d_%H%M%S")
        running_time = end_running_time - start_running_time
        print("[ {} -> {} ] Start Time: {}, End Time: {}, Running Time: {}".format(train_domain, test_domain,
                                                                                   start_running_timestamp,
                                                                                   end_running_timestamp, running_time))
        if is_mixed_train:
            best_metric = best_f1_p + best_f1_g
        else:
            best_metric = best_f1_p if train_domain == "politifact" else best_f1_g
        if is_one_round:
            break
    if optuna_flag:
        return best_metric


def save_best_model(best_model_path, hd, optimizer, epoch, best_f1, current_f1, tag):
    # Save the best hd
    if current_f1 > best_f1:
        best_f1 = current_f1
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            saved_f1 = checkpoint.get('macro_f1', 0.0)
            if best_f1 > saved_f1:
                print(f"\n New best HD found on {tag}! Old F1: {saved_f1:.4f}, New F1: {best_f1:.4f}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': hd.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'macro_f1': best_f1,
                }, best_model_path)
            else:
                print(
                    f"\n Current F1 {best_f1:.4f} not better than saved F1 {saved_f1:.4f} on {tag}. HD not saved."
                )
        else:
            print(f"\n Saving first best HD on {tag} with Macro-F1: {best_f1:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': hd.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1': best_f1,
            }, best_model_path)
    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("-rn", "--run_name", default="debug")
    parser.add_argument("--root_path", default="D:/WorkSpace/workspace/LLMFND/")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5) 
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-eps", "--adam_epsilon", type=float, default=1e-8) 
    # disentangler
    parser.add_argument("-dp_size", "--dglr_proj_size", type=int, default=32)  
    parser.add_argument("-cp_size", "--clf_proj_size", type=int, default=64)  
    # hyper parameters
    parser.add_argument("-proj", "--input_proj_out", type=int, default=32)  
    parser.add_argument("--optuna_flag", action="store_true")
    args = parser.parse_args()

    save_path = args.root_path + "Results/disentangler/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    terminal_output_save_path = save_path + "terminal_output/"
    if not os.path.exists(terminal_output_save_path):
        os.makedirs(terminal_output_save_path)
    log_file = open(f"{terminal_output_save_path}{args.run_name}.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print("Arguments:", vars(args))

    pre_train_disentangler_save(
        run_name=args.run_name,
        root_path=args.root_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        adam_epsilon=args.adam_epsilon,
        input_proj_out=args.input_proj_out,
        dglr_proj_size=args.dglr_proj_size,
        clf_proj_size=args.clf_proj_size,
        optuna_flag=args.optuna_flag
    )

