import torch
import torch.nn as nn
from layers import MLP
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
import warnings
import random
from toolkits import (compute_auc, compute_accuracy, compute_macro_f1, compute_micro_f1,
                      compute_precision, compute_recall)

warnings.filterwarnings("ignore")
random.seed(2025)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
huggingface_cache_dir = "D:/WorkSpace/huggingface_cache/"


# ==================== LMs Initialization ====================
def init_roberta(root_path="D:/WorkSpace/"):
    local_model_path = root_path + "/huggingface_cache/roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(local_model_path, do_lower_case=True)
    roberta = RobertaModel.from_pretrained(local_model_path)
    roberta.to(DEVICE)
    return tokenizer, roberta


def init_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=huggingface_cache_dir)
    gpt2 = GPT2Model.from_pretrained('gpt2', cache_dir=huggingface_cache_dir)
    gpt2.to(DEVICE)
    return tokenizer, gpt2


# ==================== For FND ====================
def roberta_fnd(news_batch):
    tokenizer, model = init_roberta()
    pca = PCA(n_components=2)

    with torch.no_grad():
        model.eval()

        content_embedding_list = []
        for news_content in news_batch:
            news_content = news_content.replace("\n", " ")
            words_list = news_content.split(" ")
            trancated_words_list = words_list[:256]
            trancated_news_content = " ".join(trancated_words_list)

            encoded_dict = tokenizer.encode_plus(trancated_news_content,
                                                      add_special_tokens=True,
                                                      max_length=256,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt',
                                                      )
            input_ids = encoded_dict['input_ids'].to(DEVICE)
            attention_mask = encoded_dict['attention_mask'].to(DEVICE)
            content_embedding = model(input_ids, attention_mask=attention_mask)[1]
            content_embedding_list.append(content_embedding)
        content_embedding = torch.cat(content_embedding_list, dim=0)
        logits = pca.fit_transform(content_embedding.cpu().detach().numpy())
    return logits


def fake_news_detecting(batch_size, setting, news_ids_list, news_dict, veracity_dict,
                        comment_content_dict, comment_id_dict=None):
    logits = []
    veracity_labels = []
    real_comments_count = 0
    for i in range(0, len(news_ids_list), batch_size):
        print("Processing Batch: {} / {}".format(i // batch_size, len(news_ids_list) // batch_size))
        news_ids_batch = news_ids_list[i: i + batch_size]
        # [Setting 1]: News Content Only
        if setting == 1:
            news_content_batch = [news_dict[k] for k in news_ids_batch]
        # [Setting 2]: News Content with Common Users
        if setting == 2:
            news_content_batch = []
            real_news_comment_dict = comment_id_dict
            comments_dict = comment_content_dict
            for news_id in news_ids_batch:
                news_content = news_dict[news_id]
                comment_ids = real_news_comment_dict[news_id]
                real_comments_count += len(comment_ids)
                for comment_id in comment_ids:
                    news_content += " " + comments_dict[comment_id]
                news_content_batch.append(news_content)
        # [Setting 3]: News Content with Generated Users
        if setting == 3:
            news_content_batch = []
            generated_comment_dict = comment_content_dict
            for news_id in news_ids_batch:
                news_content = news_dict[news_id]
                generated_comment = generated_comment_dict[news_id]
                news_content += " " + generated_comment
                news_content_batch.append(news_content)

        veracity_batch = [veracity_dict[k] for k in news_ids_batch]
        veracity_batch = torch.tensor(veracity_batch)
        veracity_labels.append(veracity_batch)

        logits_batch = roberta_fnd(news_content_batch)
        logits.append(logits_batch)

    logits = np.concatenate(logits, 0)
    veracity_labels = torch.cat(veracity_labels, dim=0).cpu().detach().numpy()

    auc = compute_auc(logits, veracity_labels)
    accuracy = compute_accuracy(logits, veracity_labels)
    precision = compute_precision(logits, veracity_labels)
    recall = compute_recall(logits, veracity_labels)
    macro_f1 = compute_macro_f1(logits, veracity_labels)
    micro_f1 = compute_micro_f1(logits, veracity_labels)

    if setting == 1:
        print("======== Setting 1: News Content Only ========")
    elif setting == 2:
        print("======== Setting 2: News Content with Common Users ========")
        print("Real Comments Count: {}".format(real_comments_count))
    elif setting == 3:
        print("======== Setting 3: News Content with Generated Users ========")

    print("Precision: {:.4f}".format(precision))
    print("Accuracy: {:.4f}".format(accuracy))
    print("Recall: {:.4f}".format(recall))
    print("Micro F1: {:.4f}".format(micro_f1))
    print("Macro F1: {:.4f}".format(macro_f1))
    print("AUC: {:.4f}".format(auc))
