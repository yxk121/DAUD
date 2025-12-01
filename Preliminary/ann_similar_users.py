import random
from tqdm import tqdm
import torch
import numpy as np
import faiss
import os
from Preliminary.disentanglers import encode_content
from toolkits import read_embedding_to_batch_npz, save_embedding_from_batch_npz, news_dataloader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

random.seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsSearch:
    def __init__(self, neighbor_num, similarity_threshold, idx_dict_list, news_features_dict, lm_list,
                 selected_user_num, get_subset, root_path):
        self.neighbor_num = neighbor_num
        self.similarity_threshold = similarity_threshold
        self.news_embeddings_pooled, self.current_idx_row_dict = load_news_embeddings(idx_dict_list,
                                                                                      news_features_dict,
                                                                                      lm_list,
                                                                                      selected_user_num,
                                                                                      get_subset, root_path)
        self.news_embeddings_pooled = self.news_embeddings_pooled.squeeze(1)
        embedding_dim = self.news_embeddings_pooled.shape[1]
        self.news_embeddings_pooled = self.news_embeddings_pooled.cpu().numpy()
        res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(embedding_dim))
        self.gpu_index.add(self.news_embeddings_pooled)

    def _get_similar_news(self, engagements_idx):
        row_idx_dixt = {v: k for k, v in self.current_idx_row_dict.items()}
        engagements_row = [int(self.current_idx_row_dict[int(idx)]) for idx in engagements_idx]

        engaged_news_embeddings_pooled = self.news_embeddings_pooled[engagements_row]
        D, I = self.gpu_index.search(engaged_news_embeddings_pooled, self.neighbor_num + 1)

        indices = I.flatten()
        distances = D.flatten()
        mask = distances < self.similarity_threshold
        similar_news_row = set(indices[mask])
        similar_news_dist = distances[mask]
        similar_news_idx_all = set([str(row_idx_dixt[int(row)]) for row in similar_news_row])
        similar_news_idx = similar_news_idx_all - set(engagements_idx)
        return list(similar_news_idx)

    def engagement_augmentation(self, u_at_dict, news_id_idx_dict):
        similar_news_dict = {}
        for user_id in list(u_at_dict.keys()):
            records = u_at_dict[user_id]
            engagements_idx = [news_id_idx_dict[news_id] for news_id, _ in records]
            similar_news_idx = self._get_similar_news(engagements_idx)
            similar_news_dict[user_id] = similar_news_idx

        return similar_news_dict


def load_news_embeddings(idx_dict_list, news_features_dict,
                         lm_list, selected_user_num, get_subset, root_path):
    all_included_news = []
    for idx_dict in idx_dict_list:
        all_included_news += list(idx_dict.keys())

    file_path = root_path + "Results/domain_aware_agents/"
    if get_subset:
        news_features_path = file_path + f"news_features/{selected_user_num}/"
    else:
        news_features_path = file_path + "news_features/all_users/"

    news_features_embeddings_list, news_features_mask_list = [], []
    batches = _split_into_batches(all_included_news, batch_size=128)
    for news_batch in tqdm(batches, desc="encoding news features"):
        try:
            news_features_embeddings_batch_pooled = read_embedding_to_batch_npz(news_batch,
                                                                                news_features_path + "embedding/").to(DEVICE)
            news_features_embeddings_list.append(news_features_embeddings_batch_pooled)
        except FileNotFoundError:
            new_features_list = []
            for news_idx in news_batch:
                news_idx = str(news_idx)
                each_content = news_features_dict.get(news_idx, "")
                new_features_list.append(each_content)
            news_features_embeddings_batch, news_features_mask_batch = encode_content(lm_list, new_features_list,
                                                                                      max_length=256)
            news_features_embeddings_batch_pooled = mean_max_pooling(news_features_embeddings_batch,
                                                                     news_features_mask_batch)

            news_features_embeddings_list.append(news_features_embeddings_batch_pooled)

            save_embedding_from_batch_npz(news_features_path + "embedding/", news_features_embeddings_batch_pooled,
                                          news_batch)

    news_features_embeddings = torch.cat(news_features_embeddings_list, dim=0)
    idx_row_dict = {idx: row for row, idx in enumerate(all_included_news)}

    return news_features_embeddings, idx_row_dict


def _split_into_batches(data_list, batch_size):
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def mean_max_pooling(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9) 
    mean_pooled = sum_embeddings / sum_mask
    last_hidden_state_masked = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
    max_pooled = torch.max(last_hidden_state_masked, dim=1).values
    pooled = torch.cat([mean_pooled, max_pooled], dim=1)
    return pooled