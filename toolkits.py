import os
import json
import h5py
import pickle
import random
import torch
import threading
import numpy as np
from filelock import FileLock, Timeout
from functools import wraps
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

random.seed(42)

save_lock = threading.Lock()


def save_ndjson(filename, key, value):
    with save_lock:
        with open("{}.ndjson".format(filename), 'a') as f:
            f.write(json.dumps({key: value}))
            f.write("\n")


def read_ndjson(filename):
    out_dict = {}
    with open("{}.ndjson".format(filename), "r") as f:
        for line in f:
            record = json.loads(line)
            out_dict.update(record)
    return out_dict


def read_ndjson_clean_save(raw_filename):
    out_dict = {}
    with open(f"{raw_filename}.ndjson", "r") as f:
        if "augmented_nu_dict" in raw_filename:
            for line in f:
                record = json.loads(line)
                record_key = list(record.keys())[0]
                if record_key in out_dict:
                    origin_value = out_dict[record_key]
                    origin_value.extend(record[record_key])
                    out_dict[record_key] = list(set(origin_value))
                else:
                    out_dict.update(record)
        else:
            for line in f:
                record = json.loads(line)
                out_dict.update(record)
    with open(f"{raw_filename}.ndjson", "w") as f:
        for k, v in out_dict.items():
            json_line = json.dumps({k: v})
            f.write(json_line + "\n")
    return out_dict


def with_filelock(func):
    """装饰器：给文件操作函数加 FileLock"""
    @wraps(func)
    def wrapper(filepath, *args, **kwargs):
        folder_path = os.path.dirname(filepath)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        lock_path = filepath + ".lock"
        lock = FileLock(lock_path)
        try:
            with lock.acquire(timeout=0):
                return func(filepath, *args, **kwargs)
        except Timeout:
            raise RuntimeError(f"[LOCKED]: File {filepath}")
    return wrapper


def read_dict(filename):
    with open(f"{filename}.json", 'r') as json_file:
        d = json.load(json_file)
    return d


@with_filelock
def save_dict(filename, input_dict):
    file_path = f"{filename}.json"
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    json_str = json.dumps(input_dict)
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)


def read_csv_np(filename):
    df = pd.read_csv(filename, header=None, low_memory=False).values
    npar = np.array(df)
    npar = np.delete(npar, 0, axis=0) 
    return npar


@with_filelock
def save_list(filename, input_list):
    # save list to pickel file
    order_list_dir = os.path.dirname(filename)
    if not os.path.exists(order_list_dir):
        os.makedirs(order_list_dir)
    with open("{}.pkl".format(filename), 'wb') as f:
        pickle.dump(input_list, f)


def read_list(filename):
    with open("{}.pkl".format(filename), 'rb') as f:
         output_list = pickle.load(f)
    return output_list


@with_filelock
def save_embedding_h5(filepath, embeddings, name, compression="gzip", compression_opts=4):
    embeddings_np = embeddings.cpu().numpy()
    folder_path = os.path.dirname(filepath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with h5py.File(f"{filepath}.h5", "w") as f:
        f.create_dataset(
            name,
            data=embeddings_np,
            compression=compression,
            compression_opts=compression_opts
        )


def read_embedding_h5(name, filepath):
    with h5py.File(f"{filepath}.h5", "r") as f:
        embeddings = torch.from_numpy(f[name][:])
    return embeddings


@with_filelock
def save_embedding_npz(embeddings, id, filepath):
    file_path = os.path.join(filepath, f"{id}.npz")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    embeddings_np = embeddings.cpu().numpy()
    np.savez_compressed(
        file_path,
        id=str(id),
        embedding=embeddings_np
    )


def read_embedding_npz(id, filepath):
    file_path = os.path.join(filepath, f"{id}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    data = np.load(file_path)
    embeddings = torch.from_numpy(data["embedding"])
    return embeddings


def save_embedding_from_batch_npz(filepath, embeddings, ids):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    embeddings_np = embeddings.cpu().numpy()
    for emb, id_ in zip(embeddings_np, ids):
        file_path = os.path.join(filepath, f"{id_}.npz")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        np.savez_compressed(
            file_path,
            id=str(id_),
            embedding=emb
        )


def read_embedding_to_batch_npz(ids, filename):
    embeddings = []
    for id_ in ids:
        file_path = os.path.join(filename, f"{id_}.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        data = np.load(file_path)
        emb = torch.from_numpy(data["embedding"]).unsqueeze(0)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def save_string(string, filename):
    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        f.write(string)


# ==================== Calculate Metrics ====================
def compute_auc(preds, labels):
    prob = softmax(preds, axis=-1)
    real_prob = prob[:, 1]
    # print("==========labels is {}=========".format(labels))
    auc = roc_auc_score(y_true=labels, y_score=real_prob)
    return auc


def compute_accuracy(preds, labels):
    predictions = np.argmax(preds, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    return accuracy


def compute_macro_f1(preds, labels):
    predictions = np.argmax(preds, axis=1)
    macro_f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")
    return macro_f1


def compute_micro_f1(preds, labels):
    predictions = np.argmax(preds, axis=1)
    micro_f1 = f1_score(y_true=labels, y_pred=predictions, average="micro")
    return micro_f1


def compute_precision(preds, labels):
    predictions = np.argmax(preds, axis=1)
    precision = precision_score(y_true=labels, y_pred=predictions, average="macro")
    return precision


def compute_recall(preds, labels):
    predictions = np.argmax(preds, axis=1)
    recall = recall_score(y_true=labels, y_pred=predictions, average="macro")
    return recall


def domain_ratio(news_id_list):
    politifact_count, gossipcop_count = 0, 0
    politifact_id_list, gossipcop_id_list = [], []
    for news_id in news_id_list:
        id = news_id.lower()
        if 'politifact' in id:
            politifact_count += 1
            politifact_id_list.append(news_id)
        elif 'gossipcop' in id:
            gossipcop_count += 1
            gossipcop_id_list.append(news_id)
    return politifact_count, gossipcop_count, politifact_id_list, gossipcop_id_list


def compute_metrics_print(logits, labels, print_tag, has_return=False):
    test_prec = compute_precision(logits, labels)
    test_rec = compute_recall(logits, labels)
    test_accuracy = compute_accuracy(logits, labels)
    test_micro_f1 = compute_micro_f1(logits, labels)
    test_macro_f1 = compute_macro_f1(logits, labels)
    if torch.unique(labels).numel() < 2:
        test_auc = 0.5
    else:
        test_auc = compute_auc(logits, labels)

    print("\n==================== {} ====================".format(print_tag))
    print("  Precision: {}  Recall: {}  Accuracy: {}  Micro F1: {}  Macro F1: {}  AUC: {}  ".format(
        test_prec, test_rec, test_accuracy, test_micro_f1, test_macro_f1, test_auc))
    if has_return:
        return test_prec, test_rec, test_accuracy, test_micro_f1, test_macro_f1, test_auc


# ==================== Training ====================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_macro_f1):
        score = val_macro_f1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"\nEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ==================== Statistics ====================
def plot_bar_chart_from_dict(u_n_number_dict):
    plt.bar(*zip(*u_n_number_dict.items()))
    plt.xticks([])
    plt.xlabel("User ID")
    plt.ylabel("News Number")
    plt.title("Real News Number")
    plt.show()


def compute_log_normal_params(u_n_number_dict):
    interaction_counts = np.array(list(u_n_number_dict.values()))
    log_interaction_counts = np.log(interaction_counts + 1)
    mu = np.mean(log_interaction_counts)
    sigma = np.std(log_interaction_counts)

    return mu, sigma


def plot_bar_chart_from_mu_sigma(mu, sigma, num_users=200):
    simulated_N = np.random.lognormal(mu, sigma, num_users).astype(int)
    sorted_N = sorted(simulated_N, reverse=True)
    plt.bar(range(num_users), sorted_N)
    plt.xticks([])
    plt.xlabel("User ID")
    plt.ylabel("News Number")
    plt.title("Simulated News Number")
    plt.show()


def dict_statistics(input_dict, info):
    """
    We use user_comments_dict and news_users_dict for aggregating. However, some users or news items
    are related to too many comments or users, which may lead to memory issues.
    This function is used to analyze the distribution of comments and users in the input_dict.
    It can help us determine the best truncation length.
    :param input_dict: uc_dict or nu_dict; {key1: [value1, value2], key2: [value3], ...}
    :return: values distribution
    """
    values_counts = [len(value_list) for value_list in input_dict.values()]
    percentiles = np.percentile(values_counts, [50, 75, 90, 95, 99])
    print(f"Statistics for {info}:")
    print(f"50% of the entries have a list length ≤ {percentiles[0]:.0f}")
    print(f"75% of the entries have a list length ≤ {percentiles[1]:.0f}")
    print(f"90% of the entries have a list length ≤ {percentiles[2]:.0f}")
    print(f"95% of the entries have a list length ≤ {percentiles[3]:.0f}")
    print(f"99% of the entries have a list length ≤ {percentiles[4]:.0f}")

    mean_comments = np.mean(values_counts)
    median_comments = np.median(values_counts)
    max_comments = np.max(values_counts)
    min_comments = np.min(values_counts)
    std_comments = np.std(values_counts)

    # Print results
    print(f"Total number of entries: {len(values_counts)}")
    print(f"Average number of items: {mean_comments:.2f}")
    print(f"Median number of items: {median_comments}")
    print(f"Maximum number of items: {max_comments}")
    print(f"Minimum number of items: {min_comments}")
    print(f"Standard deviation: {std_comments:.2f}")


# ==================== Simulation ====================
def simulate_user_news_comments(num_users, comments_total_politics, comments_total_entertainment,
                                political_news_list, entertainment_news_list):

    user_politics_comments = np.random.multinomial(comments_total_politics, [1 / num_users] * num_users)
    user_entertainment_comments = np.random.multinomial(comments_total_entertainment, [1 / num_users] * num_users)

    user_engagements = {}
    for user_id in range(num_users):
        politics_comments = user_politics_comments[user_id]
        entertainment_comments = user_entertainment_comments[user_id]

        politics_news_selected = random.choices(political_news_list, k=politics_comments)
        entertainment_news_selected = random.choices(entertainment_news_list, k=entertainment_comments)

        politics_news_counts = {news: politics_news_selected.count(news)
                                for news in set(politics_news_selected)}
        entertainment_news_counts = {news: entertainment_news_selected.count(news)
                                     for news in set(entertainment_news_selected)}

        user_engagements[f"user_{user_id}"] = {
            "politics": politics_news_counts,
            "entertainment": entertainment_news_counts
        }

    return user_engagements

# ==================== Datasets ====================
def dataset_split(news_set, labels_set, val_size):
    train_test_set, val_set, train_test_label, val_label = train_test_split(news_set, labels_set,
                                                                            test_size=val_size, random_state=42,
                                                                            shuffle=True, stratify=labels_set)
    train_test_size = 1 - val_size
    test_size = val_size
    split_size = test_size / train_test_size
    train_set, test_set, train_label, test_label = train_test_split(train_test_set, train_test_label,
                                                                    test_size=split_size, random_state=42,
                                                                    shuffle=True, stratify=train_test_label)
    return train_set, test_set, val_set, train_label, test_label, val_label


def news_dataloader(news_idx_set, veracity_labels, domain_labels, batch_size, shuffle=False):
    news_idx_set = torch.tensor(news_idx_set)
    veracity_labels = torch.tensor(veracity_labels)
    domain_labels = torch.tensor(domain_labels)

    dataset = NewsDataset(news_idx_set, veracity_labels, domain_labels)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        worker_init_fn=_init_fn,
        drop_last=True,
    )
    return dataloader


def get_text_data_batch(news_idx_batch, idx_content_dict):
    news_idx = news_idx_batch
    batch_content_list = []
    for idx in news_idx:
        idx_str = str(idx.item())
        batch_content_list.append(idx_content_dict[idx_str])
    return batch_content_list


def _init_fn():
    np.random.seed(2021)


class NewsDataset(Dataset):
    def __init__(self, news_idx_set, veracity_label, domain_label):
        self.news_idx = news_idx_set
        self.veracity_label = veracity_label  # tensor
        self.domain_label = domain_label

    def __len__(self):
        return len(self.veracity_label)

    def __getitem__(self, index):
        return (self.news_idx[index],
                self.veracity_label[index],
                self.domain_label[index],
                )

    def collate_fn(data):
        return data


# performance improvement calculate
def calculate_performance_improvement(p1_metrics_list, p2_metrics_list):
    p1_metrics = np.array(p1_metrics_list)
    p2_metrics = np.array(p2_metrics_list)
    improvement = (p2_metrics - p1_metrics) / p1_metrics * 100
    improvement = np.round(improvement, 2)
    avg_improvement = np.mean(improvement)
    print("Performance Improvement: {}".format(improvement))
    print("AVG Performance Improvement: {}%".format(avg_improvement))