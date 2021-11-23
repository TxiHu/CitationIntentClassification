import sklearn
import random
from utils.util import *
import nltk
from nltk.corpus import stopwords
import collections
from pathlib import Path
import re


def reverse_sampler(train_data):
    random.seed(0)
    counter_train = dict(train_data['citation_class_label'].value_counts())
    num_list = list(counter_train.values())
    class_list = list(counter_train.keys())
    max_num = max(num_list)
    class_weight = [max_num / i for i in num_list]
    sum_weight = sum(class_weight)
    class_dict = dict()
    sampled_examples = []
    for i in class_list:
        class_dict[i] = train_data[train_data['citation_class_label'] == i].index.values.tolist()
    total_samples = sum(num_list)
    for _ in range(total_samples):
        rand_number, now_sum = random.random() * sum_weight, 0
        for j in class_list:
            now_sum += class_weight[class_list.index(j)]
            if rand_number <= now_sum:
                sampled_examples.append(random.choice(class_dict[j]))
                break
    print('reverse sample count{}'.format(collections.Counter(train_data.iloc[sampled_examples, :]['citation_class_label'])))
    revere_data = train_data.iloc[sampled_examples, :].reset_index(drop=True)   # 不reset_index的话因为重复采样会出现多个相同的索引
    return revere_data


def delete_aug(data):
    for index in range(data.shape[0]):
        # cited_author = data['cited_author'][index]
        citation_text = re.sub(r"#AUTHOR_TAG", ' ', data['citation_context'][index])
        data.loc[index, 'citation_context'] = citation_text
    return data.loc[:, ('citation_context', 'citation_class_label')]


def generate_batch_data(data, batch_size=16):
    print('train generate_batch_data')
    stop_words = stopwords.words('english')
    stop_words = ['et', 'al', 'e', 'g'] + stop_words
    batch_count = int(data.shape[0] / batch_size)
    sentences_list, target_list = [], [],
    for i in range(batch_count):
        mini_batch_sentences, mini_batch_target = [], []
        for j in range(batch_size):
            citation_text = re.sub(r'[^a-zA-Z]', ' ', data['citation_context'][i * batch_size + j]).lower()
            citation_text = nltk.word_tokenize(citation_text)
            citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
            mini_batch_sentences.append(citation_text)
            mini_batch_target.append(data['citation_class_label'][i * batch_size + j])
        sentences_list.append(mini_batch_sentences)
        target_list.append(mini_batch_target)
    if data.shape[0] % batch_size != 0:
        last_sentences_list = []
        last_target_list = []
        for i in range(batch_count * batch_size, data.shape[0]):
            citation_text = re.sub(r'[^a-zA-Z]', ' ', data['citation_context'][i]).lower()
            citation_text = nltk.word_tokenize(citation_text)
            citation_text = [word for word in citation_text if (word not in stop_words and len(word) > 1)]
            last_sentences_list.append(citation_text)
            last_target_list.append(data['citation_class_label'][i])
        sentences_list.append(last_sentences_list)
        target_list.append(last_target_list)
    return {'sen': sentences_list, 'tar': target_list}


def load_data(batch_size=None):
    assert batch_size is not None
    data = {}
    path = Path('/content/citation_classification') # root path
    train_set = pd.read_csv(path / 'dataset/SDP_train.csv', sep=',')
    test = pd.read_csv(path / 'dataset/SDP_test.csv', sep=',').merge(
        pd.read_csv(path / 'dataset/sample_submission.csv'), on='unique_id')
    train_set = sklearn.utils.shuffle(train_set, random_state=0).reset_index(drop=True)
    train = train_set.loc[:int(train_set.shape[0] * 0.8) - 1]
    print(train['citation_class_label'].value_counts())
    print(collections.Counter(train['citation_class_label']).items())
    val = (train_set.loc[int(train_set.shape[0] * 0.8):]).reset_index(drop=True)

    reverse_data = reverse_sampler(train)
    reverse_data = delete_aug(reverse_data)
    data['reverse'] = generate_batch_data(reverse_data, batch_size)

    mul_sec = pd.read_csv(path / 'dataset/section_name.csv')
    mul_num = train.shape[0]
    mul_section = mul_sec.head(mul_num)
    mul_section_batch = generate_batch_data(mul_section, mul_section.shape[0] // (train.shape[0]//batch_size))
    data['section'] = mul_section_batch

    train = delete_aug(train)
    val = delete_aug(val)
    test = delete_aug(test)
    data['train'] = generate_batch_data(train, batch_size)
    data['val'] = generate_batch_data(val, batch_size)
    data['test'] = generate_batch_data(test, batch_size)

    return data