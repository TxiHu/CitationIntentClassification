# -*- coding: utf-8 -*-
import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import json
import collections

root_path = Path('/content/citation_classification/dataset')
section = root_path / 'sections-scaffold-train.jsonl'

section_text = []
section_name = []
section_dict = {'introduction': 0, 'related work': 1, 'method': 2, 'experiments': 3, 'conclusion': 4}
with jsonlines.open(section, mode='r') as reader:
    for row in reader:
        section_text.append(row['text'])  # 原文
        section_name.append(section_dict[row['section_name']])
print(section_name)
print(collections.Counter(section_name))
section_location = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
for i in range(len(section_name)):
    section_location.loc[i] = {'citation_context': section_text[i],
                               'citation_class_label': section_name[i]}
section_location.to_csv('/content/citation_classification/dataset/section_name.csv', sep=',', index=False)
