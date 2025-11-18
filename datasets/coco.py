import os
import json
import re
import ast
import math
from torch.utils.data import Dataset
import numpy as np
import clip
from PIL import Image
from pycocotools.coco import COCO
import nltk

from vocab import deserialize_vocab

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

@DATASET_REGISTRY.register()
class coco(DatasetBase):

    dataset_dir = "prepare_data/coco"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.vocab = deserialize_vocab("vocab/mscoco_split2014_vocab.json")
        train_text_file = os.path.join(self.dataset_dir, 'train.txt')
        test_text_file = os.path.join(self.dataset_dir, 'testall.txt')

        train_data = self.read_data(train_text_file, 'triplets/coco/train_action_caption_gpt_azure', 'train')
        test_data = self.read_data(test_text_file, 'triplets/coco/test_action_caption_gpt', 'test')

        super().__init__(train_x=train_data, test=test_data)

    def extract_gpt_texts(self, gpt_list):
        texts = []
        for item in gpt_list:
            if isinstance(item, str):
                if item.startswith('{') and item.endswith('}'):
                    item_new = item.replace("'s", "s")
                    item_new = item_new.replace("don't", "do not")
                    item_dict = ast.literal_eval(item_new)
                    texts.extend(item_dict.values())
                else:
                    texts.append(item)
        return texts

    def read_data(self, path_file, path_cap_file, split):
        items = []

        with open(path_file, 'r', encoding='utf-8') as f:
            folders = f.readlines()
        data_rel = []

        captions = []

        for label, folder in enumerate(folders):
            folder = folder.strip("\n")
            split_list = folder.split('*')
            impath = split_list[1]
            classname = str(split_list[2])
            captions.append(classname)

        for i in range(len(folders)):
            rel_list = []
            
            json_data = os.path.join(self.dataset_dir, path_cap_file, str(i) + '.json')
            with open(json_data, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            triple_str = data.get("Triple", "[]")
            triplet_list = ast.literal_eval(triple_str)

            if len(triplet_list) == 0:
                rel = 'subject' + ' ' + 'action' + ' ' + 'object'
                tokens = nltk.tokenize.word_tokenize(str(rel).lower())
                rel = []
                rel.append(self.vocab('<start>'))
                rel.extend([self.vocab(token) for token in tokens])
                rel.append(self.vocab('<end>'))
                rel_list.append(rel)
            else:
                for triplet_str in triplet_list:
                    triplet_str = triplet_str.strip('<>')
                    triplet_parts = [part.strip() for part in triplet_str.split(',')]
                    
                    if len(triplet_parts) >= 3:
                        rel = triplet_parts[0] + ' ' + triplet_parts[2] + ' ' + triplet_parts[1]
                    else:
                        rel = ' '.join(triplet_parts)
                    
                    tokens = nltk.tokenize.word_tokenize(str(rel).lower())
                    rel = []
                    rel.append(self.vocab('<start>'))
                    rel.extend([self.vocab(token) for token in tokens])
                    rel.append(self.vocab('<end>'))
                    rel_list.append(rel)

            caps = []

            try:
                description = json.loads(data["Description"])
                if len(description) != 0:
                    for key in description.keys():
                        caps.append(description[key])
                else:
                    caps.append('NULL')
            except json.JSONDecodeError:
                senten = data["Description"]
                error_prefixes = [
                    "I'm sorry", "I believe there", "To describe the action",
                    "The action", "Here is the JSON file with the",
                    "I will provide a concise", "Based on the triplet",
                    "The description of the action", "To generate a concise description",
                    "I think there might be a misunderstanding", "Certainly!",
                    "I think there might", "It seems like there",
                    "The description for the action", "The triplet"
                ]
                
                if any(senten.startswith(prefix) for prefix in error_prefixes):
                    caps.append('NULL')
                else:
                    caps.append(senten)

            pattern = re.compile(r'(?<!\{)\n  \"')
            cap1 = [pattern.sub(',\n  "', sentence) for sentence in caps]
            cap2 = [item.replace('",\n}', '"\n}') for item in cap1]
            cap = [re.sub(r',+', ',', item) for item in cap2]
            cap = self.extract_gpt_texts(cap)

            rel_list.append(captions[i])
            rel_list.append(cap)
            rel_list = str(rel_list)
            data_rel.append(rel_list)

        for label, folder in enumerate(folders):
            folder = folder.strip("\n")
            split_list = folder.split('*')
            impath = split_list[1]
            classname = str(split_list[2])

            item = Datum(impath=impath, label=label, classname=classname, domain=data_rel[label])
            items.append(item)

        return items
