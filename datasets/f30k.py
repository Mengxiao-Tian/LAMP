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
class f30k(DatasetBase):

    dataset_dir = "prepare_data/flickr"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.vocab = deserialize_vocab("vocab/flickr30k_split_vocab.json")
        train_text_file = os.path.join(self.dataset_dir, 'train_f30k.txt')
        test_text_file = os.path.join(self.dataset_dir, 'test_f30k.txt')

        train_data = self.read_data(train_text_file, 'triplets/flickr/train_action_caption_gpt_azure_f30k_add_queshi', 'train')
        test_data = self.read_data(test_text_file, 'triplets/flickr/test_action_caption_gpt_f30k', 'test')

        super().__init__(train_x=train_data, test=test_data)


    def extract_gpt_texts(self, gpt_list, template):
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

            template = "The fine-grained state description of action triplet {} is {}"

            try:
                if i == 86985 or i == 107245:
                    description = data["Description"]
                    for key in description.keys():
                        result = template.format(key, description[key])
                        caps.append(result)
                else:
                    description = json.loads(data["Description"])
                    if len(description) != 0:
                        for key in description.keys():
                            result = template.format(key, description[key])
                            caps.append(result)
                    else:
                        caps.append('NULL')
            except json.JSONDecodeError:
                senten = data["Description"]
                error_prefixes = [
                    "I'm sorry", "I believe there", "To describe the action",
                    "The action", "Here is the JSON file with the",
                    "I will provide a concise", "Based on the triplet",
                    "The description of the action", "Based on the structure of the sentence and the provided",
                    "To generate a concise description", "I think there might be a misunderstanding",
                    "Certainly!", "I think there might", "It seems like there",
                    "The description for the action", "The triplet",
                    "Examples of concise descriptions", "I will provide concise descriptions",
                    "After analyzing the action", "Here are the descriptions",
                    "Based on the context of the sentence", "To provide a concise description of the action",
                    "I understand that you would like a description",
                    "From the context provided in the sentence", "To generate the description for"
                ]
                
                if any(senten.startswith(prefix) for prefix in error_prefixes):
                    caps.append('NULL')
                else:
                    pattern = re.compile(r'(?<!\{)\n  \"')
                    senten = pattern.sub(',\n  "', senten)
                    senten = senten.replace('",\n}', '"\n}')
                    senten = re.sub(r',+', ',', senten)
                    senten = re.sub(r"'(.*?)'", r'"\1"', senten)
                    senten = re.sub(r"'s\b|\"s\b", "", senten)
                    match = re.search(r'"([^"]*)"', senten)
                    
                    if match:
                        matches = re.findall(r"It seems there might|Apologies|I will generate a concise", senten)
                        if not matches:
                            senten = senten.replace("'", "\"")
                            senten = json.loads(senten)
                            for key in senten.keys():
                                result = template.format(key, senten[key])
                                caps.append(result)
                        else:
                            caps.append('NULL')
                    else:
                        caps.append('NULL')

            rel_list.append(captions[i])
            rel_list.append(caps)
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
