#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

class TX2TInstructionDataset(Dataset):
    """
    T + X - T instruction Dataset
    """
    def __init__(self, data_path: str, mm_root_path: str = None, dataset_type: str='ImageToText'):
        super(TX2TInstructionDataset, self).__init__()

        self.mm_root_path = mm_root_path
        self.instruction_list = []
        self.mm_path_list = []
        self.dataset_category = 't2t' if mm_root_path is None else 'tx2t'
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for instance in tqdm(res, total=len(res)):
            # import pdb;pdb.set_trace()
            self.instruction_list.append(instance['conversations'])
            if self.dataset_category == 'tx2t':
                # Text + X -> Text dataset
                if instance['input_modality'] == 'image':
                    self.mm_path_list.append(os.path.join(mm_root_path, "image", instance['image_path']))
                if instance['input_modality'] == 'video':
                    self.mm_path_list.append(os.path.join(mm_root_path,  instance['video_path']))
                if instance['input_modality'] == 'audio':
                    self.mm_path_list.append(os.path.join(mm_root_path, "audio", instance['audio_path']))

        self.dataset_type_list = [dataset_type for _ in range(len(self.instruction_list))]
        print(f'[!] collect {len(res)} samples for training')

    def __len__(self):  # number of instances
        return len(self.instruction_list)

    def __getitem__(self, i):
        if self.dataset_category == 'tx2t':
            # Text + X -> Text dataset
            return dict(mm_paths=self.mm_path_list[i], output_texts=self.instruction_list[i],
                        dataset_types=self.dataset_type_list[i])
        else:
            # Text -> Text dataset
            return dict(output_texts=self.instruction_list[i], dataset_types=self.dataset_type_list[i])

    def collate(self, instances):
        if self.dataset_category == 'tx2t':
            mm_paths, output_texts, dataset_types = tuple(
                [instance[key] for instance in instances] for key in ("mm_paths", "output_texts", "dataset_types"))
            return dict(
                mm_paths=mm_paths,
                output_texts=output_texts,
                dataset_types=dataset_types
            )
        else:
            output_texts, dataset_types = tuple(
                [instance[key] for instance in instances] for key in ("output_texts", "dataset_types"))
            return dict(
                output_texts=output_texts,
                dataset_types=dataset_types
            )

