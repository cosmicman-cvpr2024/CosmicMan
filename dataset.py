#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers.utils import check_min_version
from PIL import  ImageOps, ImageFile
from PIL.Image import Image as Img
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import itertools
import json


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

from typing import List, Tuple

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')



class ImageStore:
    def __init__(self, args, logger=None) -> None:
        json_files = [f for f in os.listdir(args.dataset)]
        json_files = sorted(json_files, key=self.custom_sort)[:3]
        self.train_data_root = args.train_data_root

        meta_data = []
        for json_i, json_name in enumerate(json_files):
            # logger.info("load json_{}: {}".format(json_i, json_name))
            meta_data += self.load_data(os.path.join(args.dataset, json_name))
        random.shuffle(meta_data)
        self.meta_data = meta_data
        self.logger = logger
        logger.info("number of all img is {}".format(len(meta_data)))

        self.num_of_parsing_categories = args.num_of_parsing_categories
        self.dropout_rate_of_parsing = args.dropout_rate_of_parsing
        self.all_aa_text_keys = [
            (['country','age'], ["gender"]),
            (['body shape'], ""),
            (['background'], ""),
            (['hair color','hair style','hair length'], "hair"),
            (['sleeve length'], ['type']),
            (['pattern','material','sleeve length','collar shape','length','shoulder exposure level'], ['type']),
            (['pattern','material','sleeve length','top length','collar shape', 'color', 'graphic'], ['type']),
            (['pattern','material','coat length','collar shape', 'color','graphic'], ['type']),
            (['pattern','material','length','bottom shape', 'color', 'graphic'], ['type']),
            (['pattern','material','boots length', 'color'], ['type']),
            (['material'], ['type']),
            (['material'], ['type']),
            (['material'], "belt"),
            (['pattern','material'], "scarf"),
            (['overall-style'], ""),
            (['pattern', 'material'], "headband"),
            (['pattern', 'material'], "headscarf"),
            (['pattern', 'material'], "veil"),
            (['pattern', 'material'], "socks"),
            (['pattern', 'material'], "ties"),
        ]
        self.person_type_dict = {
            'full_body': 'a full-body shot',
            'nearly_full_body': 'a nearly full-body shot',
            'upper_body': 'a upper body shot',
            'portrait': 'a close up portrait shot',
            'headshot': 'a headshot',
        }

    def load_data(self, parquet_file_path):
        df = pd.read_parquet(parquet_file_path)
        def convert_to_dict(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return x
        df['aa_text'] = df['aa_text'].apply(convert_to_dict)
        df['aa_ids'] = df['aa_ids'].apply(convert_to_dict)
        json_str = df.to_json(orient='records')
        json_list = json.loads(json_str)
        return json_list

    def __len__(self) -> int:
        return len(self.meta_data)

    def custom_sort(self, item):
        if item.startswith('laion1B-nolang'):
            type_order = 0
        else:
            type_order = 1
        number = int(item.split('_')[1])
        return (type_order, number)

    def resize_rgb(self, image_path: str, w: int, h: int) -> Img:
        return ImageOps.fit(
                Image.open(image_path),
                (w, h),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS
            ).convert(mode='RGB')

    def resize_gray(self, image_path: str, w: int, h: int) -> Img:
        return ImageOps.fit(Image.open(image_path),
                    (w, h),
                    bleed=0.0,
                    centering=(0.5, 0.5),
                    method=Image.Resampling.LANCZOS)

    def entries_iterator(self) -> Generator[Tuple[list, int], None, None]:
        for f in range(len(self)):
            img_size = (int(float(self.meta_data[f]["height"])), int(float(self.meta_data[f]["width"])))
            yield img_size, f

    def concat_adj(self, attr, type_):
        k = len(attr)
        res = ''
        for i in range(k):
            res += attr[i]
            res += ' '
        return res + type_ if type_ != "" else res[:-1]

    def get_aa_text_i(self, index, aa_text_label_i):
        if index >= 14 and random.random() < 0.5:
            return ""

        adj_list = []
        for adj_key in self.all_aa_text_keys[index][0]:
            if adj_key in aa_text_label_i.keys():
                adj_list += [aa_text_label_i[adj_key]]
        random.shuffle(adj_list)
        try:
            body_part_text_i = self.concat_adj(
                adj_list,
                aa_text_label_i[self.all_aa_text_keys[index][1][0]] if isinstance(self.all_aa_text_keys[index][1], list) else self.all_aa_text_keys[index][1],
            )
            body_part_text_i = "a " + body_part_text_i if index == 0 else body_part_text_i
        except:
            body_part_text_i = ""
        return body_part_text_i

    def get_image_and_parsing(self, ref: Tuple[int, int, int]) -> Img:
        try:
            # Load img and original parsing
            img = self.resize_rgb(os.path.join(self.train_data_root, self.meta_data[ref[0]]["path"]), ref[1], ref[2])
            parsing = self.resize_gray(os.path.join(self.train_data_root, self.meta_data[ref[0]]["segm_path"]), ref[1], ref[2])

            # Convert each parsing to one_hot mask
            one_hot_parsing = np.eye(self.num_of_parsing_categories + 1)[np.array(parsing)]
            aa_ids = self.meta_data[ref[0]]["aa_ids"]

            convert_each_parsing_to_one_hot = []
            for aa_ids_index_i, aa_ids_i in enumerate(aa_ids):
                if aa_ids_index_i == 1: # type: body shape, parsing: foreground = 1 - background
                    convert_each_parsing_to_one_hot.append(1 - one_hot_parsing[:, :, 0])
                else:
                    if aa_ids_i is not None:
                        zero_parsing = np.zeros(one_hot_parsing.shape[:2])
                        for prasing_i in aa_ids_i:
                            zero_parsing = np.logical_or(zero_parsing, one_hot_parsing[:, :, prasing_i]).astype(np.int32)
                        convert_each_parsing_to_one_hot.append(zero_parsing)
                    else:
                        convert_each_parsing_to_one_hot.append(one_hot_parsing[:, :, self.num_of_parsing_categories])
            parsing = [Image.fromarray(p) for p in convert_each_parsing_to_one_hot]

            body_part_text = []
            for aa_text_index_i, aa_text_label_i in enumerate(self.meta_data[ref[0]]["aa_text"]):
                if aa_text_label_i is not None:
                    body_part_text.append(self.get_aa_text_i(aa_text_index_i, aa_text_label_i))
                else:
                    body_part_text.append("")
            
            # Shuffle
            set_parsing_range = [0, 1, 2]
            parsing_range = [i for i in range(3, len(body_part_text))]
            random.shuffle(parsing_range)
            parsing_range = set_parsing_range + parsing_range

            # Dropout parsing
            new_parsing = []
            new_body_part_text = []
            for i in parsing_range:
                if random.random() > self.dropout_rate_of_parsing:
                    new_parsing.append(parsing[i])
                    new_body_part_text.append(body_part_text[i])
                else:
                    new_parsing.append(Image.fromarray(one_hot_parsing[:, :, self.num_of_parsing_categories]))
                    new_body_part_text.append("")

            if "person_type" in self.meta_data[ref[0]]:
                person_type = self.meta_data[ref[0]]["person_type"]
                person_type_text = self.person_type_dict[person_type] if person_type in self.person_type_dict.keys() else ''
            else:
                person_type_text = ""

            aa_text = ""
            new_body_part_text = [person_type_text] + new_body_part_text
            new_parsing = [Image.fromarray(1 - one_hot_parsing[:, :, 0])] + new_parsing
            for p_text_i, p_text in enumerate(new_body_part_text):
                if p_text != '':
                    if p_text_i != 0:
                        aa_text += ', '
                        aa_text += p_text
                    else:
                        aa_text += p_text
            return img, new_parsing, new_body_part_text, aa_text
        except:
            for i in range(10):
                try:       
                    ind = random.randint(0,len(self.meta_data))
                    return self.get_image_and_parsing((ind, ref[1], ref[2]))
                except:
                    continue
            raise RuntimeError("Too much bad data.")

class AspectBucket:
    def __init__(self, store: ImageStore,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 512,
                 bucket_side_max: int = 1024,
                 bucket_side_increment: int = 64,
                 max_image_area: int = 1024 * 1024,
                 max_ratio: float = 2):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()

        self.init_buckets()
        self.fill_buckets()

    def _sort_by_ratio(self, bucket: tuple) -> float:
        return bucket[0] / bucket[1]

    def _sort_by_area(self, bucket: tuple) -> float:
        return bucket[0] * bucket[1]

    def init_buckets(self):
        possible_lengths = list(range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                        if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)
        buckets_by_ratio = {}

        # group the buckets by their as pect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=self._sort_by_area)[-1]
                                        for buckets in buckets_by_ratio.values()], key=self._sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {*np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=self._sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)
        # print("self.__bucket_interp", self._bucket_ratios)
        for b in buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            (index, w, h)

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in (entries):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, size: list, index: int) -> bool:
        aspect = size[1]/size[0]
        
        # print(aspect)
        # aspect = entry.width / entry.height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        # del entry

        return True

class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, num_replicas: int = 1, rank: int = 0):
        super().__init__(None)
        self.bucket = bucket
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.bucket.get_batch_iterator()
        indices = list(indices)[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.bucket.get_batch_count() // self.num_replicas

class SD_AspectDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            store: ImageStore, 
            tokenizer: CLIPTokenizer, 
            text_encoder: CLIPTextModel, 
            device: torch.device, 
            ucg: float = 0.1, 
            logger=None
        ):

        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.ucg = ucg
        self.logger = logger

        if type(self.text_encoder) is torch.nn.parallel.DistributedDataParallel:
            self.text_encoder = self.text_encoder.module

        self.transform_norm = transforms.Compose([
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def find_last_non_empty_element_index(self, lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != "":
                return i
        return -1

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {}
        image_file, parsing_mask, parsing_text, aa_text = self.store.get_image_and_parsing(item)

        p = np.random.choice([0, 1])
        transform_share = transforms.Compose([
            transforms.RandomHorizontalFlip(p),
            transforms.ToTensor()
        ])
        return_dict['pixel_values'] = self.transform_norm(transform_share(image_file))
        return_dict['parsing_mask'] = torch.stack([transform_share(p) for p in parsing_mask], dim=0)

        aa_token_len = len(self.tokenizer(aa_text).input_ids)
        while aa_token_len > 77:
            find_last_non_empty_element_i = self.find_last_non_empty_element_index(parsing_text)
            delete_len = len(", " + parsing_text[find_last_non_empty_element_i])
            aa_text = aa_text[:-1 * delete_len]
            parsing_text[find_last_non_empty_element_i] = ""

            aa_token_len = len(self.tokenizer(aa_text).input_ids)

        return_dict['parsing_text'] = parsing_text
        return_dict['aa_text'] = aa_text

        return return_dict

    def find_sublist_index(self, sublist, mainlist, subtext, maintext, logger):
        if len(sublist) == 0:
            return None
        index_list = []
        sublist_length = len(sublist)
        for i in range(len(mainlist)):
            if mainlist[i] == sublist[0]:
                if mainlist[i:i+sublist_length].tolist() == sublist:
                    index_list.append(i)
        if len(index_list) == 1:
            return (index_list[0], index_list[0] + len(sublist))
        else:
            if len(index_list) > 1:
                logger.warning("find_sublist_index error: {} in {}".format(subtext, maintext))
            return None

    def collate_fn(self, examples):
        if random.random() > self.ucg:
            pixel_values = torch.stack([example['pixel_values'] for example in examples])
            pixel_values.to(memory_format=torch.contiguous_format).float()
            parsing_mask = torch.stack([example['parsing_mask'] for example in examples])
            parsing_mask.to(memory_format=torch.contiguous_format).float()

            prompt = [example['aa_text'] for example in examples]
            # Get text embedding
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_inputs_mask = text_inputs.attention_mask

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            parse_index = []
            parse_split = []
            for example_i, example in enumerate(examples):
                parse_index += [self.find_sublist_index(self.tokenizer(text).input_ids[1:-1], text_input_ids[example_i], text, prompt[example_i], logger) for text in example['parsing_text']]
                parse_split += [(0, len(example['parsing_text'])) if len(parse_split) == 0 else (parse_split[-1][1], parse_split[-1][1] + len(example['parsing_text']))]

            return {
                'pixel_values': pixel_values,
                'parsing_mask': parsing_mask,

                'input_ids': prompt_embeds,

                'parse_split': parse_split,
                'parse_index': parse_index,
                'prompt': prompt,
                'text_inputs_mask': text_inputs_mask,

                'original_size': pixel_values.shape[-2:],
                'use_parsing': not all(all(parsing_text_i == '' for parsing_text_i in example['parsing_text']) for example in examples),
            }
        else:
            pixel_values = torch.stack([example['pixel_values'] for example in examples])
            pixel_values.to(memory_format=torch.contiguous_format).float()

            prompt = ["" for example in examples]
            # Get text embedding
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_inputs_mask = text_inputs.attention_mask

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

            return {
                'pixel_values': pixel_values,
                'input_ids': prompt_embeds,
                'prompt': prompt,
                'text_inputs_mask': text_inputs_mask,
                'original_size': pixel_values.shape[-2:],
                'use_parsing': False,
            }

class SDXL_AspectDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            store: ImageStore, 
            tokenizer: CLIPTokenizer, 
            text_encoder: CLIPTextModel, 
            tokenizer_2: CLIPTokenizer, 
            text_encoder_2: CLIPTextModelWithProjection,  
            device: torch.device, 
            ucg: float = 0.1, 
            logger=None
        ):

        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.tokenizer_2 = tokenizer_2
        self.text_encoder_2 = text_encoder_2
        self.device = device
        self.ucg = ucg
        self.logger = logger

        if type(self.text_encoder) is torch.nn.parallel.DistributedDataParallel:
            self.text_encoder = self.text_encoder.module
        if type(self.text_encoder_2) is torch.nn.parallel.DistributedDataParallel:
            self.text_encoder_2 = self.text_encoder_2.module

        self.transform_norm = transforms.Compose([
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def find_last_non_empty_element_index(self, lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != "":
                return i
        return -1

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {}
        image_file, parsing_mask, parsing_text, aa_text = self.store.get_image_and_parsing(item)

        p = np.random.choice([0, 1])
        transform_share = transforms.Compose([
            transforms.RandomHorizontalFlip(p),
            transforms.ToTensor()
        ])
        return_dict['pixel_values'] = self.transform_norm(transform_share(image_file))
        return_dict['parsing_mask'] = torch.stack([transform_share(p) for p in parsing_mask], dim=0)

        aa_token_len = len(self.tokenizer(aa_text).input_ids)
        while aa_token_len > 77:
            find_last_non_empty_element_i = self.find_last_non_empty_element_index(parsing_text)
            delete_len = len(", " + parsing_text[find_last_non_empty_element_i])
            aa_text = aa_text[:-1 * delete_len]
            parsing_text[find_last_non_empty_element_i] = ""

            aa_token_len = len(self.tokenizer(aa_text).input_ids)

        return_dict['parsing_text'] = parsing_text
        return_dict['aa_text'] = aa_text

        return return_dict

    def find_sublist_index(self, sublist, mainlist, subtext, maintext, logger):
        if len(sublist) == 0:
            return None
        index_list = []
        sublist_length = len(sublist)
        for i in range(len(mainlist)):
            if mainlist[i] == sublist[0]:
                if mainlist[i:i+sublist_length].tolist() == sublist:
                    index_list.append(i)
        if len(index_list) == 1:
            return (index_list[0], index_list[0] + len(sublist))
        else:
            if len(index_list) > 1:
                pass
                # logger.warning("find_sublist_index error: {} in {}".format(subtext, maintext))
            return None

    def collate_fn(self, examples):
        if random.random() > self.ucg:
            pixel_values = torch.stack([example['pixel_values'] for example in examples])
            pixel_values.to(memory_format=torch.contiguous_format).float()
            parsing_mask = torch.stack([example['parsing_mask'] for example in examples])
            parsing_mask.to(memory_format=torch.contiguous_format).float()
            
            # UPDATE: use latest version of diffusers.
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = ([self.text_encoder, self.text_encoder_2])
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompt = [example['aa_text'] for example in examples]
            # Get text embedding
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                text_inputs_mask = text_inputs.attention_mask

                text_encoder.cuda()
                prompt_embeds = text_encoder(
                    text_input_ids.to(self.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, 1, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
            
            parse_index = []
            parse_split = []
            for example_i, example in enumerate(examples):
                parse_index += [self.find_sublist_index(tokenizers[-1](text).input_ids[1:-1], text_input_ids[example_i], text, prompt[example_i], None) for text in example['parsing_text']]
                parse_split += [(0, len(example['parsing_text'])) if len(parse_split) == 0 else (parse_split[-1][1], parse_split[-1][1] + len(example['parsing_text']))]

            return {
                'pixel_values': pixel_values,
                'parsing_mask': parsing_mask,

                'input_ids': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,

                'parse_split': parse_split,
                'parse_index': parse_index,
                'prompt': prompt,
                'text_inputs_mask': text_inputs_mask,

                'original_size': pixel_values.shape[-2:],
                'use_parsing': not all(all(parsing_text_i == '' for parsing_text_i in example['parsing_text']) for example in examples),
            }
        else:
            pixel_values = torch.stack([example['pixel_values'] for example in examples])
            pixel_values.to(memory_format=torch.contiguous_format).float()
            
            # UPDATE: use latest version of diffusers.
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = ([self.text_encoder, self.text_encoder_2])
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompt = ["" for example in examples]
            # Get text embedding
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                text_inputs_mask = text_inputs.attention_mask

                text_encoder.cuda()
                prompt_embeds = text_encoder(
                    text_input_ids.to(self.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, 1, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
            
            return {
                'pixel_values': pixel_values,
                'input_ids': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,
                'prompt': prompt,
                'text_inputs_mask': text_inputs_mask,
                'original_size': pixel_values.shape[-2:],
                'use_parsing': False,
            }

