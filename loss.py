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



import torch
import torch.nn.functional as F

import ipdb


class HOLA_LOSS_Tools():
    def __init__(self, attn_map_size):
        self.attn_map_size = attn_map_size

    def flatten_dict(self, d, parent_key='', sep='-'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_map_loss(self, attention_maps, batch):
        ori_B, _, _, ori_H, ori_W = batch['parsing_mask'].shape
        parse_list_split = batch['parse_split']

        if not (attention_maps.shape[3] == 77 and (attention_maps.shape[2] == round(ori_H / self.attn_map_size) * round(ori_W / self.attn_map_size))):
            return None, None
        else:
            attn_map_size = int(self.attn_map_size)
            attention_maps = attention_maps.reshape(ori_B, attention_maps.shape[1], round(ori_H / attn_map_size), round(ori_W / attn_map_size), 77)
            ori_parsing_mask = batch['parsing_mask'][:, :, 0]
            parsing_mask = F.interpolate(ori_parsing_mask, size=(round(ori_H / attn_map_size), round(ori_W / attn_map_size)), mode='bilinear', align_corners=False)
            parsing_mask = parsing_mask[:, :, None, :, :]
            parse_index = batch['parse_index']

            attn_map_list = [[] for i in range(ori_B)]
            parsing_target_list = [[] for i in range(ori_B)]
            map_loss = [[] for i in range(ori_B)]
            for i, parse_split in enumerate(parse_list_split):
                for j, parse in enumerate(parse_index[parse_split[0]: parse_split[1]]):
                    if parse is not None:
                        maps = attention_maps[i, :, :, :, parse[0]:parse[1]]
                        mask = parsing_mask[i, j, :, :, :].unsqueeze(3)
                        attn_map_list[i] += [maps, maps.mean(-1, keepdim=True)]
                        parsing_target_list[i] += [mask.expand(-1, -1, -1, maps.shape[-1] + 1).to(maps.device, dtype=maps.dtype)]
            attn_maps = [torch.cat(ori_pair_i, dim=-1) if len(ori_pair_i) != 0 else [] for ori_pair_i in attn_map_list]
            parsing_targets = [torch.cat(pos_pair_i, dim=-1) if len(pos_pair_i) != 0 else [] for pos_pair_i in parsing_target_list]
            for i, (attn_map_i, parsing_target_i) in enumerate(zip(attn_maps, parsing_targets)):
                if len(attn_map_i) != 0:
                    map_loss[i] += [torch.mean((((attn_map_i - parsing_target_i.to(attn_map_i.device, dtype=attn_map_i.dtype)) ** 2)), dim=(1, 2, 3))]
            return map_loss, attn_map_size

    def merge_loss(self, other_out, train_batch_size, device=None):
        flatten_other_out = self.flatten_dict(other_out)
        _loss = [[] for i in range(train_batch_size)]
        for key in [key if "hola_loss" in key else [] for key in flatten_other_out]:
            if not isinstance(key, list):
                for batch_i, loss_i in enumerate(flatten_other_out[key]):
                    _loss[batch_i] += loss_i
        loss = torch.Tensor([0.]).to(device)[0]
        loss_cnt = 0
        for i in range(train_batch_size):
            if len(_loss[i]) != 0:
                loss += torch.cat(_loss[i], dim=0).mean()
                loss_cnt += 1
        return loss / loss_cnt if loss_cnt != 0 else loss
