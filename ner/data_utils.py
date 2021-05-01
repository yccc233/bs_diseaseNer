# encoding = utf8
import re
import math
import codecs
import random

import numpy as np
import jieba
#jieba.initialize()


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1

    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = []
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i*batch_size) : int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        '''
        把batch整理为一个[5, ]的数组
        :param batch:
        :return:
        '''
        strings = []
        segment_ids = []
        chars = []
        mask = []
        targets = []
        for string, seg_ids, char, msk, target in batch:
            strings.append(string)
            segment_ids.append(seg_ids)
            chars.append(char)
            mask.append(msk)
            targets.append(target)
        return [strings, segment_ids, chars, mask, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
