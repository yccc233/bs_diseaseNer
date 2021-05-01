import os
import re
import codecs

#from bert import tokenization
from albert_zh import tokenization
from utils import convert_single_example


project_dir = os.path.dirname(os.path.abspath(__file__))

# tokenizer = tokenization.FullTokenizer(vocab_file='%s/chinese_L-12_H-768_A-12/vocab.txt'%project_dir,
#                                        do_lower_case=True)
tokenizer = tokenization.FullTokenizer(vocab_file='%s/albert_tiny/vocab.txt'%project_dir,
                                       do_lower_case=True)

from data_utils import create_dico, create_mapping, zero_digits


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            # assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]

    dico = create_dico(tags)
    dico['[SEP]'] = len(dico) + 1
    dico['[CLS]'] = len(dico) + 2

    tag_to_id, id_to_tag = create_mapping(dico)
    # logger.info("Found {} unique named entity tags".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, max_seq_length, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0].strip() for w in s]
        char_line = ' '.join(string)   # 使用空格把汉字拼起来
        text = tokenization.convert_to_unicode(char_line)

        if train:
            tags = [w[-1] for w in s]
        else:
            tags = ['O' for _ in string]

        labels = ' '.join(tags)     # 使用空格把标签拼起来
        labels = tokenization.convert_to_unicode(labels)

        ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                                   tag_to_id=tag_to_id,
                                                                   max_seq_length=max_seq_length,
                                                                   tokenizer=tokenizer,
                                                                   label_line=labels)
        data.append([string, segment_ids, ids, mask, label_ids])

    return data


def input_from_line(line, max_seq_length, tag_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    string = [w[0].strip() for w in line]
    # chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
    #         for w in string]
    char_line = ' '.join(string)  # 使用空格把汉字拼起来
    text = tokenization.convert_to_unicode(char_line)

    tags = ['O' for _ in string]

    labels = ' '.join(tags)  # 使用空格把标签拼起来
    labels = tokenization.convert_to_unicode(labels)

    ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
                                                               tag_to_id=tag_to_id,
                                                               max_seq_length=max_seq_length,
                                                               tokenizer=tokenizer,
                                                               label_line=labels)
    import numpy as np
    segment_ids = np.reshape(segment_ids,(1, max_seq_length))
    ids = np.reshape(ids, (1, max_seq_length))
    mask = np.reshape(mask, (1, max_seq_length))
    label_ids = np.reshape(label_ids, (1, max_seq_length))
    return [string, segment_ids, ids, mask, label_ids]
