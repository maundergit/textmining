#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         make_dict.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-04-10 09:40:18>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
# import pykakasi
import jaconv
import MeCab

import numpy as np
import pandas as pd

from textmining_lib import get_yomi

VERSION = 1.0

# sudo update-alternatives --config mecab-dictionary
MECAB_DICT = "/var/lib/mecab/dic/ipadic-utf8"


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
remark:
  語句と読みのカラムを持つCSVからMeCab用辞書を生成する。
  読みが未指定の場合は、内部で読みを生成する。
  品詞は名詞で固定指定となっており、名詞細分類1はデフォルトは固有名詞である。'--pos1'により名詞細分類列が指定された場合はその値が用いられる。

example:

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))
    arg_parser.add_argument("--pos1",
                            dest="POS1",
                            help="column name for part of speech subdivision",
                            type=str,
                            metavar="COLUMN",
                            default=None)

    arg_parser.add_argument("--user_dict", dest="USER_DICT", help="path of user dictionary", type=str, metavar='DCITIONARY', default=None)
    arg_parser.add_argument("--header", dest="HEADER", help="with column header", action="store_true", default=False)

    arg_parser.add_argument('csv_file', metavar='FILE', help='files to read, if empty, stdin is used', type=argparse.FileType('r'))
    arg_parser.add_argument('word_column', metavar='COLUMN', help="column name for word", type=str)
    arg_parser.add_argument('yomi_column', metavar='COLUMN', help="column name for yomigana", type=str)
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = init()
    csv_file = args.csv_file

    word_column = args.word_column
    yomi_column = args.yomi_column
    pos1_column = args.POS1

    f_header = args.HEADER
    user_dict = args.USER_DICT

    mecab_option = ""
    if user_dict is not None:
        mecab_option = f"{mecab_option} -u {user_dict}"

    csv_df = pd.read_csv(csv_file)

    columns = [word_column, yomi_column]

    if pos1_column is not None:
        columns.append(pos1_column)

    for col in columns:
        if col not in csv_df.columns:
            mes = f"??error:make_dict:{col} was not found in {csv_file}"
            print(mes, file=sys.stderr)
            sys.exit(1)
            # raise ValueError(mes)

    output_columns = ["表層形", "左文脈ID", "右文脈ID", "コスト", "品詞", "品詞細分類1", "品詞細分類2", "品詞細分類3", "活用型", "活用形", "原形", "読み", "発音"]

    output_df = pd.DataFrame(columns=output_columns)

    output_df["表層形"] = csv_df[word_column]
    output_df["左文脈ID"] = "1288"
    output_df["右文脈ID"] = "1288"
    output_df["コスト"] = "10"
    output_df["品詞"] = "名詞"
    if pos1_column is not None:
        output_df["品詞細分類1"] = csv_df[pos1_column]
    else:
        output_df["品詞細分類1"] = "固有名詞"
    output_df["品詞細分類2"] = "一般"
    output_df["品詞細分類3"] = "*"
    output_df["活用型"] = "*"
    output_df["活用形"] = "*"
    output_df["原形"] = csv_df[word_column]
    output_df["読み"] = csv_df[yomi_column]
    output_df["発音"] = csv_df[yomi_column]

    for idx, ds in output_df[csv_df[yomi_column].isnull()].iterrows():
        word = ds["表層形"]
        if ds["読み"] is np.nan:
            yomi = "".join(get_yomi(word, mecab_option=mecab_option))
            ds["読み"] = yomi
            ds["発音"] = yomi
            output_df.iloc[idx] = ds

    output_df.to_csv(sys.stdout, index=False, header=f_header)
