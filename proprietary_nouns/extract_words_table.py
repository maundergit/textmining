#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         extract_words_table.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-04-11 16:15:38>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
import unicodedata
import jaconv
import MeCab

import pandas as pd

from textmining_lib import normalize_text, get_table_by_mecab

VERSION = 1.0

# sudo update-alternatives --config mecab-dictionary
MECAB_DICT = "/var/lib/mecab/dic/ipadic-utf8"


def init():
    arg_parser = argparse.ArgumentParser(description="語句テーブルを生成する",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
example:

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--sort", dest="SORT", help="column name to sort", type=str, metavar='COLUMN', default="原形")
    arg_parser.add_argument("--user_dict", dest="USER_DICT", help="path of user dictionary", type=str, metavar='DCITIONARY', default=None)

    arg_parser.add_argument("--unicode_normalized_mode",
                            dest="UNORM",
                            help="mode of unicode normalization, default=NFKC",
                            choices=["NFD", "NFC", "NFCD", "NFKC"],
                            default="NFKC")

    arg_parser.add_argument("--with_neologdn", dest="WNLD", help="processing with neologdn", action="store_true", default=False)
    arg_parser.add_argument("--for_mecab", dest="FOR_MECAB", help="MECABで記号とする半角記号文字列,ex:'()[]{}'", type=str, default=None)
    arg_parser.add_argument("--without_kansuji", dest="WO_KANSUJI", help="漢数字の半角数字への変換を行わない", action="store_true", default=False)

    arg_parser.add_argument("--output",
                            dest="OUTPUT",
                            help="path of output file : default=stdout",
                            type=str,
                            metavar='FILE',
                            default=sys.stdout)

    arg_parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = arg_parser.parse_args()
    return args


def trim_sentence(sentence, with_symbol=False):
    sentence = re.sub(r"[\t]+", " ", sentence)
    sentence = unicodedata.normalize("NFKC", sentence)
    # sentence = re.sub(r"[\t\r\n]+", " ", sentence)
    # sentence = jaconv.h2z(sentence, kana=True, digit=False, ascii=False)
    # sentence = jaconv.z2h(sentence, kana=False, digit=True, ascii=True)
    if not with_symbol:
        sentence = re.sub(r"[!-/:-@[-`{-~]", "　", sentence)
    return sentence


if __name__ == "__main__":
    args = init()
    files = args.files
    output_csv = args.OUTPUT

    sort_key = args.SORT
    user_dict = args.USER_DICT

    wnld = args.WNLD
    unicode_normalize = args.UNORM
    for_mecab_s = args.FOR_MECAB
    kansuji = not args.WO_KANSUJI

    for_mecab = []
    if for_mecab_s is not None:
        for_mecab = [for_mecab_s, jaconv.h2z(for_mecab_s, ascii=True)]

    mecab_option = ""
    if user_dict is not None:
        mecab_option = f"{mecab_option} -u {user_dict}"

    output_df: pd.DataFrame = None
    for line in fileinput.input(files=files):
        line = line.rstrip()
        # line = trim_sentence(line)
        line = normalize_text(line, unicode_normalize=unicode_normalize, with_neologdn=wnld, for_mecab=for_mecab, kansuji=kansuji)
        output_df = get_table_by_mecab(output_df, line, mecab_option=mecab_option)

    output_df.sort_values(by=sort_key, axis=0, inplace=True)
    output_df.to_csv(output_csv, index=False)
