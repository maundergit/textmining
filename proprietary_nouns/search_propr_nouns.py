#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         search_propr_nouns.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-04-08 18:51:42>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
# import jaconv
import MeCab
import unicodedata

# from pykakasi import kakasi
import pandas as pd

from typing import List

from textmining_lib import get_mecab_parser

VERSION = 1.0

# sudo update-alternatives --config mecab-dictionary
# MECAB_DICT = "/var/lib/mecab/dic/ipadic-utf8"

IGNORE_POSIDS_LIST = ["*", "読点", "句点", "係助詞", "格助詞", "自立", "非自立", "接続助詞", "括弧開", "括弧閉", "副詞可能", "連体化", "空白"]


def init():
    arg_parser = argparse.ArgumentParser(description="名詞が連続するものを固有名詞の候補として抽出する。",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent(f'''
remark:
  品詞が名詞のものが連続するものを固有名詞候補として抽出
  ただし、品詞細分類が以下に含まれるものは、連続語句として扱わない。
    {IGNORE_POSIDS_LIST}
  数字で始まるもの(例：1装置)を出力に含めたい場合は、'--start_with_number'を指定する。
  数字を含むもの(例：〇〇１装置)を出力に含めたい場合は、'--with_number'を指定する。
  記号を含むもの(例：装置[A])を出力に含めたい場合は、'--with_symbol'を指定する。

example:

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--output",
                            dest="OUTPUT",
                            help="path of output file : default=stdout",
                            type=str,
                            metavar='FILE',
                            default=sys.stdout)

    arg_parser.add_argument("--with_symbol", dest="W_SYMBOL", help="treating symbol as word", action="store_true", default=False)
    arg_parser.add_argument("--with_number", dest="W_NUMBER", help="treating numbr as word", action="store_true", default=False)
    arg_parser.add_argument("--start_with_number",
                            dest="START_W_NUM",
                            help="treating word starting with number",
                            action="store_true",
                            default=False)

    arg_parser.add_argument("--user_dict", dest="USER_DICT", help="path of user dictionary", type=str, metavar='DCITIONARY', default=None)

    arg_parser.add_argument('csv_files', metavar='FILE', nargs="*", help='files to read, if empty, stdin is used')
    args = arg_parser.parse_args()
    return args


def trim_sentence(sentence, with_symbol=False):
    sentence = re.sub(r"[\t]+", " ", sentence)
    sentence = unicodedata.normalize("NFKC", sentence)
    # sentence = jaconv.h2z(sentence, kana=True, digit=False, ascii=False)
    # sentence = jaconv.z2h(sentence, kana=False, digit=True, ascii=True)
    if not with_symbol:
        sentence = re.sub(r"[!-/:-@[-`{-~]", "　", sentence)

    return sentence


def search_proprietary_nouns_by_mecab(sentence, start_with_number=True, with_number=True, mecab_option=""):
    """名詞が連続するものを固有名詞候補としてピックアップする

    :param sentence: 
    :param start_with_number: 
    :param with_number: 
    :param mecab_option: 
    :returns: 
    :rtype: 

    """
    if start_with_number:
        with_number = True
    ignore_posids = IGNORE_POSIDS_LIST
    if not with_number:
        ignore_posids.extend(["数", "数接続"])
    ignore_symbols = ["(", ")", "[", "]"]
    m = get_mecab_parser("", mecab_option=mecab_option)
    # m = MeCab.Tagger(f"-d {MECAB_DICT} {mecab_option}")
    node = m.parseToNode(sentence)
    proprietry_nouns = []
    proprietry_noun_buffer = []
    prev_posid = ""
    while node:
        word = node.surface
        if len(word) == 0:
            node = node.next
            continue
        cvs = re.split(r",", node.feature)
        posid_0 = cvs[0]
        posid_1 = cvs[1]
        if len(cvs) < 8:
            yomi = word
        else:
            yomi = cvs[7]

        if (prev_posid != posid_0
                and prev_posid == "名詞") or (not start_with_number
                                            and word[0].isdigit()) or posid_1 in ignore_posids or word in ignore_symbols:
            if len(proprietry_noun_buffer) > 1 and proprietry_noun_buffer[0]["posid"] not in ["接尾"]:
                w = "".join([v["word"] for v in proprietry_noun_buffer])
                ym = "".join([v["yomi"] for v in proprietry_noun_buffer])
                if w not in [v["word"] for v in proprietry_nouns]:
                    proprietry_nouns.append({"word": w, "posids": [v["posid"] for v in proprietry_noun_buffer], "yomi": ym})
            proprietry_noun_buffer = []
        else:
            proprietry_noun_buffer.append({"word": word, "posid": posid_1, "yomi": yomi})
        prev_posid = posid_0

        node = node.next

    if len(proprietry_noun_buffer) > 1 and proprietry_noun_buffer[0]["posid"] not in ["接尾"]:
        w = "".join([v["word"] for v in proprietry_noun_buffer])
        ym = "".join([v["yomi"] for v in proprietry_noun_buffer])
        if w not in [v["word"] for v in proprietry_nouns]:
            proprietry_nouns.append({"word": w, "posids": [v["posid"] for v in proprietry_noun_buffer], "yomi": ym})

    return proprietry_nouns


def search_in_sentence(sentence, proprietry_nouns, add_length=5, limit=20):
    """固有名詞候補リストの各語句について、文章中の出現箇所を切り出しリスト化する

    :param sentence: 
    :param proprietry_nouns: 
    :param add_length: 
    :returns: 
    :rtype: 

    """

    results = []
    for pn in proprietry_nouns:
        w = pn["word"]
        ym = pn["yomi"]
        ps = pn["posids"]
        ss = []
        for m in re.finditer(re.escape(w), sentence):
            idx = m.start()
            l_w = len(w)
            if idx - add_length < 0:
                isx = 0
            else:
                isx = idx - add_length
            iex = isx + l_w + add_length * 2
            ss.append(sentence[isx:iex])
            if len(ss) > limit:
                print(f"#warn:search_proper_nouns:number of sentences for '{w}' is more than {limit}", file=sys.stderr)
                break
        results.append({"word": w, "sentences": ss, "posids": ps, "yomi": ym})

    return results


if __name__ == "__main__":
    args = init()
    csv_files = args.csv_files
    output_csv = args.OUTPUT

    w_symbol = args.W_SYMBOL
    w_number = args.W_NUMBER
    start_w_num = args.START_W_NUM

    user_dict = args.USER_DICT

    mecab_option = ""
    if user_dict is not None:
        mecab_option = f"{mecab_option} -u {user_dict}"

    lines = []
    proprietry_nouns = []
    words_list: List[str] = []
    count = 0
    for line in fileinput.input(files=csv_files):
        count += 1
        if count % 100 == 1:
            print(f"-- processing {count}", file=sys.stderr)
        line = line.rstrip()
        line = trim_sentence(line, with_symbol=w_symbol)
        p_ns = search_proprietary_nouns_by_mecab(line, start_with_number=start_w_num, with_number=w_number, mecab_option=mecab_option)
        for ns in p_ns:
            if ns["word"] not in words_list:
                proprietry_nouns.append(ns)
                words_list.append(ns["word"])
        lines.append(line)

    sentence = " ".join(lines)

    nouns_samples = search_in_sentence(sentence, proprietry_nouns)

    output_cols = ["Check", "語句", "読み候補", "出現数", "品詞構成", "用例"]
    output_df = pd.DataFrame(columns=output_cols)
    # kakasi = kakasi()
    # kakasi.setMode('J', 'H')
    # kakasi_conv = kakasi.getConverter()
    for ns in nouns_samples:
        word = ns["word"]
        yomi = ns["yomi"]
        posids = "|".join(ns["posids"])
        sts = "|".join(ns["sentences"])
        # print(f"{word} , \"{posids}\" , \"{sts}\"")
        output_df = output_df.append({
            "語句": word,
            "読み候補": yomi,
            "出現数": len(ns["sentences"]),
            "品詞構成": posids,
            "用例": sts
        },
                                     ignore_index=True)

    output_df.to_csv(output_csv, index=False)
