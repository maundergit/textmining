#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         csv_text_normalize.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-04-17 11:51:31>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
import csv
import pandas as pd

import jaconv
from textmining_lib import normalize_text

VERSION = 1.0


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
remark:
  '--unicode_normalized_mode' で指定される正規化を行い、漢数字を半角数字として変換する。

  mecabで記号の扱いで問題が出る場合は、'--for_mecab="()[]{}"'を指定して、記号を全角として強制設定する
  根本的な解決には'unk.def'を編集して辞書を再作成する必要がある。
  mecabで半角記号が名詞,サ変接続になるのを解決する ： nymemo https://nymemo.com/mecab/564/

  '--no_header'を使用するとCSVではない入力を処理することができる。

example:

  python csv_text_normalize.py --include_pattern='.+D' text_sample.csv
  python csv_text_normalize.py --include_columns=BzxyD text_sample.csv
  python csv_text_normalize.py --include_columns=HIJ text_sample.csv

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--unicode_normalized_mode",
                            dest="UNORM",
                            help="mode of unicode normalization, default=NFKC",
                            choices=["NFD", "NFC", "NFCD", "NFKC"],
                            default="NFKC")

    arg_parser.add_argument("--with_neologdn", dest="WNLD", help="processing with neologdn", action="store_true", default=False)
    arg_parser.add_argument("--for_mecab", dest="FOR_MECAB", help="MECABで記号とする半角記号文字列,ex:'()[]{}'", type=str, default=None)
    arg_parser.add_argument("--without_kansuji", dest="WO_KANSUJI", help="漢数字の半角数字への変換を行わない", action="store_true", default=False)

    arg_parser.add_argument("--no_header", dest="NO_HEADER", help="一行目からデータとして処理する", action="store_true", default=False)
    arg_parser.add_argument("--quote_all", dest="QUOTE_ALL", help="全カラムをQuoteする", action="store_true", default=False)

    arg_parser.add_argument("--include_columns",
                            dest="INC_COLUMNS",
                            help="names of colunmns to process, default=all",
                            type=str,
                            metavar='COLUMNS[,COLUMNS,[COLUMNS,...]]',
                            default=None)
    arg_parser.add_argument("--exclude_columns",
                            dest="EXC_COLUMNS",
                            help="names of colunmns to exclude",
                            type=str,
                            metavar='COLUMNS[,COLUMNS,[COLUMNS,...]]',
                            default=None)
    arg_parser.add_argument("--include_pattern",
                            dest="INC_PATTERN",
                            help="name pattern of colunmns to process",
                            type=str,
                            metavar='REGEX',
                            default=None)
    arg_parser.add_argument("--exclude_pattern",
                            dest="EXC_PATTERN",
                            help="name pattern of colunmns to exclude",
                            type=str,
                            metavar='REGEX',
                            default=None)

    arg_parser.add_argument("--output", dest="OUTPUT", help="path of output, default is stdout", type=str, metavar='STR', default=None)

    arg_parser.add_argument('csv_file', metavar='FILE', help='files to read, if empty, stdin is used')

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = init()
    csv_file = args.csv_file
    wnld = args.WNLD
    unicode_normalize = args.UNORM
    for_mecab_s = args.FOR_MECAB
    kansuji = not args.WO_KANSUJI

    no_header = args.NO_HEADER
    quote_all_mode = args.QUOTE_ALL

    include_columns_s = args.INC_COLUMNS
    exclude_columns_s = args.EXC_COLUMNS

    include_pattern = args.INC_PATTERN
    exclude_pattern = args.EXC_PATTERN

    csv_output = args.OUTPUT
    if csv_output is None:
        csv_output = sys.stdout
    if csv_file == "-":
        csv_file = sys.stdin

    include_columns = []
    if include_columns_s is not None:
        include_columns = re.split(r"\s*(?<!\\),\s*", include_columns_s)
    exclude_columns = []
    if exclude_columns_s is not None:
        exclude_columns = re.split(r"\s*(?<!\\),\s*", exclude_columns_s)

    if no_header:
        header = None
    else:
        header = 0

    if quote_all_mode:
        quote_mode = csv.QUOTE_ALL
    else:
        quote_mode = csv.QUOTE_MINIMAL

    csv_df = pd.read_csv(csv_file, header=header)

    columns = list(csv_df.columns)

    for_mecab = []
    if for_mecab_s is not None:
        for_mecab = [for_mecab_s, jaconv.h2z(for_mecab_s, ascii=True)]

    if include_pattern is not None and len(include_pattern) > 0:
        columns = [v for v in columns if re.search(include_pattern, v) is not None]
    if exclude_pattern is not None and len(exclude_pattern) > 0:
        columns = [v for v in columns if re.search(exclude_pattern, v) is None]

    if len(include_columns) > 0:
        if len(set(include_columns) - set(columns)) > 0:
            print(f"#warn:csv_text_normalize:{set(include_columns)-set(columns)} was not found, but continue.", file=sys.stderr)
        columns = list(set(include_columns) - (set(include_columns) - set(columns)))
    if len(exclude_columns) > 0:
        columns = list(set(columns) - set(exclude_columns))

    if len(columns) == 0:
        print("??error:csv_text_normalize:there are no columns.", file=sys.stderr)
        sys.exit(1)

    print(f"%inf:csv_text_normalize:{columns} will be processed.", file=sys.stderr)

    for col in columns:
        csv_df[col] = csv_df[col].apply(
            lambda x: normalize_text(x, unicode_normalize=unicode_normalize, with_neologdn=wnld, for_mecab=for_mecab, kansuji=kansuji))

    csv_df.to_csv(csv_output, index=False, quoting=quote_mode, header=header is not None)

    print(f"%inf:csv_text_normalize:results was written into {csv_output}", file=sys.stderr)
