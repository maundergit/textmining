#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         csv_text_solr_search.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-07-30 18:42:37>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
import pandas as pd
import json
import pprint

import datetime

from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Type, NoReturn

from textmining_lib import removeSentences, replaceSentences
from csv_text_tfidf import get_words_0, read_extend_word
from csv_text_solr import communicateSlor, trim_sentence

VERSION = 1.0

# EXTENDED_PREFIX = "_ext"

F_indent_text = lambda x: textwrap.fill(x, width=130, initial_indent="\t", subsequent_indent="\t")


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
remark:
  using '--extend_word_file', bow of sentence is extended with additional words.
  this option may be usefull when the configuration of devices is important.
  format of 'extend word file' there is 'word word1,word2' in a line.

  for searching, see about query paramters:
    The Standard Query Parser | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/the-standard-query-parser.html
  Query format (only for '--search_detail*'):
    boosting: word^float
    ranging : [ V1 TO V2 ], { V1 TO V2 ]

  only for '-search_ex', followings have effects:
    '--user_dictionary', '--minmum_length_of_word', '--remove_words', '--remove_pattern', '--replace_pattern_file',and  '--extend_word_file'
    

  NOTE: in search keywords, single cuqitation and double quotation have diffrent meanings.
        the string enclosed in double quotes is used for the search as a full match.

example:
  cat <<EOF > rep_pat.txt
  s/《[^》]*》//
  s/(?i)GHI/KLM/
  EOF
  cat <<EOF > tfidf_extend_word.txt
  # regexp word1,word2,...
  書生	学生,学校
  原$	野原
  EOF
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_field=content --search "主人" "学生"
  csv_text_solr_search.py --core wagahaiwa_nekodearu --extend_word_file=tfidf_extend_word.txt --search_field=content --search "主人" "学生" \\
                   --search_ex "よく主人の所へ遊びに来る馬鹿野郎"
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail="date_local:\"01-12\""
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail="content:主人" 

  cat <<EOF > search_detail.query
  content:"主人が熱心"
  content:学生
  EOF
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail_file=search_detail.query
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail='content:"猫"^0.1 content:"吾輩"^2' --search_limit=10
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail='date:[2021-12-31T00:00:00Z TO *]' --search_limit=10
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail='date:[NOW-1MONTH/DAY TO * ]' --search_limit=10
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail="content:書生 content:\+騒々しい"
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_limit=10 --search_detail='content:書生 content:見る'
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_limit=10 --search_detail='content:書生 !content:見る'

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--host",
                            dest="HOST",
                            help="ip address of host, default=127.0.0.1",
                            type=str,
                            metavar='IP_ADDRESS',
                            default="127.0.0.1")
    arg_parser.add_argument("--port", dest="PORT", help="port number, default=8983", type=int, metavar='PORT', default=8983)
    arg_parser.add_argument("--core", dest="CORE", help="name of core of solr", type=str, metavar='NAME', required=True)

    arg_parser.add_argument("--extend_columns",
                            dest="EX_COLUMN",
                            help="decompose sentence and extend words",
                            type=str,
                            metavar="FIELD[,FIELD..]",
                            default=None)

    arg_parser.add_argument("--user_dictionary", dest="UDICT", help="path of user dictionary", type=str, metavar='FILE', default="")
    arg_parser.add_argument("--minmum_length_of_word",
                            dest="MLENGTH",
                            help="minimum length of word, default=2",
                            type=int,
                            metavar='INT',
                            default=2)
    arg_parser.add_argument("--remove_words",
                            dest="RWORDS",
                            help="list of words to remove, default='。,、,？,.,\,,?'",
                            type=list,
                            metavar='WORDS[,WORD,...]',
                            default="。,、,？,.,\,,?")
    arg_parser.add_argument("--remove_pattern",
                            dest="RPATTERN",
                            help="regex pattern to remove before analyzing or file",
                            type=str,
                            metavar='REGEX_OR_FILE',
                            default="")
    arg_parser.add_argument("--replace_pattern_file",
                            dest="REP_FILE",
                            help="path of file that has regex pattern to replace before analyzing",
                            type=str,
                            metavar='FILE',
                            default="")
    arg_parser.add_argument("--extend_word_file",
                            dest="EXT_FILE",
                            help="path of file that has regex pattern and word to add at deriving words",
                            type=str,
                            metavar='FILE',
                            default=None)

    arg_parser.add_argument("--search", dest="SEARCH", help="sentence(s) to search", type=str, metavar='STR', nargs="+", default=None)
    arg_parser.add_argument("--search_field", dest="SEARCH_FIELD", help="field to search", type=str, metavar='FIELD', default=None)
    arg_parser.add_argument(
        "--search_ex",
        dest="SEARCH_EX",
        help=
        "sentence(s) to search with exntending by '--user_ditctionary','--remove_words','--remove_pattern','--replace_pattern_File','--extended_word_file'",
        type=str,
        metavar='STR',
        nargs="+",
        default=None)
    # arg_parser.add_argument("--search_morelikethis", dest="SEARCH_MLT", help="enable morelkethis", action="store_true", default=False)
    arg_parser.add_argument("--search_limit",
                            dest="SEARCH_LIMIT",
                            help="limit of search results, default=50",
                            type=int,
                            metavar='INT',
                            default=50)
    arg_parser.add_argument("--search_operator",
                            dest="SEARCH_OP",
                            help="operator to search, default='OR'",
                            choices=["OR", "AND"],
                            default="OR")
    arg_parser.add_argument("--search_detail",
                            dest="SEARCH_DETAIL",
                            help="detail search for each field",
                            type=str,
                            metavar='FIELD:KEYWORD[,FIELD:KEYWORD..]',
                            default=None)
    arg_parser.add_argument("--search_detail_file",
                            dest="SEARCH_DETAIL_FILE",
                            help="path of file that have detail search queriesfor each field",
                            type=str,
                            metavar="FILE",
                            default=None)

    arg_parser.add_argument("--search_output",
                            dest="SEARCH_OUT",
                            help="path of csv file to store result, default=stdout",
                            type=str,
                            metavar="CSV_FILE",
                            default=sys.stdout)

    arg_parser.add_argument('csv_files', metavar='CSV_FILE', nargs='*', help='csv files to read. if empty, stdin is used')

    args = arg_parser.parse_args()
    return args


def search_sentences_on_solr(com_solr: communicateSlor,
                             field: str,
                             search_sentences: List[str],
                             search_ex_sentences: List[str],
                             search_detail: List[str],
                             remove_pattern_c=None,
                             replace_pattern_c=None,
                             min_length=2,
                             user_dict=None,
                             remove_words=None,
                             extend_words=None,
                             operator=None,
                             morelikethis=False,
                             limit_of_hits=50) -> Tuple[pd.DataFrame, int, float]:
    """TODO describe function

    :param com_solr: communicateSlorクラスのインスタンス
    :param field: 検索対象のSolrのフィールド
    :param search_sentences: 検索文のリスト
    :param search_ex_sentences: 分かち書き処理を行い利用する検索文のリスト
    :param search_detail: SolrのQuery形式の検索クエリー
    :param remove_pattern_c: removeSentencesクラスのインスタンス
    :param replace_pattern_c: replaceSentencesクラスのインスタンス
    :param min_length: 語句として残す最小の文字列長さ
    :param user_dict: Mecabのユーザ辞書ファイルのパス
    :param remove_words: 分かち書き後に除去する語句のリスト
    :param extend_words: 分かち書き後の各語句に対する追加国辞書
    :param operator: 検索文の論理結合式(OR 又は AND)
    :returns: DataFrame, 検索結果のレコード数, 最大スコア

    """
    q_str = []  # 対象フィールドに対する検索語句リスト
    if search_sentences is not None:
        q_str.extend(search_sentences)

    if search_ex_sentences is not None:
        for sts in search_ex_sentences:
            trimed_sentence = trim_sentence(sts,
                                            remove_pattern_c=remove_pattern_c,
                                            replace_pattern_c=replace_pattern_c,
                                            min_length=min_length,
                                            user_dict=user_dict,
                                            remove_words=remove_words,
                                            extend_words=extend_words)
            if isinstance(trimed_sentence, list):
                q_str.extend(trimed_sentence)
            else:
                q_str.append(trimed_sentence)

    if len(q_str) == 0 and len(search_detail) == 0:
        print(f"#warn:csv_text_solr_search:search:no query", file=sys.stderr)
        df = pd.DataFrame()
        nhits = 0
        max_score = 0
    else:
        q_str = list(set(q_str))
        if search_detail is not None:
            q_mes = F_indent_text(f"{q_str},{search_detail}")
        else:
            q_mes = F_indent_text(f"{q_str}")

        print(f"%inf:csv_text_solr_search:search:query strings:\n{q_mes}", file=sys.stderr)
        res = com_solr.search(field, q_str, detail=search_detail, operator=operator, morelikethis=morelikethis, limit=limit_of_hits)

        result = json.loads(res.data)
        if morelikethis:
            print(f"#warn:csv_text_solr_search:'--search_morelikethis' is not implemented.", file=sys.stderr)
            pass
            # docs = []
            # max_score = []
            # nhits = []
            # for _, result in result["moreLikeThis"].items():
            #     docs.extend(result["docs"])
            #     max_score.append(result["maxScore"])
            #     nhits.append(result["numFound"])
        else:
            docs = result["response"]["docs"]
            max_score = result["response"]["maxScore"]
            nhits = result["response"]["numFound"]
        # start_no = result["response"]["start"]

        df = pd.DataFrame(docs)

    return df, nhits, max_score


def read_search_detail_from_file(file_path: str) -> List[str]:
    """read query from file

    :param file_path: 
    :returns: 

    """
    dts = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            dts.append(line)

    return dts


if __name__ == "__main__":
    args = init()

    core_name = args.CORE
    host = args.HOST
    port = args.PORT

    user_dict = args.UDICT
    remove_pattern = args.RPATTERN
    replace_pattern_file = args.REP_FILE
    extend_word_file = args.EXT_FILE
    extend_words = read_extend_word(extend_word_file)
    min_length = args.MLENGTH
    remove_words = args.RWORDS

    csolr = communicateSlor(host, port, core_name)
    remove_pattern_c = removeSentences(remove_pattern) if len(remove_pattern) else None
    replace_pattern_c = replaceSentences(replace_pattern_file) if len(replace_pattern_file) else None

    search_sentences = args.SEARCH
    search_field = args.SEARCH_FIELD
    search_ex_sentences = args.SEARCH_EX
    search_limit_of_hits = args.SEARCH_LIMIT
    search_detail_s = args.SEARCH_DETAIL
    search_detail_file = args.SEARCH_DETAIL_FILE
    # search_mlt = args.SEARCH_MLT
    search_mlt = False
    search_operator = args.SEARCH_OP
    search_output = args.SEARCH_OUT

    # ==== Search mode ====
    search_detail = []
    if search_detail_file is not None:
        search_detail = read_search_detail_from_file(search_detail_file)
        if search_detail_s is not None:
            print(f"#warn:csv_text_solr_search:'--serach_detail' was defined, but ignored.", file=sys.stderr)
    elif search_detail_s is not None:
        search_detail = re.split(r"\s*(?<!\\),\s*", search_detail_s)
    if search_field is not None and search_sentences is None and search_ex_sentences is None:
        print(f"??error:csv_text_solr_search:'--search_field' was defined, '--search' or '--search_ex' should be defined.",
              file=sys.stderr)
        sys.exit(1)
    if search_sentences is not None or search_ex_sentences is not None or search_detail is not None:
        if search_detail is None and search_field is None:
            print(f"??error:csv_text_solr_search:'--search_field' was required.", file=sys.stderr)
            sys.exit(1)
        print("$inf:csv_text_solr_search:search mode", file=sys.stderr)
        search_df, nhits, max_score = search_sentences_on_solr(csolr,
                                                               search_field,
                                                               search_sentences,
                                                               search_ex_sentences,
                                                               search_detail,
                                                               remove_pattern_c=remove_pattern_c,
                                                               replace_pattern_c=replace_pattern_c,
                                                               min_length=min_length,
                                                               user_dict=user_dict,
                                                               remove_words=remove_words,
                                                               extend_words=extend_words,
                                                               operator=search_operator,
                                                               morelikethis=search_mlt,
                                                               limit_of_hits=search_limit_of_hits)
        # -- output result of searching
        search_df.to_csv(search_output, index=False)
        nrec = min(search_limit_of_hits, nhits)
        print(
            f"\n%inf:csv_text_solr_search:search: (number of records in result={nrec}) {'less than' if nrec < nhits else 'equal to'} (number of all searched records={nhits})",
            file=sys.stderr)
        if isinstance(search_output, str):
            print(f"                          result was stored into '{search_output}'", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"??error:csv_text_solr_search: there is no options, '--search_*' must be given", file=sys.stderr)
        sys.exit(1)
    # ==== end of search mode ====
