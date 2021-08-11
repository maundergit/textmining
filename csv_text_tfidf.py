#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         csv_text_tfidf.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-07-10 14:21:16>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

from pathlib import Path

import re
import json

from collections import Counter

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from textmining_lib import get_words, check_hiragana, check_ascii_symbol, check_zenkaku_symbol_with_japanese, removeSentences, replaceSentences, get_count_of_word

from typing import Union, List, Dict, Callable, Any, Type, Tuple, Optional
import typing

VERSION = 1.0
DEBUG = False
PATH_OF_USERDICT = ""
EXTEND_WORD_LIST: Union[Dict[str, List[str]], Any] = None
MINIMUM_LENGTH_OF_WORD = 2
REMOVE_WORDS = ["。", "、", "？", ".", ",", "?"]


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
remark:
  when using '--model_file', csv file and column as arguments must be given but those are not used and may be dummy.

  using '--extend_word_file', bow of sentence is extended with additional words.
  this option may be usefull when the configuration of devices is important.
  format of 'extend word file' there is 'word word1,word2' in a line.

  using '--status', simple statistics are printed. 
  you may remove words that are too frequent by using '--remove_pattern' or '--replace_pattern_file'.

  learning steps:
    1) make model by '--only_learn'
    2) estimate sentence by '--model_file'
    3) make new sequential learned model, by '--sequential_learn'

example:
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py - content
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot - content
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0.25 - content > test.dot

  THR=0.4
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0.3  --use_tf - content |\\
      perl -ne "if(/label=\"([.\d]+)\"/ && \$1>${THR}){s/\]/ color=\"red\"]/} print;" > test.dot
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 --remove_pattern='《[^》]*》' - content > test.dot

  cat <<EOF > rep_pat.txt
  s/《[^》]*》//
  s/(?i)GHI/KLM/
  EOF
  head -40 wagahaiwa_nekodearu.csv |\\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 --check_frequency=0.1 --replace_pattern_file=rep_pat.txt --debug - content> test.dot

  cat <<EOF > tfidf_extend_word.txt
  # regexp word1,word2,...
  書生	学生,学校
  原$	野原
  EOF
  head -40 wagahaiwa_nekodearu.csv |\\
       csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \\
                         --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt - content> test.dot

  # using learned model
  csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \\
                    --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --only_learn=tfidf.db wagahaiwa_nekodearu.csv content
  echo 吾輩《わがはい》は猫である。名前はまだ無い。 |\\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \\
                        --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --model_file=tfidf.db wagahaiwa_nekodearu.csv content

  # sequential learning
  echo -e "date,content\\n2021-07-22,引き止めて、一ヶ月経って、裏の書斎にこもっている。" > new_data.csv
  csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \\
                    --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --sequential_learn=tfidf.db,tfidf2.db new_data.csv content
  echo -e "引き止めて、一ヶ月経って、裏の書斎にこもっている。" |\\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \\
                        --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --model_file=tfidf2.db wagahaiwa_nekodearu.csv content
'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--index",
                            dest="INDEX",
                            help="column name for index, default=row number",
                            type=str,
                            metavar='COLUMN',
                            default=None)
    arg_parser.add_argument("--additional_columns",
                            dest="ACOLUMNS",
                            help="list of column names for additional columns into simlarity table, default=None",
                            type=str,
                            metavar='COLUMN[,COLUMN...]',
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

    arg_parser.add_argument("--output_mode",
                            dest="MODE",
                            help="format of output, default=simple",
                            type=str,
                            choices=["simple", "json", "dot"],
                            default="simple")
    arg_parser.add_argument("--only_learn",
                            dest="VECS_FILE_0",
                            help="path of file to store model",
                            type=str,
                            metavar='FILE',
                            default=None)
    arg_parser.add_argument("--model_file", dest="VECS_FILE_1", help="path of file to load model", type=str, metavar='FILE', default=None)
    arg_parser.add_argument("--sequential_learn",
                            dest="SEQ_MODE",
                            help="path of file that has result of learned model and new one",
                            type=str,
                            metavar='IN_FILE,OUT_FILE',
                            default=None)

    arg_parser.add_argument("--dot_cut_off",
                            dest="CUT_OFF",
                            help="threshold for cutting off, only available with '--output_mode=dot'",
                            type=float,
                            metavar="FLOAT",
                            default=None)
    arg_parser.add_argument("--check_frequency",
                            dest="CFLIMIT",
                            help="ratio for checking frequency of words, only available with '--debug', default=0.2",
                            type=float,
                            metavar="FLOAT",
                            default=0.2)

    arg_parser.add_argument("--use_tf", dest="USE_TF", help="use term frequency", action="store_true", default=False)
    arg_parser.add_argument("--use_idf", dest="USE_IDF", help="use inverse document frequency", action="store_true", default=False)

    arg_parser.add_argument("--status", dest="STATUS", help="print status", action="store_true", default=False)
    arg_parser.add_argument("--debug", dest="DEBUG", help="debug mode", action="store_true", default=False)

    arg_parser.add_argument('csv_file', metavar='FILE', help='csv file to read, if empty, stdin is used')
    arg_parser.add_argument('column', metavar='COLUMN', help="column nameto process")

    args = arg_parser.parse_args()
    return args


# scikit-learnのCountVectorizerやTfidfVectorizerで追加学習させる | ITに頼って生きていく https://boomin.yokohama/archives/1468
class SeqVectorizer(TfidfVectorizer):
    """追加学習機能対応のTfidfVectotizer

    """
    def __init__(self, analyzer="word", binary=True, use_idf=False, encoding="utf-8", strip_accents="unicode"):
        super().__init__(analyzer=analyzer, binary=binary, use_idf=use_idf, encoding=encoding, strip_accents=strip_accents)
        self.__seq_analyzer = analyzer
        self.__seq_binary = binary
        self.__seq_use_idf = use_idf
        self.__seq_encoding = encoding
        self.__seq_strip_accents = strip_accents

    def fit(self, X):
        result = super().fit(X)
        self.n_docs = len(X)

    def sequential_fit(self, X):
        max_idx = max(self.vocabulary_.values())

        intervec = TfidfVectorizer(analyzer=self.__seq_analyzer,
                                   binary=self.__seq_binary,
                                   use_idf=self.__seq_use_idf,
                                   encoding=self.__seq_encoding,
                                   strip_accents=self.__seq_strip_accents)

        for a in X:
            #update vocabulary_
            if self.lowercase: a = a.lower()
            intervec.fit([a])
            tokens = intervec.get_feature_names()
            for w in tokens:
                if w not in self.vocabulary_:
                    max_idx += 1
                    self.vocabulary_[w] = max_idx

            # update idf_
            if self.__seq_use_idf:
                df = (self.n_docs + self.smooth_idf) / np.exp(self.idf_ - 1) - self.smooth_idf
                self.n_docs += 1
                df.resize(len(self.vocabulary_), refcheck=False)
                for w in tokens:
                    df[self.vocabulary_[w]] += 1
                idf = np.log((self.n_docs + self.smooth_idf) / (df + self.smooth_idf)) + 1
                self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))


def check_frequency_words(sentences: List[str], limit_ratio: float = 0.2) -> typing.Counter[Any]:
    """語句の出現頻度を多い順に印字する

    :param sentences: 文字列のリスト
    :param limit_ratio: 印字範囲、文字列リストの長さに対する比率で指定する
    :returns: 語句の頻度を格納したcollection.Counter
    :rtype: collection.Counter

    """
    sts = " ".join(sentences)

    nlimit = int(len(sentences) * limit_ratio + 0.5)
    words = get_words_0(sts, debug=False)
    wc = Counter(words)
    # wc = get_count_of_word(sts, mecab_option="", path_of_userdict=PATH_OF_USERDICT, remove_words=["。", "、", "？", ".", ",", "?"])
    # wc_mc = wc.most_common(most_common)

    result_str = ""
    for k in sorted(wc, key=wc.get, reverse=True):
        v = wc[k]
        if v >= nlimit:
            r = v / len(sentences) * 100
            result_str += f"\t{k}:{v} ({r:6.2f}%)\n"

    if len(result_str) > 0:
        print(f"%inf:csv_text_tfidf:check_frequency_words:more than limit:ratio={limit_ratio}, count={nlimit}", file=sys.stderr)
        print(result_str, file=sys.stderr)
    else:
        print(f"%inf:csv_text_tfidf:check_frequency_words:no words whose count are more than limit:ratio={limit_ratio}, count={nlimit}",
              file=sys.stderr)

    return wc


def get_words_0(sentence: str,
                min_length: int = None,
                debug: bool = True,
                user_dict: str = None,
                remove_words: List[str] = None,
                extend_words: Dict[str, List[str]] = None) -> Union[List[str], str]:
    """文字列からわかち書きで語句リストを生成する

    :param sentence: 文字列
    :param min_length: 語句として選定する文字列下限値
    :param debug: 
    :returns: 語句リスト
    :rtype: 

    """
    if user_dict is not None:
        udict = user_dict
    else:
        udict = PATH_OF_USERDICT
    if remove_words is not None:
        rwords = remove_words
    else:
        rwords = REMOVE_WORDS
    if extend_words is not None:
        ewords = extend_words
    else:
        ewords = EXTEND_WORD_LIST

    # words = get_words(sentence, mecab_option="", path_of_userdict=PATH_OF_USERDICT, remove_words=["。", "、", "？", ".", ",", "?"])
    words = get_words(sentence, mecab_option="", path_of_userdict=udict, remove_words=rwords)

    if min_length is None:
        min_length = MINIMUM_LENGTH_OF_WORD

    ws = []
    for w in words:
        p1 = check_hiragana(w)
        p2 = check_ascii_symbol(w)
        p3 = check_zenkaku_symbol_with_japanese(w)
        if len(w) >= min_length and not p1 and not p2 and not p3:
            ws.append(w)
            if extend_words is not None:
                for p in ewords:
                    if re.search(p, w) is not None:
                        ws.extend(ewords[p])

    if len(ws) == 0:
        # ws = [""]
        ws = sentence
    if debug and DEBUG:
        print(f"%inf:csv_text_tfidf:get_words_0:list of words\n{sentence}\n =>{words}=>{ws}", file=sys.stderr)

    return ws


# Pythonで文章の類似度を計算する方法〜TF-IDFとcos類似度〜 | データサイエンス情報局 https://analysis-navi.com/?p=688
def get_words_vecs(sentences: List[str], use_tf: bool = False, use_idf: bool = False) -> Any:
    """

    :param sentences: 
    :param use_tf: 
    :param use_idf: 
    :returns: 
    :rtype: 

    """
    # 【sklearn】TfidfVectorizerの使い方を丁寧に - gotutiyan’s blog https://gotutiyan.hatenablog.com/entry/2020/09/10/181919
    # vectorizer = TfidfVectorizer(analyzer=get_words_0, binary=not use_tf, use_idf=use_idf, encoding='utf-8', strip_accents="unicode")
    vectorizer = SeqVectorizer(analyzer=get_words_0, binary=not use_tf, use_idf=use_idf, encoding='utf-8', strip_accents="unicode")
    docs = np.array(sentences)
    # vecs = vectorizer.fit_transform(docs)
    vectorizer.fit(docs)
    vecs = vectorizer.transform(docs)

    return vecs, vectorizer


def get_words_vecs_with_sequential(vectorizer: SeqVectorizer, sentences: List[str]) -> Any:
    """追加学習の実施

    :param sentences: 
    :param use_tf: 
    :param use_idf: 
    :returns: 
    :rtype: 

    """
    docs = np.array(sentences)
    # vecs = vectorizer.fit_transform(docs)
    vectorizer.sequential_fit(docs)
    vecs = vectorizer.transform(docs)
    return vecs, vectorizer


def get_cosine_similarity_matrix(sentences: List[str], use_tf: bool = False, use_idf: bool = False) -> Tuple[List[Any], Any, Any]:
    """類似度の推定の実施

    :param sentences: 
    :param use_tf: 
    :param use_idf: 
    :returns: 
    :rtype: 

    """
    vecs, learned_model = get_words_vecs(sentences, use_tf=use_tf, use_idf=use_idf)
    precs = 3
    v1_input = vecs.toarray()
    v1_db = vecs.toarray()
    cs_similarity = np.round(cosine_similarity(v1_input, v1_db), precs)

    return cs_similarity, learned_model, vecs.toarray()


def get_cosine_similarity_matrix_with_sequential(vectorizer: SeqVectorizer, sentences: List[str]) -> Tuple[List[Any], Any, Any]:
    """追加学習モードの類似度の推定の実施

    :param sentences: 
    :param use_tf: 
    :param use_idf: 
    :returns: 
    :rtype: 

    """
    vecs, learned_model = get_words_vecs_with_sequential(vectorizer, sentences)

    precs = 3
    v1_input = vecs.toarray()
    v1_db = vecs.toarray()
    cs_similarity = np.round(cosine_similarity(v1_input, v1_db), precs)

    return cs_similarity, learned_model, vecs.toarray()


def get_cosine_similarity_matrix_by_learne_model(vectorizer: SeqVectorizer,
                                                 sentences: List[str],
                                                 learned_vecs,
                                                 use_tf: bool = False,
                                                 use_idf: bool = False) -> List[Any]:
    """学習済みモデルを用いた推論の実施

    :param sentences: 
    :param use_tf: 
    :param use_idf: 
    :returns: 
    :rtype: 

    """
    precs = 3

    docs = np.array(sentences)
    vecs_0 = vectorizer.transform(docs)
    vecs = np.vstack((vecs_0.toarray(), learned_vecs))
    cs_similarity = np.round(cosine_similarity(vecs, vecs), precs)

    return cs_similarity


def get_status_simlarity(sim_table: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """類似度の統計情報を算出する

    :param sim_table: 
    :returns: 
    :rtype: 

    """
    ar: List[float] = []
    for id, rec in enumerate(sim_table):
        idx_name = rec["index"]
        sts = rec["sentence"]
        sims = rec["similarity"]
        ar.extend(sims[id + 1:len(sims)])

    vmean = sum(ar) / len(ar)
    vmax = max(ar)
    vmin = min(ar)
    return vmean, vmin, vmax


def print_status(sim_table: List[Dict[str, Any]]):
    """類似度の統計情報を表示する

    :param sim_table: 
    :returns: 
    :rtype: 

    """
    vmean, vmin, vmax = get_status_simlarity(sim_table)
    print(f""" -- status of cosine similarity
  mean={vmean}
  min ={vmin}
  max ={vmax}
""", file=sys.stderr)


def print_as_dot(sim_table: List[Dict[str, Any]], cut_off: Optional[float] = None) -> str:
    """graphviz形式の類似度関連性グラフの出力

    :param sim_table: 
    :param cut_off: 
    :returns: 
    :rtype: 

    """
    result = ""

    vmean, vmin, vmax = get_status_simlarity(sim_table)
    if cut_off is not None:
        vmin = max(vmin, cut_off)
    for id, rec in enumerate(sim_table):
        idx_name = rec["index"]
        sts = rec["sentence"]
        sims = rec["similarity"]
        sts = re.sub(r"。", "。\\\\n", sts)
        sts = re.sub(r"\\n$", "", sts)

        a_columns = set(rec.keys()) - set(["index", "sentence", "similarity"])
        if len(a_columns) > 0:
            a_attrs = "," + ",".join([f"{v}=\"{rec[v]}\"" for v in a_columns])
        else:
            a_attrs = ""

        result += f"\t{id}[label=\"{idx_name}:{sts}\"{a_attrs}];\n"
        for id2 in range(id + 1, len(sims)):
            v = sims[id2]
            if cut_off is not None and v <= cut_off:
                continue
            w = int(v * 10) - int(vmin * 10)
            pw = 1 + v * 5
            result += f"\t{id}->{id2}[label=\"{v}\" weight={w} penwidth={pw:6.4f}];\n"

    if len(result) > 0:
        result = f"""
digraph g {{
{result}
}}
"""

    return result


def read_extend_word(extend_word_file: str) -> Dict[str, List[str]]:
    """わかち書きで分離された単語から、語句を追加するためのテーブルを読み込む

    :param extend_word_file: わかち書きされた語句のパターンと追加語句のリストを持つファイル
                             pattern  word
    :returns: 
    :rtype: 
    :remark:
    example of extend_word_file:
    書生	学生
    原$	        野原

    """
    if extend_word_file is None:
        return
    ext_dict: Dict[str, List[str]] = {}
    with open(extend_word_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            m = re.search(r"\s*(\S+)\s+(\S+)", line)
            if m is not None:
                k = m.group(1)
                v = m.group(2)
                if k in ext_dict:
                    print(f"#warn:csv_text_tfidf:read_extend_word:{k} is duplicated, {k}:{v} will be ignored.", file=sys.stderr)
                ext_dict[k] = re.split(r"\s*(?<!\\),\s*", v)
            else:
                print(f"#warn:csv_text_tfidf:read_extend_word:invalid format:{line}", file=sys.stderr)

    return ext_dict


def load_learned_model(model_file: str) -> Tuple[Any, Any, Any]:
    """モデルファイルの読み込み

    :param model_file: 
    :returns: 
    :rtype: 

    """
    with open(model_file, "rb") as f:
        p_data = pickle.load(f)
    learned_model = p_data[0]
    learned_vecs = p_data[1]
    learned_sim_table = p_data[2]

    # print(model_file)
    # print(type(learned_model))
    # print(type(learned_vecs))

    return learned_model, learned_vecs, learned_sim_table


def estimate_by_learned_model(model_file: str, sentences: List[str]) -> Tuple[pd.DataFrame, str]:
    """学習済みモデルを用いた推論

    :param model_file: 
    :param sentences: 
    :returns: 
    :rtype: 

    """
    learned_model, learned_vecs, learned_sim_table = load_learned_model(model_file)
    cs_similarity = get_cosine_similarity_matrix_by_learne_model(learned_model, sentences, learned_vecs)

    n_in = len(sentences)
    columns = ["index", "sentence"]
    df = pd.DataFrame(columns=columns)
    df["index"] = [v["index"] for v in learned_sim_table]
    df["sentence"] = [v["sentence"] for v in learned_sim_table]

    # 入力文書毎に列を追加し、各行の文書との類似度をDataFrameに格納する
    message = ""
    for ir in range(n_in):
        df[f"S_{ir}"] = cs_similarity[ir].tolist()[n_in:]
        if df[f"S_{ir}"].max() == 0:
            message += f"#warn:csv_text_tfidf:no relation were found for 'S_{ir}',\n" + "      you should learn new sentences again by '--sequential_learn'.\n"

    return df, message.rstrip()


def preprocess_of_sentences(sentences: List[str], remove_pattern: str = "", replace_pattern_file: str = "") -> List[str]:
    """推論処理前の文字列への処理

    :param sentences: 
    :param remove_pattern: 
    :param replace_pattern_file: 
    :returns: 
    :rtype: 

    """
    rp = None
    rof = None
    if len(remove_pattern) > 0:
        rp = removeSentences(remove_pattern)
    if len(replace_pattern_file) > 0:
        rpf = replaceSentences(replace_pattern_file)

    results = []
    for sts in sentences:
        if len(sts) == 0:
            continue
        # 入力文書に対する処理
        if rp is not None:
            sts = rp.do(sts)
        if rpf is not None:
            sts = rpf.do(sts)
        results.append(sts)

    return results


def use_learned_model(model_file: str, remove_pattern: str = "", replace_pattern_file: str = "") -> pd.DataFrame:
    """標準入力から読み込んだ文字列に対して学習済みモデルによる推論を実施

    :param model_file: 
    :param remove_pattern: 
    :param replace_pattern_file: 
    :returns: 
    :rtype: 

    """

    print("%inf:csv_text_tfidf:read text from stdin:please enter text", file=sys.stderr)
    sentences = []
    for line in fileinput.input(files="-"):
        line = line.rstrip()
        if len(line) == 0:
            continue
        sentences.append(line)
    print(">>>", file=sys.stderr)

    sentences = preprocess_of_sentences(sentences, remove_pattern=remove_pattern, replace_pattern_file=replace_pattern_file)

    df, message = estimate_by_learned_model(model_file, sentences)

    df.to_csv(sys.stdout, index=False)
    print(message, file=sys.stderr)

    return df


def do_remove_pattern(sentences: List[str], remove_pattern: str) -> List[str]:
    """FIXME! briefly describe function

    :param sentences: 
    :param remove_pattern: 
    :returns: 
    :rtype: 

    """
    if len(remove_pattern) > 0:
        rp = removeSentences(remove_pattern)
        stses = []
        for sts in sentences:
            sts = rp.do(sts)
            stses.append(sts)
        sentences = stses

    return sentences


def do_replace_pattern(sentences: List[str], replace_pattern_file: str) -> List[str]:
    if len(replace_pattern_file) > 0:
        rpf = replaceSentences(replace_pattern_file)
        stses = []
        for sts in sentences:
            sts = rpf.do(sts)
            stses.append(sts)
        sentences = stses

    return sentences


def make_similarity_table(csv_df: pd.DataFrame,
                          cs_similarity: List[Any],
                          sim_table: List[Dict[str, Any]] = None,
                          additional_columns: List[str] = None) -> List[Dict[str, Any]]:
    """類似度算出結果から、類似度テーブルを生成する

    :param csv_df: 
    :param cs_similarity: 
    :param sim_table: 
    :returns: 
    :rtype: 

    """
    if sim_table is None:
        sim_table = []
    for ir in range(len(csv_df)):
        rec = csv_df.iloc[ir]
        if idx_name is not None:
            idx = rec[idx_name]
        else:
            idx = ir
        sts = rec[column]

        df_rec = {"index": idx, "sentence": sts, "similarity": cs_similarity[ir].tolist()}
        if additional_columns is not None:
            df_rec.update(rec[additional_columns])
        sim_table.append(df_rec)

    return sim_table


def stack_vecs(vecs1: np.array, vecs2: np.array) -> np.array:
    """2次元配列を行方向に追加する。列数は多い方に合わせたものとする。

    :param vecs1: 
    :param vecs2: 
    :returns: 
    :rtype: 

    """

    vecs_tmp = np.zeros((vecs1.shape[0], max(vecs1.shape[1], vecs2.shape[1])))
    vecs_tmp[:vecs1.shape[0], :vecs1.shape[1]] = vecs1
    vecs = np.vstack((vecs_tmp, vecs2))

    return vecs


if __name__ == "__main__":
    args = init()
    csv_file = args.csv_file
    column = args.column
    add_columns_s = args.ACOLUMNS

    idx_name = args.INDEX
    output_mode = args.MODE
    use_tf = args.USE_TF
    use_idf = args.USE_IDF

    cut_off = args.CUT_OFF
    check_frequency_limit = args.CFLIMIT

    PATH_OF_USERDICT = args.UDICT
    remove_pattern = args.RPATTERN
    replace_pattern_file = args.REP_FILE
    extend_word_file = args.EXT_FILE

    model_file_out = args.VECS_FILE_0
    model_file_in = args.VECS_FILE_1
    seq_learn = args.SEQ_MODE

    status_mode = args.STATUS
    debug_mode = args.DEBUG
    DEBUG = debug_mode
    EXTEND_WORD_LIST = read_extend_word(extend_word_file)
    min_length = args.MLENGTH
    MINIMUM_LENGTH_OF_WORD = min_length
    REMOVE_WORDS = args.RWORDS

    if add_columns_s is not None:
        add_columns = re.split(r"\s*(?<!\\),\s*", add_columns_s)
    else:
        add_columns = None

    if seq_learn is not None:
        print(f"%inf:csv_text_tfidf:sequential learning mode: {seq_learn}", file=sys.stderr)
        if seq_learn.find(",") == -1:
            print(f"??error:csv_text_tfidf:invalid format of '--sequential_learn':{seq_learn}", file=sys.stderr)
            sys.exit(1)
        m_file, model_file_out = re.split(r"\s*(?<!\\),\s*", seq_learn)
        learned_model, learned_vecs, learned_sim_table = load_learned_model(m_file)
        if model_file_in is not None:
            print(f"#warn:csv_text_tfidf:'--model_file' will be ignored", file=sys.stderr)
    else:
        learned_model = None
        learned_vecs = None
        learned_sim_table = None

    # print(args, file=sys.stderr)

    # -- only evaluating by using learned model
    if model_file_in is not None:
        print(f"%inf:csv_text_tfidf:only etimation mode:", file=sys.stderr)
        use_learned_model(model_file_in, remove_pattern=remove_pattern, replace_pattern_file=replace_pattern_file)
        sys.exit(0)

    if csv_file == "-":
        csv_file = sys.stdin

    csv_df = pd.read_csv(csv_file)
    if column not in csv_df:
        print(f"??error:csv_text_tfidf:{column} not in {csv_file}", file=sys.stderr)
        sys.exit(1)
    if add_columns is not None and any([v not in csv_df for v in add_columns]):
        print(f"??error:csv_text_tfidf:{add_columns} not in {csv_file}", file=sys.stderr)
        sys.exit(1)

    print(f"%inf:csv_text_tfidf:{len(csv_df)} records from {csv_file}", file=sys.stderr)

    # -- pre-processed
    csv_df.fillna("", inplace=True)
    sentences = csv_df[column].values
    all_s_0 = "".join(sentences)
    sentences = do_remove_pattern(sentences, remove_pattern)
    sentences = do_replace_pattern(sentences, replace_pattern_file)

    if status_mode:
        all_s_1 = "".join(sentences)
        print(f"%inf:csv_text_tfidf:number of records={len(csv_df)}, characters=[orig={len(all_s_0)},prepro={len(all_s_1)}]",
              file=sys.stderr)
        check_frequency_words(sentences, limit_ratio=check_frequency_limit)
        sys.exit(0)

    if (len(remove_pattern) > 0 or len(replace_pattern_file) > 0) and debug_mode:
        check_frequency_words(sentences, limit_ratio=check_frequency_limit)
        print(f"%inf:debug:csv_text_tfidf:pre-processed text", file=sys.stderr)
        print("\n".join([f"{i} : {v}" for i, v in enumerate(sentences)]), file=sys.stderr)

    # -- learning
    if learned_model is not None:
        # sequential learning
        cs_similarity, learned_model, vecs = get_cosine_similarity_matrix_with_sequential(learned_model, sentences)
        if learned_vecs.shape[1] < vecs.shape[1]:
            print(f"#warn:csv_text_tfidf:new words were detected:{vecs.shape[1]-learned_vecs.shape[1]}", file=sys.stderr)
        vecs = stack_vecs(learned_vecs, vecs)
    else:
        # normal learning
        print(f"%inf:csv_text_tfidf:normal learning mode:", file=sys.stderr)
        cs_similarity, learned_model, vecs = get_cosine_similarity_matrix(sentences, use_tf=use_tf, use_idf=use_idf)
        # print(type(learned_model))

    sim_table = make_similarity_table(csv_df, cs_similarity, sim_table=learned_sim_table, additional_columns=add_columns)

    if model_file_out is not None:
        with open(model_file_out, "wb") as f:
            pickle.dump([learned_model, vecs, sim_table], f)
        print(f"%inf:csv_text_tfidf:learned model was written into {model_file_out}", file=sys.stderr)
        sys.exit(0)

    result_str_simple = ""
    for rec in sim_table:
        idx = rec["index"]
        if add_columns is not None:
            a_rec_s = "," + ",".join([rec[v] for v in add_columns])
        else:
            a_rec_s = ""
        sim = rec["similarity"]
        result_str_simple += f"{idx},{sim}{a_rec_s}\n"
    if debug_mode:
        print(f"%inf:debug:csv_text_tfidf:cosine similarity matrix:", file=sys.stderr)
        print(result_str_simple, file=sys.stderr)
    if output_mode == "json":
        result_str = json.dumps(sim_table, ensure_ascii=False, indent=2)
    elif output_mode == "dot":
        result_str = print_as_dot(sim_table, cut_off=cut_off)
    elif output_mode == "simple":
        result_str = result_str_simple

    if len(result_str) == 0:
        print("#warn:csv_text_tfidf:result was empty", file=sys.stderr)
    print(result_str)

    print_status(sim_table)
