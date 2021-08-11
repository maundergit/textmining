#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         text_mining_lib.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-04-18 12:49:28>
# Licence:
# ----------------------------------------------------------------------

import re
import regex

from pathlib import Path

from collections import Counter

import jaconv
import pykakasi
from kanjize import kanji2int
import neologdn
import unicodedata

import MeCab

import pandas as pd

# sudo update-alternatives --config mecab-dictionary
MECAB_DICT = "/var/lib/mecab/dic/ipadic-utf8"


def check_hiragana(word):
    # Pythonの正規表現で漢字・ひらがな・カタカナ・英数字を判定・抽出・カウント | note.nkmk.me https://note.nkmk.me/python-re-regex-character-type/
    p = regex.compile(r'\p{Script=Hiragana}+')
    result = p.fullmatch(word) is not None
    return result


def check_ascii_symbol(word):
    # Pythonの正規表現で漢字・ひらがな・カタカナ・英数字を判定・抽出・カウント | note.nkmk.me https://note.nkmk.me/python-re-regex-character-type/
    p = re.compile('[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]+')
    result = p.fullmatch(word) is not None
    return result


def check_zenkaku_symbol_with_japanese(word):
    # Pythonの正規表現で漢字・ひらがな・カタカナ・英数字を判定・抽出・カウント | note.nkmk.me https://note.nkmk.me/python-re-regex-character-type/
    p = re.compile('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]+')
    result = p.fullmatch(word) is not None
    return result


def check_zenkaku_symbol(word):
    # Pythonの正規表現で漢字・ひらがな・カタカナ・英数字を判定・抽出・カウント | note.nkmk.me https://note.nkmk.me/python-re-regex-character-type/
    p = re.compile('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]+')
    result = p.fullmatch(word) is not None
    return result


def get_mecab_parser(path_of_userdic, mecab_option=""):
    if path_of_userdic is not None and len(path_of_userdic) > 0:
        mecab_option = mecab_option + f" -u {path_of_userdic}"
    m = MeCab.Tagger(f"-d {MECAB_DICT} {mecab_option}")

    return m


class removeSentences():
    def __init__(self, sentences_to_remove, replace_to=""):
        self.__semtemces_to_remove_pattern = None
        self.__replace_to = replace_to
        if sentences_to_remove is None:
            return
        if isinstance(sentences_to_remove, list):
            pat = "(" + "|".join(sentences_to_remove) + ")"
        elif isinstance(sentences_to_remove, str) and Path(sentences_to_remove).exists():
            with open(sentences_to_remove, "r") as f:
                pat = "(" + "|".join([line.strip() for line in f.readlines()]) + ")"
        elif isinstance(sentences_to_remove, str):
            pat = sentences_to_remove
        self.__semtemces_to_remove_pattern = re.compile(pat)

    def do(self, sentence):
        result = sentence
        if self.__semtemces_to_remove_pattern is not None:
            result = re.sub(self.__semtemces_to_remove_pattern, self.__replace_to, sentence)
        return result


class replaceSentences():
    def __init__(self, pattern_file):
        if not Path(pattern_file).exists():
            raise FileNotFoundError(f"??error:textmining_lib:replaceSentences:{pattern_file} was not found")
        self.__patterns = []
        with open(pattern_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                m = re.match(r"s/(.+)/(.*)/", line)
                if m is None:
                    raise ValueError(f"??error:textmining_lib:replaceSentences:invalid pattern {line} in {pattern_file}")
                pat = m.group(1)
                rep = m.group(2)
                self.__patterns.append([re.compile(pat), rep])

    def do(self, sentence):
        result = sentence
        for ps in self.__patterns:
            result = re.sub(ps[0], ps[1], result)

        return result


def get_words(sentence, mecab_option="", path_of_userdict="", remove_words=["。", "、", "？", ".", ",", "?"]):
    """分かち書きによる語句リストの生成

    :param sentence: 文章
    :param mecab_option:Mecab向けオプション 
    :param path_of_userdict: ユーザー辞書
    :param remove_words: 結果から削除すべき語句リスト
    :returns: 語句リスト
    :rtype: 

    """
    m = get_mecab_parser(path_of_userdict, mecab_option=mecab_option)
    # m = MeCab.Tagger(f"-d {MECAB_DICT} {mecab_option}")
    node = m.parseToNode(sentence)

    words = []
    while node:
        word = node.surface
        if len(word) > 0 and word not in remove_words:
            words.append(word)
        node = node.next

    return words


def get_yomi(sentence, mecab_option=""):
    """文字列の語句単位の読みのカナのリストをせいせいする

    :param sentence: 
    :param mecab_option: 
    :returns: 
    :rtype: 

    """
    kakasi = pykakasi.kakasi()
    m = get_mecab_parser("", mecab_option=mecab_option)
    # m = MeCab.Tagger(f"-d {MECAB_DICT} {mecab_option}")
    node = m.parseToNode(sentence)
    yomis = []
    while node:
        word = node.surface
        if len(word) == 0:
            node = node.next
            continue
        cvs = re.split(r",", node.feature)
        posid_0 = cvs[0]
        posid_1 = cvs[1]
        if len(cvs) < 8:
            ym = word
            for ym_k in kakasi.convert(word):
                ym = ym + ym_k["kana"]
        else:
            ym = cvs[7]
        yomis.append(ym)
        node = node.next

    return yomis


def get_count_of_word(sentence, mecab_option="", path_of_userdict="", remove_words=["。", "、", "？", ".", ",", "?"]):
    words = get_words(sentence, mecab_option=mecab_option, path_of_userdict=path_of_userdict, remove_words=remove_words)
    words_count = Counter(words)
    return words_count


def normalize_text_genkei(text,
                          link_type=False,
                          pos_list=None,
                          unicode_normalize="NFKC",
                          with_neologdn=False,
                          for_mecab=None,
                          kansuji=True,
                          mecab_option=""):
    """文字列を正規化するとともに語句を表層形から原形へ変換する

    :param text: 入力文字列
    :param link_type: True指定時に、[変換前語句](変換後語句)形式で文字列に埋め込む
    :param pos_list: 変換対象とする品詞のリスト、無指定時は全品詞が原形に変換される
    :param unicode_normalize: unicodedataのユニコード正規化方式、["NFD", "NFC", "NFCD", "NFKC"]
    :param with_neologdn: 長音表記を正規化する場合にTrueを指定(はーーーい => はーい)
    :param for_mecab: mecabのオプション文字列
    :param kansuji: True指定時に名詞の一部ではない漢数字を半角数字へ変換する(例：一万五百円=>10500円、九州=>九州)
    :param mecab_option: 
    :returns: 処理後の文字列
    :rtype: 

    :remark:
    日本語テキストの前処理：neologdn、大文字小文字、Unicode正規化 - tuttieee’s blog http://tuttieee.hatenablog.com/entry/ja-nlp-preprocess

    """
    result = normalize_text(text, unicode_normalize=unicode_normalize, with_neologdn=with_neologdn, for_mecab=for_mecab, kansuji=kansuji)
    word_df = get_table_by_mecab(None, result, mecab_option=mecab_option)

    for idx in range(len(word_df)):
        rec = word_df.iloc[idx]
        w = rec["表層形"]
        g = rec["原形"]
        p = rec["品詞"]
        if pos_list is not None and p not in pos_list:
            continue
        # if len(w) < 2 :
        #     continue
        if g == "*" or w == g:
            continue
        if link_type:
            result = re.sub("(" + w + ")", f"[\\1]({g})", result)
        else:
            result = re.sub(w, g, result)
    return result


def normalize_text(text, unicode_normalize="NFKC", with_neologdn=False, for_mecab=None, kansuji=False, mecab_option=""):
    """テキストの正規化を行う

    :param text: 入力文字列
    :param unicode_normalize: unicodedataのユニコード正規化方式、["NFD", "NFC", "NFCD", "NFKC"]
    :param with_neologdn: 長音表記を正規化する場合にTrueを指定(はーーーい => はーい)
    :param for_mecab: mecabのオプション文字列
    :param kansuji: True指定時に名詞の一部ではない漢数字を半角数字へ変換する(例：一万五百円=>10500円、九州=>九州)
    :returns: 処理後の文字列
    :rtype: 

    :remark:
    日本語テキストの前処理：neologdn、大文字小文字、Unicode正規化 - tuttieee’s blog http://tuttieee.hatenablog.com/entry/ja-nlp-preprocess

    """
    result = text

    if not isinstance(result, str):
        return result

    result = re.sub(r"[ \t]+", " ", result)
    if kansuji:
        # result = re.sub(r"[一二三四五六七八九零百千万億兆京]+", lambda x: str(kanji2int(x.group(0))), result)
        result = search_numeric_and_convert(result, mecab_option=mecab_option)

    # result = jaconv.normalize(result, unicode_normalize)
    if unicode_normalize is not None and unicode_normalize in ["NFD", "NFC", "NFCD", "NFKC"]:
        # mode in [NFD (Normalization Form Canonical Decomposition),
        #          NFC (Normalization Form Canonical Composition),
        #          NFCD (Normalization Form Compatibility Decomposition),
        #          NFKC (Normalization Form Compatibility Composition)]
        # NFKC:
        # "ｱｲｳあいう１２３ＡＢＣ一二三（）［］－＋＝" => 'アイウあいう123ABC一二三()[]-+='
        result = unicodedata.normalize(unicode_normalize, result)

    if with_neologdn:
        # 表記ゆれの正規化: unicodedata.normalizeに加えて長音の処理などが含まれている
        result = neologdn.normalize(result)

    if for_mecab is not None and len(for_mecab) > 1:
        tr_table = str.maketrans(dict(zip(for_mecab[0], for_mecab[1])))
        result = result.translate(tr_table)

    return result


def search_numeric_and_convert(sentence, mecab_option=""):
    """名詞/数が連続するものを半角数字に変換する。
    名詞の一部となっている漢数字は変換しない。

    :param sentence: 
    :param mecab_option: 
    :returns: 
    :rtype: 

    :remark:
    名詞の一部か否かにかかわらず漢数字を全て半角数字に変換する場合は以下のスクリプトで変換できる
    from kanjize import kanji2int
    re.sub(r"[一二三四五六七八九零百千万億兆京]+", lambda x: str(kanji2int(x.group(0))), text)


    """
    result = ""
    m = get_mecab_parser("", mecab_option=mecab_option)
    node = m.parseToNode(sentence)
    numeric_words = []
    while node:
        word = node.surface
        if len(word) == 0:
            node = node.next
            continue
        cvs = re.split(r",", node.feature)
        posid_0 = cvs[0]
        posid_1 = cvs[1]

        if (posid_0 != "名詞" or posid_1 != "数") and len(numeric_words) > 0:
            # word = re.sub(r"[一二三四五六七八九零百千万億兆京]+", lambda x: str(kanji2int(x.group(0))), word)
            w = "".join(numeric_words)
            if re.search(r"[一二三四五六七八九零百千万億兆京]+", w) is not None:
                w2 = str(kanji2int(w))
            else:
                w2 = w
            result += w2 + word
            numeric_words = []
        else:
            if posid_0 == "名詞" and posid_1 == "数":
                numeric_words.append(word)
            else:
                result += word

        node = node.next

    if len(numeric_words) > 0:
        # word = re.sub(r"[一二三四五六七八九零百千万億兆京]+", lambda x: str(kanji2int(x.group(0))), word)
        w = "".join(numeric_words)
        if re.search(r"[一二三四五六七八九零百千万億兆京]+", w) is not None:
            w2 = str(kanji2int(w))
        else:
            w2 = w
        result += w2

    return result


def get_table_by_mecab(output_df, sentence, mecab_option=""):
    """文字列中の語句の品詞等と出現数のPandasデータフレームを生成する

    :param output_df: 出力用データフレーム、None指定時は自動的に生成される
    :param sentence: 
    :param mecab_option: 
    :returns: Pandasデータフレーム
    :rtype: 

    """

    output_columns = ["表層形", "品詞", "品詞細分類1", "品詞細分類2", "品詞細分類3", "原形", "読み", "発音", "出現数", "出現長"]
    if output_df is None:
        output_df = pd.DataFrame(columns=output_columns)

    m = get_mecab_parser("", mecab_option=mecab_option)
    # m = MeCab.Tagger(f"-d {MECAB_DICT} {mecab_option}")
    ma = m.parse(sentence)
    for line in ma.split("\n"):
        line = line.rstrip()
        if line == "EOS":
            break
        cvs = line.split("\t", maxsplit=1)
        rec = {}
        if any(output_df["表層形"] == cvs[0]):
            output_df.loc[output_df["表層形"] == cvs[0], "出現数"] += 1
            continue
        rec["表層形"] = cvs[0]

        cvs = cvs[1].split(",")
        rec["品詞"] = cvs[0]
        for i in range(1, 4):
            # if cvs[i] == "*":
            #     break
            rec[output_columns[i + 1]] = cvs[i]
        rec["原形"] = cvs[6] if len(cvs) > 6 else ""
        rec["読み"] = cvs[7] if len(cvs) > 7 else ""
        rec["発音"] = cvs[8] if len(cvs) > 8 else ""
        rec["出現数"] = 1
        rec["出現長"] = len(rec["表層形"])
        output_df = output_df.append(rec, ignore_index=True)

    return output_df


if __name__ == "__main__":
    import fileinput
    import sys

    pos_list = ["名詞"]
    for line in fileinput.input():
        line = line.rstrip()
        if len(line) == 0:
            continue
        result = normalize_text_genkei(line, link_type=True, pos_list=pos_list)
        # result = search_numeric_and_convert(lines)
        # result = get_yomi(line)
        print(result)
