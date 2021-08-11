#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         ja_depndency_analysis.py
# Description:
#
# Author:       m.akei
# Copyright:    (c) 2021 by m.na.akei
# Time-stamp:   <2021-06-06 12:03:21>
# Licence:
# ----------------------------------------------------------------------
import argparse
import fileinput
import textwrap
import sys

import re
from pathlib import Path
import csv

import ginza
import spacy
from spacy import displacy
import deplacy

import pandas as pd

VERSION = 1.0


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
example:
  ja_depndency_analysis.py --output=wagahaiwa_nekodearu_ana.csv wagahaiwa_nekodearu_utf8.txt
  ja_depndency_analysis.py --read_csv --word=吾輩 wagahaiwa_nekodearu_ana.csv
  ja_depndency_analysis.py --read_csv --print_subject=test.txt wagahaiwa_nekodearu_ana.csv

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))

    arg_parser.add_argument("--word", dest="WORD", help="word to make dependency", type=str, metavar="WORD", default=None)

    arg_parser.add_argument("--read_csv", dest="READ_CSV", help="read csv file as analized result", action='store_true')

    arg_parser.add_argument("--print", dest="PRINT", help="printdisplacy as text", action="store_true")
    arg_parser.add_argument("--print_svg", dest="PRINT_SVG", help="path of displacy as svg", type=str, metavar="SVG_FILE", default=None)
    arg_parser.add_argument("--print_dot", dest="PRINT_DOT", help="path of displacy as dot", type=str, metavar="DOT_FILE", default=None)
    arg_parser.add_argument("--print_subject",
                            dest="PRINT_SUBJ",
                            help="path of subject/dependency",
                            type=str,
                            metavar="TEXT_FILE",
                            default=None)

    arg_parser.add_argument("--output",
                            dest="OUTPUT",
                            help="path of output csv :default=stdout",
                            type=str,
                            metavar='FILE',
                            default=sys.stdout)

    arg_parser.add_argument('text_file', metavar='FILE', help='file to read')
    # arg_parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    # arg_parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = arg_parser.parse_args()
    return args


# GiNZAを使って係り受け解析をやってみる - Re:ゼロから始めるML生活 https://www.nogawanogawa.com/entry/ginza-2
# 何もない所から一瞬で、自然言語処理と係り受け解析をライブコーディングする手品を、LTでやってみた話 - Qiita https://qiita.com/youwht/items/b047225a6fc356fd56ee
class ja_dependency_analysis:
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.__doc = None
        self.__ana_result = []

    def __get_analized_results(self, text):
        print("%inf:ja_dependency_analysis:parsing..", file=sys.stderr)
        self.__doc = self.nlp(text)

        results = []

        print("%inf:ja_dependency_analysis:processing sentences", file=sys.stderr)
        count = 0
        for sent in self.__doc.sents:
            count += 1
            if count % 100 == 1:
                print(f"  processing: {count}", file=sys.stderr)
            # print(sent)
            for token in sent:
                i_dict = {}
                # about universal dependency, see https://www.anlp.jp/proceedings/annual_meeting/2015/pdf_dir/E3-4.pdf
                i_dict["index"] = token.i  #token index
                i_dict["original"] = token.orth_  # original text
                i_dict["yomi"] = ginza.reading_form(token)  # yomi
                i_dict["posid"] = token.pos_  # posid: UD
                i_dict["posid_text"] = token.tag_  # posid as text
                i_dict["lemma"] = token.lemma_  # base
                i_dict["katsuyou"] = ginza.inflection(token)
                i_dict["rank"] = token.rank
                i_dict["genkei"] = token.norm_
                i_dict["is_oov"] = token.is_oov  # undefined?
                i_dict["is_stop"] = token.is_stop  # stop word?
                i_dict["has_vector"] = token.has_vector  # word2vec ?
                i_dict["lefts"] = list(token.lefts)  # left side
                i_dict["rights"] = list(token.rights)  # right side
                i_dict["depend"] = token.dep_  # denpendency: UD
                i_dict["head_index"] = token.head.i  # token index for denpendency
                i_dict["head_text"] = token.head.text  # text for dependency
                results.append(i_dict)

        return results

    def __get_words(self, m, head_index, ana_result):

        if m["head_index"] == head_index:
            for n in ana_result:
                self.__get_words(n, m["index"], ana_result)

            self.__results_indices.append(m["index"])

        return

    def read_csv(self, csv_file):
        csv_df = pd.read_csv(csv_file)
        self.__ana_result = csv_df.to_dict(orient='records')

    def set_text(self, text):
        print("%inf:ja_dependency_analysis:start analysis", file=sys.stderr)
        self.__ana_result = self.__get_analized_results(text)

    def show_displacy(self):
        # Visualizers · spaCy Usage Documentation https://spacy.io/usage/visualizers
        if self.__doc is None:
            print(f"??error:ja_dependency_analysis:you must call 'set_text' before this", file=sys.stderr)
            sys.exit(1)
        svg_str = displacy.render(self.__doc, style='dep', options={'distance': 90})
        return svg_str

    def show_displacy_by_dot(self):
        # Visualizers · spaCy Usage Documentation https://spacy.io/usage/visualizers
        if self.__doc is None:
            print(f"??error:ja_dependency_analysis:you must call 'set_text' before this", file=sys.stderr)
            sys.exit(1)
        # print(deplacy.serve(self.__doc))
        dot_str = deplacy.dot(self.__doc)
        return dot_str

    def print_dependecy(self):
        if self.__doc is None:
            print(f"??error:ja_dependency_analysis:you must call 'set_text' before this", file=sys.stderr)
            sys.exit(1)
        deplacy.render(self.__doc)

    def get_result_as_list(self):
        if len(self.__ana_result) == 0:
            print(f"??error:ja_dependency_analysis: 'set_text' must be called before this.", file=sys.stderr)
            exit(1)
        return self.__ana_result

    def get_result_as_dataframe(self):
        if len(self.__ana_result) == 0:
            print(f"??error:ja_dependency_analysis: 'set_text' must be called before this.", file=sys.stderr)
            exit(1)

        csv_df = pd.DataFrame(self.__ana_result)

        return csv_df

    def get_indecies_for_word(self, word):
        # search target word in analized result
        if len(self.__ana_result) == 0:
            print(f"??error:ja_dependency_analysis: 'set_text' must be called before this.", file=sys.stderr)
            exit(1)

        tgt_idxs = []
        for idx, ar in enumerate(self.__ana_result):
            if ar["original"] == word:
                tgt_idxs.append(idx)
        # if len(tgt_idxs) == 0:
        #     print(f"??error:ja_dependency_analysis:'{word}' was not found", file=sys.stderr)
        # print(f"%inf:ja_dependency_analysis:index={tgt_idxs} of {word}", file=sys.stderr)
        return tgt_idxs

    def get_dependency(self, target_index):
        if len(self.__ana_result) == 0:
            print(f"??error:ja_dependency_analysis: 'set_text' must be called before this.", file=sys.stderr)
            exit(1)

        tgt_idx = target_index

        head_index = self.__ana_result[tgt_idx]["head_index"]

        self.__results_indices = [head_index]
        for m in self.__ana_result:
            # print(">>", head_index, m["head_index"], m["depend"], m["head_index"] == head_index)
            if m["head_index"] == head_index and m["depend"] != "ROOT":
                self.__get_words(m, head_index, self.__ana_result)

        clauses = []
        target_word = self.__ana_result[tgt_idx]["original"]
        print(f"%inf:ja_dependency_analysis:{target_word}:dependences={sorted(self.__results_indices)}", file=sys.stderr)
        for clause_num in sorted(self.__results_indices):
            clauses.append(str(self.__ana_result[clause_num]["original"]))

        return clauses

    def get_subject_dependency(self):
        # GiNZA+Elasticsearchで係り受け検索の第一歩 - Taste of Tech Topics https://acro-engineer.hatenablog.com/entry/2019/12/06/120000
        # 自然言語処理ライブラリのGiNZAを使って係り受け解析を試す https://www.virment.com/ginza-dependency-parse/
        nsubj_list = []
        for i_dict in self.__ana_result:
            if i_dict["depend"] in ["nsubj", "iobj"]:
                nsubj_list.append({"subject": i_dict["lemma"], "dependency": self.__ana_result[i_dict["head_index"]]["lemma"]})
        return nsubj_list


def get_dependency(jda, word):
    idxs = jda.get_indecies_for_word(word)
    clauses = []
    for idx in idxs:
        clauses.append(jda.get_dependency(idx))

    return clauses


if __name__ == "__main__":
    args = init()
    text_file = args.text_file
    output_csv = args.OUTPUT

    target_word = args.WORD

    read_csv = args.READ_CSV

    text_mode = args.PRINT
    displacy_mode = args.PRINT_SVG
    dot_mode = args.PRINT_DOT
    subject_mode = args.PRINT_SUBJ

    if text_file == "-":
        text_file = sys.stdin

    jda = ja_dependency_analysis()

    if read_csv:
        jda.read_csv(text_file)
    else:
        lines = []
        if text_file == sys.stdin:
            text_file_f = text_file
        else:
            text_file_f = open(text_file, "r")

        for line in text_file_f.readlines():
            line = line.rstrip()
            lines.append(line)

        jda.set_text("".join(lines))

        csv_df = jda.get_result_as_dataframe()
        csv_df.to_csv(output_csv, index=False)

    if displacy_mode is not None:
        svg_str = jda.show_displacy()
        with open(displacy_mode, "w") as f:
            print(svg_str, file=f)
        print(f"%inf:ja_dependency_analysis:dependency were written into {displacy_mode}", file=sys.stderr)

    if dot_mode is not None:
        dot_str = jda.show_displacy_by_dot()
        with open(dot_mode, "w") as f:
            print(dot_str, file=f)
        print(f"%inf:ja_dependency_analysis:dependency were written into {dot_mode}", file=sys.stderr)

    if subject_mode is not None:
        subj_list = jda.get_subject_dependency()
        with open(subject_mode, "w") as f:
            print(f"subject,dependency", file=f)
            for sd in subj_list:
                print(f"{sd['subject']},{sd['dependency']}", file=f)
        print(f"%inf:ja_dependency_analysis:subject/dependency were written into {subject_mode}", file=sys.stderr)

    if text_mode:
        jda.print_dependecy()

    if target_word is not None:
        clauses = get_dependency(jda, target_word)
        f = csv.writer(sys.stdout)
        for cl in clauses:
            f.writerow(cl)
