#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Name:         csv_text_solr.py
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
import urllib3
import json
import pprint

import datetime

from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Type, NoReturn

from textmining_lib import removeSentences, replaceSentences
from csv_text_tfidf import get_words_0, read_extend_word

VERSION = 1.0

# EXTENDED_PREFIX = "_ext"

F_indent_text = lambda x: textwrap.fill(x, width=130, initial_indent="\t", subsequent_indent="\t")


def init():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent('''
remark:
  using '--init_sample', template files for initial script and definition of fields are made.

  In '--fields_type', 'TYPE' means 'type' of field type of solr.
  if 'pdate' was used, then value of the field was converted into UTC, that has format defined 
  by '--datetime_format'.

  '--auto_add_fields' is used to add all columns of csv into solr.

  following options only make sense of '--extend_columns':
  '--user_dictionary', '--minmum_length_of_word', '--remove_words', '--remove_pattern', '--replace_pattern_file', '--extend_word_file'

  using '--extend_word_file', bow of sentence is extended with additional words.
  this option may be usefull when the configuration of devices is important.
  format of 'extend word file' there is 'word word1,word2' in a line.

  in result of '--retrieve_terms_status', df and ttf mean following:
    - docFreq: number of documents this term occurs in.
    - totalTermFreq: number of tokens for this term.
  see TermStatistics (Lucene 8.1.1 API) https://lucene.apache.org/core/8_1_1/core/org/apache/lucene/search/TermStatistics.html

  about datetime range, see following:
    Working with Dates | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/working-with-dates.html

example:
  ${SOLR_HOME}/bin/solr delete -c wagahaiwa_nekodearu
  ${SOLR_HOME}/bin/solr create_core -c wagahaiwa_nekodearu
  csv_text_solr.py --add_fields wagahaiwa_nekodearu_fields.json --core wagahaiwa_nekodearu
  csv_text_solr.py --core wagahaiwa_nekodearu --fields_information
  csv_text_solr.py --core wagahaiwa_nekodearu --datetime_format="%Y-%m-%d" wagahaiwa_nekodearu.csv
  csv_text_solr.py --core wagahaiwa_nekodearu --datetime_format="%Y-%m-%d" --key=date --extend_columns=content wagahaiwa_nekodearu.csv

  cat <<EOF > rep_pat.txt
  s/《[^》]*》//
  s/(?i)GHI/KLM/
  EOF
  cat <<EOF > tfidf_extend_word.txt
  # regexp word1,word2,...
  書生	学生,学校
  原$	野原
  EOF
  csv_text_solr.py --core wagahaiwa_nekodearu --datetime_format="%Y-%m-%d" --key=date --extend_columns=content \\
                   --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt wagahaiwa_nekodearu.csv

  csv_text_solr.py --core wagahaiwa_nekodearu --retrieve_terms

'''))

    arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(VERSION))
    arg_parser.add_argument("--init_sample",
                            dest="INIT_SAMPLE",
                            help="make sample script and definition of fields",
                            action="store_true",
                            default=False)

    arg_parser.add_argument("--host",
                            dest="HOST",
                            help="ip address of host, default=127.0.0.1",
                            type=str,
                            metavar='IP_ADDRESS',
                            default="127.0.0.1")
    arg_parser.add_argument("--port", dest="PORT", help="port number, default=8983", type=int, metavar='PORT', default=8983)
    arg_parser.add_argument("--core", dest="CORE", help="name of core of solr", type=str, metavar='NAME', required=True)
    arg_parser.add_argument("--key", dest="KEY", help="name of column as key", type=str, metavar='COLUMN', default=None)

    arg_parser.add_argument("--column_map",
                            dest="CMAP",
                            help="mapping table between column and field",
                            type=str,
                            metavar="COLUMN:FIELD[,COLUMN:FIELD...]",
                            default=None)

    arg_parser.add_argument("--auto_add_fields",
                            dest="FIELDS_ADD",
                            help="to add fields, that are in column map but not in solr, when '--column_map' is used..",
                            action="store_true",
                            default=False)

    arg_parser.add_argument("--add_fields",
                            dest="FIELDS",
                            help="path of json file to define fields",
                            type=str,
                            metavar='JSON_FILE',
                            default=None)

    arg_parser.add_argument("--show_fields_information",
                            dest="FIELDS_INF",
                            help="retrieve information of fields",
                            action="store_true",
                            default=False)

    arg_parser.add_argument("--fields_type",
                            dest="FIELDS_TYPE",
                            help="list of type of each field, those was used to convert data type when adding ones to solr.",
                            type=str,
                            metavar="FIELD:TYPE[,FIELD:TYPE..]",
                            default=None)

    arg_parser.add_argument(
        "--datetime_format",
        dest="DATETIME_FMT",
        help="format of datetime in columns whose type is 'pdate' in '--fields_type', default='%%Y-%%m-%%d %%H:%%M:%%S'",
        type=str,
        metavar="datetime_format",
        default="%Y-%m-%d %H:%M:%S")

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

    arg_parser.add_argument("--retrieve_terms_status",
                            dest="RET_TERMS",
                            help="retrieve df(docFreq) and ttf(totalTermFreq) about terms from solr, top100, with csv format",
                            type=str,
                            metavar="FIELD",
                            default=None)

    arg_parser.add_argument("--analysis_terms",
                            dest="ANA_TERMS",
                            help="retrieve analyzed result for string",
                            type=str,
                            metavar="ANA_TERMS",
                            default=None)

    arg_parser.add_argument('csv_files', metavar='CSV_FILE', nargs='*', help='csv files to read. if empty, stdin is used')

    args = arg_parser.parse_args()
    return args


class communicateSlor():
    """Solr通信用クラス
    """
    def __init__(self, host: str, port: str, core_name: str):
        """communicateSlorインスタンスの生成

        :param host: 
        :param port: 
        :param core_name: 
        :returns: 

        """
        self.__host = host
        self.__port = port
        self.__core_name = core_name
        self.__connect = urllib3.PoolManager()

        # Uploading Data with Index Handlers | Apache Solr Reference Guide 6.6 https://solr.apache.org/guide/6_6/uploading-data-with-index-handlers.html
        self.__schema_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/schema"
        self.__document_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/update/json/docs"
        self.__commit_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/update?commit=true"
        self.__search_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/query"
        self.__morelikethis_search_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/query?mlt=true&mlt.indf=1&ml.mintf=1&mlt.fl="
        # The Terms Component | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/the-terms-component.html
        self.__terms_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/terms?terms.ttf=true&terms.stats=true"
        # Implicit RequestHandlers | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/implicit-requesthandlers.html#analysis-handlers
        self.__analysis_url = f"http://{self.__host}:{self.__port}/solr/{self.__core_name}/analysis/field?analysis.fieldtype=text_ja"

    def __post_json_to_solr(self, url: str, encoded_data: str) -> urllib3.response.HTTPResponse:
        """Solrへの通信処理

        :param url: 
        :param encoded_data: 
        :returns: 

        """
        res = self.__connect.request('POST', url, headers={'Content-Type': 'application/json'}, body=encoded_data.encode("utf-8"))
        return res

    def retrieve_analysis_result(self, field_value: str) -> Dict[str, Any]:
        """retrieve analyzed result for string

        :param field_value: 
        :returns: 

        """
        url_str = f"&analysis.fieldvalue={field_value}"
        res = self.__connect.request('GET', self.__analysis_url + url_str)
        result_d = json.loads(res.data)
        result_ana = result_d["analysis"]["field_types"]["text_ja"]["index"]
        result_ana_d = {result_ana[i]: result_ana[i + 1] for i in range(0, len(result_ana), 2)}
        return result_ana_d

    def get_fields_information(self) -> List[Dict[str, Any]]:
        """Solrの指定フィールドの情報を取得する

        :returns: 
        :remark:
        [Schema API \| Apache Solr Reference Guide 8\.9](https://solr.apache.org/guide/8_9/schema-api.html#list-fields)

        """
        res = self.__connect.request('GET', self.__schema_url + "/fields")
        fields_inf = json.loads(res.data)
        del fields_inf["responseHeader"]
        _, result_list = fields_inf.popitem()
        return result_list

    def get_one_field_information(self, field: str) -> Dict[str, Any]:
        """Solrの指定フィールドの情報を取得する

        :param field: 
        :returns: 
        :remark:
        [Schema API \| Apache Solr Reference Guide 8\.9](https://solr.apache.org/guide/8_9/schema-api.html#list-fields)

        """
        res = self.__connect.request('GET', self.__schema_url + f"/fields/{field}")
        field_inf = json.loads(res.data)
        if res.status != 200:
            mes = f"??error:commucateSolr: '--add_fields' may be required.\n{pprint.pformat(field_inf['error'])}"
            raise Exception(mes)
        result_d = field_inf["field"]
        return result_d

    def replace_field_def(self, field: str, replaced_field_def: Dict[str, Any]):
        """Solrの既存のフィールド情報を置き換える

        :param field: 
        :param replaced_field_def: 
        :returns: 

        """
        field_def = self.get_one_field_information(field)
        field_def.update(replaced_field_def)
        fd_json = json.dumps({"replace-field": field_def}, ensure_ascii=False)
        res = self.__post_json_to_solr(self.__schema_url, fd_json)
        if res.status != 200:
            res_str = pprint.pformat(res.data)
            raise Exception(f"??error:communicateSlor:add_fields_from_json: {field_def}:\nresponse=\n{res.data.decode('utf-8')}")

    def add_field(self, field_def: Dict[str, Any]):
        """Solrへフィールド情報を登録する

        :param field_def: 
        :returns: 

        """
        fd_json = json.dumps({"add-field": field_def}, ensure_ascii=False)
        res = self.__post_json_to_solr(self.__schema_url, fd_json)
        if res.status != 200:
            res_str = pprint.pformat(res.data)
            raise Exception(f"??error:communicateSlor:add_fields_from_json: {field_def}:\nresponse=\n{res.data.decode('utf-8')}")

    def add_fields_from_file(self, json_file: str) -> List[str]:
        """Solrへファイルからフィールド情報を登録する

        :param json_file: 
        :returns: 

        """
        names = []
        with open(json_file, "r") as f:
            f_defs_list = json.load(f)
            _, f_defs = f_defs_list.popitem()
            for fd in f_defs:
                self.add_field(fd)
            names = [v["name"] for v in f_defs]
        return names

    def add_document(self, doc: Dict[str, Any], commit: bool = False) -> urllib3.response.HTTPResponse:
        """Solrへの文処理登録処理

        :param doc: Solrの各フィールドをキーとする辞書
        :param commit: 
        :returns: 

        """
        json_data = json.dumps(doc, ensure_ascii=False)
        res = self.__post_json_to_solr(self.__document_url, json_data)

        if res.status == 200 and commit:
            res = self.commit()
        return res

    def commit(self) -> urllib3.response.HTTPResponse:
        """Solrへの文処理登録処理のコミット要求発行

        :returns: 

        """
        res = self.__connect.request('GET', self.__commit_url)
        return res

    def escape_query_string(self, query: str) -> str:
        """Solr用Queryのエスケープ処理

        :param query: 
        :returns: 

        """
        query = re.sub(r"(?<!\\)([\+\-!\(\){}\[\]^\"~*?:/])", r"\\\1", query)
        quey = re.sub(r"(?<!\\)(&&|\|\|)", r"\\\1", query)
        return query

    def search(self,
               field: str,
               sentences: List[str],
               detail: List[str] = None,
               full_match: bool = False,
               operator: str = "OR",
               limit: int = 50,
               morelikethis: bool = False) -> urllib3.response.HTTPResponse:
        """Solr用検索処理

        :param field: 
        :param sentences: 
        :param detail: 
        :param full_match: 
        :param operator: 
        :param limit: 
        :param morelikethis: 
        :returns: 

        """
        # JSON Request API | Apache Solr Reference Guide 7.7 https://solr.apache.org/guide/7_7/json-request-api.html#json-request-api
        if limit == 0:
            query = {"fields": "*,score", "sort": "score desc"}
        else:
            query = {"fields": "*,score", "limit": f"{limit}", "sort": "score desc"}
        qs = []
        for sts in sentences:
            sts = f'"{sts}"' if sts.find(" ") != -1 else sts
            qs.append(f'{field}:{sts}')
        if detail is not None:
            qs.extend(detail)
        query["query"] = f" {operator} ".join(qs)
        pprint.pprint(query)
        json_data = json.dumps(query)
        res = self.__post_json_to_solr((self.__morelikethis_search_url + field) if morelikethis else self.__search_url, json_data)

        if res.status != 200:
            res_str = pprint.pformat(res.data)
            raise Exception(f"??error:communicateSlor:search: {field}:\nresponse=\n{res.data.decode('utf-8')}")

        return res

    def retrieve_terms_status(self, field: str, limit: int = 100) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Solr登録済み文書の語句頻度情報の取得

        :param limit: 
        :returns: 

        """
        res = self.__connect.request('GET', self.__terms_url + f"&terms.limit={limit}&terms.fl={field}")
        terms_inf = json.loads(res.data)
        if terms_inf["responseHeader"]["status"] != 0:
            mes = f"??error:communicateSlor:get_terms_status: can not get response from solr"
            raise Exception(mes)
        del terms_inf["responseHeader"]
        result_dic = terms_inf
        # terms_df = pd.DataFrame(columns=["term", "df", "ttf"])
        terms_list = result_dic["terms"]["content"]
        terms_dic = {terms_list[i]: terms_list[i + 1] for i in range(0, len(terms_list) - 1, 2)}
        # terms_dic={ "吾輩":{"df":99,"ttf":208}}
        # result_dic={ "terms":{ "content":[ "吾輩",{"df":99,"ttf":208}]}, "indexstats":{"numDocs":454}}

        return terms_dic, result_dic


def init_solr(solr_core_name: str):
    """solrの初期化用スクリプト及びフィールド定義用のテンプレートの作成

    :param solr_core_name: コア名
    :returns: 

    """
    sh_file_name = "solr_init.sh"
    def_file_name = "solr-fields.json"
    sh_str = f"""
#/bin/bash

# -- following environment must be defined.
SOLR_DATADIR=
SOLR_HOME=
SOLR_CORE_NAME={solr_core_name}
# --

if [ ! -e ${{SOLR_DATADIR}} ]; then
    mkdir  ${{SOLR_DATADIR}}
fi
cp ${{SOLR_HOME}}/server/solr/solr.xml ${{SOLR_DATADIR}}
echo "-- start solr server" 1>&2
${{SOLR_HOME}}/bin/solr start -d ${{SOLR_HOME}}/server -s ${{SOLR_DATADIR}}
echo "-- create core:${{SOLR_CORE_NAME}}" 1>&2
${{SOLR_HOME}}/bin/solr create_core -c ${{SOLR_CORE_NAME}}
echo "-- stop solr server" 1>&2
${{SOLR_HOME}}/bin/solr stop

echo "${{SOLR_HOME}}/bin/solr start -d ${{SOLR_HOME}}/server -s ${{SOLR_DATADIR}}" > solr_start.sh

cat <<EOF 1>&2
================================
-- ${{SOLR_CORE_NAME}} was created.

you must start solr server by following.
solr_start.sh
   
EOF
"""
    def_str = """
/* Field Properties by Use Case | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/field-properties-by-use-case.html */
/* Field Type Definitions and Properties | Apache Solr Reference Guide 8.9 https://solr.apache.org/guide/8_9/field-type-definitions-and-properties.html */

{
    "fields":[
    {
      "name":"date",
      "type":"pdate",
      "uninvertible":true,
      "indexed":true,
      "required":true,
      "stored":true,
      "multiValued": false,
      "large": false
    },
    {
      "name":"content",
      "type":"text_ja",
      "uninvertible":true,
      "indexed":true,
      "required":true,
      "stored":true,
      "multiValued": false,
      "large": false
    },
    {
      "name":"misc",
      "type":"text_general",
      "uninvertible":true,
      "indexed":true,
      "required":true,
      "stored":true,
      "multiValued": false,
      "large": false
    },
]}
"""
    with open(sh_file_name, "w") as f:
        print(sh_str, file=f)
    with open(def_file_name, "w") as f:
        print(def_str, file=f)

    print(f"%inf:csv_text_solr:make script and definition of fields:{sh_file_name},{def_file_name}", file=sys.stderr)


def get_field_def(fl_name: str, fl_type: str, multivalued: bool = False, large: bool = False) -> Dict[str, Any]:
    """フィールド定義のDictデータを生成する

    :param fl_name: フィールド名
    :param fl_type: フィールド型
    :param multivalued: 配列型とする場合はTrueを定義
    :param large: テキストで512KBを超える場合はTrueとするほうがよい、ただしmultivaluedは Falseである必要がある
    :returns: 

    :remark:
    'large'については以下を参照のこと
       https://solr.apache.org/guide/8_9/field-type-definitions-and-properties.html#:~:text=true-,large,-Large%20fields%20are

    """
    if multivalued and large:
        raise Exception("??error:csv_text_solr:get_field_def: multivalued=True must be with large=False")

    result_d = {
        'name': fl_name,
        'type': fl_type,
        'uninvertible': True,
        'indexed': True,
        'required': True,
        'stored': True,
        "multiValued": multivalued,
        "large": large
    }

    return result_d


def check_fields_definition(fields_info: List[Dict[str, str]],
                            fields: Optional[List[str]],
                            fields_type: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """既存のsolrと使用したいフィルード定義を比較、不足分の定義を生成する
    solrのフィールド定義Dictと指定フィールド定義を比較し、追加が必要なフィールドの定義D辞書を作成する

    :param fields_info: 既存solrのフィールド定義リスト
    :param fields: 使用したいフィールド名リスト
    :param fields_type: フィールドの型定義の辞書
    :returns: 追加フィールド定義の辞書のリスト

    """
    # fields_info= [{'name': '_nest_path_', 'type': '_nest_path_'}, {'name': '_root_', 'type': 'string', 'docValues': False, 'indexed': True, 'stored': False}]

    add_defs: List[Dict[str, Any]] = []  # 追加すべきフィールドの定義リスト
    fls = [v["name"] for v in fields_info]  # フィールドの名前リスト
    fls_type = {v["name"]: v["type"] for v in fields_info}  # フィールドのタイプリスト
    for fl in fields:
        if fl not in fls:
            # print(f"#warn:csv_text_solr:check_fields_definitions:{fl} was not found", file=sys.stderr)
            if fields_type is not None and fl in fields_type.keys():
                fl_type = fields_type[fl]
            else:
                fl_type = "text_general"
            add_defs.append(get_field_def(fl, fl_type))
            if fl_type == "pdate":
                # 時刻情報の場合はオリジナルを追加するフィールドを追加する
                add_defs.append(get_field_def(f"{fl}_local", "text_general"))
        elif fl in fls and fields_type is not None and fl in fields_type.keys() and fields_type[fl] != fls_type[fl]:
            print(
                "#warn:csv_text_solr:check_fields_definitions:{fl} was already exist, so '{fls_type[fl]}' is used instead of given type:'{fields_type[fl]}'.",
                file=sys.stderr)

    return add_defs


def store_to_solr(com_solr: communicateSlor,
                  df: pd.DataFrame,
                  column_map: Dict[str, Any],
                  fields_type: Dict[str, str],
                  datetime_fmt: str,
                  key_name: str,
                  extended_columns: List[str],
                  remove_pattern_c=None,
                  replace_pattern_c=None,
                  min_length=2,
                  user_dict=None,
                  remove_words=None,
                  extend_words=None) -> int:
    """DataFrameの内容をSolrに登録する

    :param com_solr: communicateSolrクラスのインスタンス
    :param df: 登録するDataFrame
    :param column_map: DataFrameのカラムとSolrのフィールドとの対応辞書
    :param fields_type: Solrのフィールドの型定義辞書
    :param datetime_fmt: DataFrame中の時刻情報カラムの書式
    :param key_name: DataFrame中のKeyとなるカラム名、Solrのidフィールドへ割り当てられる
    :param extended_columns: 文の分解、置き換えを行い拡張型とするDataFrameのカラム名
    :param remove_pattern_c: 
    :param replace_pattern_c: 
    :param min_length: 
    :param user_dict: 
    :param remove_words: 
    :param extend_words: 
    :returns: 

    """
    key_ds = None
    if key_name is not None and key_name in df:
        key_ds = df[key_name]

    # pre-process for each column
    for cname in df:
        if cname in column_map:
            fname = column_map[cname]
            if fname in fields_type:
                f_type = fields_type[fname]
            else:
                f_type = "text_general"
            if f_type == "pfloat":
                df[cname] = df[cname].astype(float)
            elif f_type == "pint":
                df[cname] = df[cname].astype("int64")
            elif f_type == "pdate":
                # 時刻情報のオリジナル(ローカル版)を保存しておく
                df[f"{cname}_local"] = df[cname]
                df[cname] = df[cname].apply(lambda x: datetime.datetime.strptime(x, datetime_fmt).astimezone(datetime.timezone.utc).
                                            strftime("%Y-%m-%dT%H:%M:%SZ"))
    nrec = 0
    npts = len(df)
    for ir in range(npts):
        if ir % max(10, npts / 100) == 0:
            print(f"%inf:csv_text_solr:processed recodes: {ir}", file=sys.stderr)
        nrec += 1
        rec = dict(df.iloc[ir])
        rec_d = {}
        # if key_name is not None and key_name in rec:
        #     rec_d["id"] = rec[key_name]
        if key_ds is not None:
            rec_d["id"] = key_ds.loc[ir]

        for cname in column_map:
            fname = column_map[cname]
            if extended_columns is not None and cname in extended_columns:
                trimed_sentence = trim_sentence(rec[cname],
                                                remove_pattern_c=remove_pattern_c,
                                                replace_pattern_c=replace_pattern_c,
                                                min_length=min_length,
                                                user_dict=user_dict,
                                                remove_words=remove_words,
                                                extend_words=extend_words)
                ex_sts = " ".join(trimed_sentence) if isinstance(trimed_sentence, list) else trimed_sentence
                # rec_d[cname + EXTENDED_PREFIX] = ex_sts
                rec_d[fname] = [rec[cname], ex_sts]
            else:
                rec_d[fname] = rec[cname]
            if fname in fields_type and fields_type[fname] == "pdate":
                # 時刻情報のローカル版をコピーする
                rec_d[f"{fname}_local"] = rec[f"{cname}_local"]

        # -- to solr
        try:
            res = com_solr.add_document(rec_d)
        except Exception as e:
            rec_str = pprint.pformat(rec_d)
            print(f"#warn:csv_text_solr:exception was occured during processing:\n" + rec_str + "\n" + str(e), file=sys.stderr)

        if res.status != 200:
            mes = f"??error:csv_text_solr:store_to_solr:error response: {res.data.decode('utf-8')}"
            raise Exception(mes)

    com_solr.commit()

    return nrec


def trim_sentence(sentence,
                  remove_pattern_c=None,
                  replace_pattern_c=None,
                  min_length=2,
                  user_dict=None,
                  remove_words=None,
                  extend_words=None) -> Union[List[str], str]:
    """入力文を語句の削除、置き換え等を実施し、分かち書き型の分解を行う

    :param sentence: 入力文
    :param remove_pattern_c: removeSentencesクラスのインスタンス
    :param replace_pattern_c: replaceSentencesクラスのインスタンス
    :param min_length: 語句として残す最小の文字列長さ
    :param user_dict: Mecabのユーザ辞書ファイルのパス
    :param remove_words: 分かち書き後に除去する語句のリスト
    :param extend_words: 分かち書き後の各語句に対する追加国辞書
    :returns: 語句を空白で連結した文字列


    """
    sentence = remove_pattern_c.do(sentence) if remove_pattern_c is not None else sentence
    sentence = replace_pattern_c.do(sentence) if replace_pattern_c is not None else sentence
    trimed_sentence = get_words_0(sentence,
                                  min_length=min_length,
                                  debug=False,
                                  user_dict=user_dict,
                                  remove_words=remove_words,
                                  extend_words=extend_words)
    return trimed_sentence


def add_fields_def_from_file(fields_def_file: str) -> List[str]:
    """JSONファイルからSolrのフィールド定義を読み込みSolrに登録する

    :param fields_def_file: JSONファイル
    :returns: 追加されたフィールド名のリスト

    """
    f_names = csolr.add_fields_from_file(fields_def_file)
    fields_info = csolr.get_fields_information()

    fl_names = [v["name"] for v in fields_info]
    for fd in fields_info:
        if fd["type"] == "pdate" and fd["name"] in f_names:
            fl_name = f"{fd['name']}_local"
            if fl_name not in fl_names:
                fl_def = get_field_def(fl_name, "text_general")
                csolr.add_field(fl_def)

    return f_names


if __name__ == "__main__":
    args = init()

    core_name = args.CORE
    if args.INIT_SAMPLE:
        init_solr(core_name)
        sys.exit(0)

    host = args.HOST
    port = args.PORT
    key_name = args.KEY
    csv_files = args.csv_files

    fields_def_file = args.FIELDS
    fields_type_s = args.FIELDS_TYPE
    show_fields_inf = args.FIELDS_INF
    column_map_s = args.CMAP
    auto_add_fields = args.FIELDS_ADD

    datetime_fmt = args.DATETIME_FMT
    extended_column_s = args.EX_COLUMN

    user_dict = args.UDICT
    remove_pattern = args.RPATTERN
    replace_pattern_file = args.REP_FILE
    extend_word_file = args.EXT_FILE
    extend_words = read_extend_word(extend_word_file)
    min_length = args.MLENGTH
    remove_words = args.RWORDS

    ret_terms = args.RET_TERMS
    ana_terms = args.ANA_TERMS

    csolr = communicateSlor(host, port, core_name)
    remove_pattern_c = removeSentences(remove_pattern) if len(remove_pattern) else None
    replace_pattern_c = replaceSentences(replace_pattern_file) if len(replace_pattern_file) else None

    # ==== retrieve df(docRef) and ttf(totalTermFreq) about terms from solr, TOP100 ====
    if ret_terms is not None:
        field = ret_terms
        print(f"%inf:csv_text_solr:retrive terms component from solr:", file=sys.stderr)
        terms_dic, df_ttf = csolr.retrieve_terms_status(field, limit=100)
        terms_df = pd.DataFrame(columns=["term", "df", "ttf"])
        for w, rec in terms_dic.items():
            rec.update({"term": w})
            terms_df = terms_df.append(rec, ignore_index=True)

        terms_df.to_csv(sys.stdout, index=False)
        sys.exit(0)
    # ==== end of retrieve df(docRef) and ttf(totalTermFreq) about terms from solr, TOP100 ====

    # ==== retrieve analyzed terms ====
    if ana_terms is not None:
        print(f"%inf:csv_text_solr:retrive analyzed result of string:", file=sys.stderr)
        result_d = csolr.retrieve_analysis_result(ana_terms)
        # pprint.pprint(result_d)
        for k, dl in result_d.items():
            print(k, [v["text"] for v in dl])
        sys.exit(0)

    # ==== end of retrieve analyzed terms ====

    # ==== add new fileds into solr from file ====
    if fields_def_file is not None:
        # f_names = csolr.add_fields_from_file(fields_def_file)
        f_names = add_fields_def_from_file(fields_def_file)
        print(f"%inf:csv_text_solr:add field definitions from {fields_def_file}:\n" + F_indent_text(f"{f_names}"), file=sys.stderr)
        sys.exit(0)
    # ==== end add new fields into solr from file ====

    # ==== options ====
    # -- get information of fields from solr
    fields_info = csolr.get_fields_information()
    # fields_info= [{'name': '_nest_path_', 'type': '_nest_path_'}, {'name': '_root_', 'type': 'string', 'docValues': False, 'indexed': True, 'stored': False}]
    if show_fields_inf:
        fields_info = [v for v in fields_info if not v["name"].startswith("_")]
        print(f"%inf:csv_text_solr:show_fields_informaiton: number of fields={len(fields_info)}", file=sys.stderr)
        pprint.pprint(fields_info)
        sys.exit(0)

    # -- triming fields type
    fields_type = {}
    if fields_type_s is not None:
        for v in re.split(r"\s*(?<!\\),\s*", fields_type_s):
            cs = re.split(r"\s*(?<!\\):\s*", v)
            fields_type[cs[0]] = cs[1]
    for fl in fields_info:
        if fl["name"] not in fields_type:
            fields_type[fl["name"]] = fl["type"]

    fields_type = {k: fields_type[k] for k in fields_type if not k.startswith("_")}
    print(f"%inf:csv_text_solr:type of fileds:\n" + F_indent_text(f"{fields_type}"), file=sys.stderr)

    # -- column map
    column_map = {}
    if column_map_s is not None:
        for v in re.split(r"\s*(?<!\\),\s*", column_map_s):
            cs = re.split(r"\s*(?<!\\):\s*", v)
            column_map[cs[0]] = cs[1]
        print(f"%inf:csv_text_solr:column_map:\n" + F_indent_text(f"{column_map}"), file=sys.stderr)
    else:
        print(f"#warn:csv_text_solr:'--column_map' was not defined, so column names in csv will be used as field names.", file=sys.stderr)

    # -- check fields and add new fields if need.
    if len(column_map) > 0:
        add_fields_defs = check_fields_definition(fields_info,
                                                  list(column_map.values()) if len(column_map) != 0 else None,
                                                  fields_type if len(fields_type) > 0 else None)
        if len(add_fields_defs) > 0 and auto_add_fields:
            print(f"%inf:csv_text_solr:fields to add:\n" + F_indent_text(f"{add_fields_defs}"), file=sys.stderr)
            for fl_def in add_fields_defs:
                csolr.add_field(fl_def)
        elif len(add_fields_defs) > 0:
            print(f"??error:csv_text_solr:insuffcient fields: {[ v['name'] for v in add_fields_defs]}", file=sys.stderr)
            sys.exit(1)
        fields_info = csolr.get_fields_information()

    # -- 'extended column'
    extended_columns = []
    if extended_column_s is not None:
        extended_columns = re.split(r"\s*(?<!\\),\s*", extended_column_s)
        fls = [v["name"] for v in fields_info]
        e_names = []
        # for fd in [get_field_def(f"{v}{EXTENDED_PREFIX}", "text_ja") for v in extended_columns if f"{v}{EXTENDED_PREFIX}" not in fls]:
        #     csolr.add_field(fd)
        #     e_names.append(fd["name"])
        for ec in extended_columns:
            csolr.replace_field_def(ec, {"multiValued": True})
            e_names.append(ec)
        print(f"%inf:csv_text_solr: extending fields by multiValued: {e_names}", file=sys.stderr)

    if len(csv_files) == 0:
        print(f"#warn:csv_text_solr:no csv_files", file=sys.stderr)
        sys.exit(0)

    # ==== end of options =====

    # ==== store documents from csv to solr.
    total_recs = 0
    for csv_file in csv_files:
        print(f"%inf:csv_text_colr:add document: processing {csv_file}...", file=sys.stderr)
        csv_df = pd.read_csv(csv_file, dtype="string")
        if len(column_map) > 0:
            c_map = column_map
        else:
            c_map = {v: v for v in csv_df}
        nrec = store_to_solr(csolr,
                             csv_df,
                             c_map,
                             fields_type,
                             datetime_fmt,
                             key_name,
                             extended_columns,
                             remove_pattern_c=remove_pattern_c,
                             replace_pattern_c=replace_pattern_c,
                             min_length=min_length,
                             user_dict=user_dict,
                             remove_words=remove_words,
                             extend_words=extend_words)
        print(f"                                 {nrec} from {csv_file}", file=sys.stderr)
        total_recs += nrec

    print(f"%inf:csv_text_colr: number of stored records={total_recs}", file=sys.stderr)
