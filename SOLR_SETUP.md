<!-- File: SOLR_SETUP.md                      -->
<!-- Description:               -->
<!-- Copyright (C) 2021 by m.na.akei   -->
<!-- Time-stamp: "2021-08-14 11:02:42" -->

# Solrの立ち上げ  #

[Welcome to Apache Solr \- Apache Solr](https://solr.apache.org/)の立ち上げについて、
作成済みのPythonスクリプトを用いて行う手順を示す。

# Solrの解凍
Solrの[サイト](https://solr.apache.org/)からパッケージをダウンロードし、解凍する。

```bash
cd ~/
mkdir ~/.local
tar zxvf solr-x.x.x.tgz
```
# SolrのCoreの生成

続いてSolrのコアと言われるドキュメント登録単位を作成する。

検索用データを作成するディレクトリをSolr本体とは別の場所に作成することを考える。

以下では`csv_text_solr/py`を用いて初期化スクリプトとフィールド定義サンプルを生成し、
コア名を`sample_core`とし、検索用データディレクトリを`~/data/solr`に変更して用いている。


```bash
mkdir -p ~/data/solr
cd ~/data/solr

csv_text_solr.py --init_sample --core sample_core

# -- edit solr_init.sh
# SOLR_HOME=~/.local/solr-x.x.x
# SOLR_DATADIR=~/data/solr
vi solr_init.sh

# -- execute solr_init.sh to create core
bash solr_init.sh

```

## Solrのフィールドの作成

ドキュメントの登録に先立ち、Solrにドキュメントの情報をどういう形で登録るかを決めて、その内容をSolrに登録する必要がある。


フィールド定義は前項の`csv_text_solr.py`の実行で生成された`solr-fields.json`をベースに作成する。
フィールド定義は[Field Type Definitions and Properties](https://solr.apache.org/guide/8_9/field-type-definitions-and-properties.html)と
[Field Properties by Use Case](https://solr.apache.org/guide/8_9/field-properties-by-use-case.html)
を参考に作成すれば良い(参照:[SolrをPythonから使う](https://omoitukidetukuttemiru.blogspot.com/2021/08/solrpython.html))。

作成されたJSONファイル(以下の例では`solr-fields.json`)を`csv_text_solr.py`により、ローカル起動したSolrに登録する。
なお、以下の例の`solr_start.sh`は前項の`solr_init.sh`で生成されたもので、Solrをローカル起動するスクリプトである。


```bash
bash solr_start.sh
csv_text_solr.py --add_fields solr-fields.json --core sample_core
```
例えば、`solr-fields.json`は以下の様になる。

```json
{
    "fields":[
    {
      "name":"content",
      "type":"text_ja",
      "uninvertible":true,
      "indexed":true,
      "required":true,
      "stored":true},
    {
      "name":"date",
      "type":"pdate",
      "uninvertible":true,
      "indexed":true,
      "required":true,
      "stored":true}]}
```

## ドキュメントの登録

CSV形式のドキュメント情報であれば、`csv_text_solr.py`を使って以下のように登録する。

```bash
csv_text_solr.py --core sample_core --datetime_format="%Y-%m-%d" --key=date wagahaiwa_nekodearu.csv
```
その他の形式であれば
[SolrをPythonから使う](https://omoitukidetukuttemiru.blogspot.com/2021/08/solrpython.html)を参考にPythonスクリプトから登録する。


## ドキュメントの検索

ドキュメントの検索は`csv_text_solr_search.py`を用いれば、スコア値付きのCSV形式で取得できる。

```bash
csv_text_solr_search.py --core wagahaiwa_nekodearu --search_field content --search "吾輩は猫である。名前はまだ無い"
```

それ以外の形式で結果のを得たいなら 
[SolrをPythonから使う -検索編](https://omoitukidetukuttemiru.blogspot.com/2021/08/solrpython_9.html)
を参考にPythonスクリプトを作成すればよい。


<!-- ------------------ -->
<!-- Local Variables:   -->
<!-- mode: markdown     -->
<!-- coding: utf-8-unix -->
<!-- End:               -->
