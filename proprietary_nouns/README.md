<!-- File: 00_README.md                      -->
<!-- Description:               -->
<!-- Copyright (C) 2021 by m.na.akei   -->
<!-- Time-stamp: "2021-04-11 11:57:13" -->

## 装置名等の固有名詞を抽出し、辞書化する

### 固有名詞候補抽出

固有名詞候補の抽出は`search_propr_nouns.py`を用いて行う。

`search_propr_nouns.py`のオプションの使い方は以下のとおりである。


- 固有名詞となる装置名等で、`〇〇装置1`の形式で数字まで含めて固有名詞として扱う場合は、`--with_number`オプションを指定する。
- 固有名詞となる装置名等で、`〇〇装置-A`の形式で記号まで含めて固有名詞として扱う場合は、`--with_symbol`オプションを指定する。
- 固有名詞となる装置名等で、`1〇〇装置`の形式で数字で始まるものも固有名詞として扱う場合は、`--start_with_number`オプションを指定する。


### 手順
1. 以下のコマンドで固有名詞候補を抽出する。

	search_propr_nouns.py input.txt > propr_nouns.csv
	
2. `propr_nouns.csv`をExcel等で開いて、固有名詞を判別しCheck欄に1を記入し、読み候補欄を修正する。
3. Check欄で絞り込んで固有名詞候補CSVを作成する

	csv_query.py propr_nouns.csv "Check==1" > recomended_nouns.csv
  
3. 固有名詞候補CSVからで辞書用CSVを作成する

	make_mecab_dict.py recomended_nouns.csv 語句 読み候補 > dictionary.csv

4. Mecab用ユーザー辞書を作成する

	make_mecab_dict.sh dictionary.csv user.dic /var/lib/mecab/dic/ipadic-utf8/

5. 確認のため再度固有名詞候補を抽出する。

	python search_propr_nouns.py --user_dict user.dic input.txt > propr_nouns_2.csv

6. `propr_nouns.csv`の中身を確認して、必要ならばステップを繰り返す。


## 文章から語句の表を抽出する

### 手順

1. `extract_words_table.py`を用いて使用語句のCSVを生成する

	python extract_words_table.py input.txt > words_list.csv
	
2. 生成されるCSVは、MeCabの出力内容に加えて、`出現数`、`出現長`を列として含んでいる。  
   出現数は入力された文章中の語句の数、出現長は語句の文字数を示している。

`csv_query.py`生成されたCSVから品詞等を絞り込んで抽出することができる。

	例) csv_query.py words_list.csv "品詞=='名詞' and 品詞細分類1=='固有名詞'"


<!-- ------------------ -->
<!-- Local Variables:   -->
<!-- mode: markdown     -->
<!-- coding: utf-8-unix -->
<!-- End:               -->
