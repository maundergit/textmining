## csv_text_normalize.py
<pre>
usage: csv_text_normalize.py [-h] [-v]
                             [--unicode_normalized_mode {NFD,NFC,NFCD,NFKC}]
                             [--with_neologdn] [--for_mecab FOR_MECAB]
                             [--without_kansuji] [--no_header] [--quote_all]
                             [--include_columns COLUMNS[,COLUMNS,[COLUMNS,...]]]
                             [--exclude_columns COLUMNS[,COLUMNS,[COLUMNS,...]]]
                             [--include_pattern REGEX]
                             [--exclude_pattern REGEX] [--output STR]
                             FILE

positional arguments:
  FILE                  files to read, if empty, stdin is used

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --unicode_normalized_mode {NFD,NFC,NFCD,NFKC}
                        mode of unicode normalization, default=NFKC
  --with_neologdn       processing with neologdn
  --for_mecab FOR_MECAB
                        MECABで記号とする半角記号文字列,ex:'()[]{}'
  --without_kansuji     漢数字の半角数字への変換を行わない
  --no_header           一行目からデータとして処理する
  --quote_all           全カラムをQuoteする
  --include_columns COLUMNS[,COLUMNS,[COLUMNS,...]]
                        names of colunmns to process, default=all
  --exclude_columns COLUMNS[,COLUMNS,[COLUMNS,...]]
                        names of colunmns to exclude
  --include_pattern REGEX
                        name pattern of colunmns to process
  --exclude_pattern REGEX
                        name pattern of colunmns to exclude
  --output STR          path of output, default is stdout

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


</pre>
## csv_text_solr.py
<pre>
usage: csv_text_solr.py [-h] [-v] [--init_sample] [--host IP_ADDRESS]
                        [--port PORT] --core NAME [--key COLUMN]
                        [--column_map COLUMN:FIELD[,COLUMN:FIELD...]]
                        [--auto_add_fields] [--add_fields JSON_FILE]
                        [--show_fields_information]
                        [--fields_type FIELD:TYPE[,FIELD:TYPE..]]
                        [--datetime_format datetime_format]
                        [--extend_columns FIELD[,FIELD..]]
                        [--user_dictionary FILE] [--minmum_length_of_word INT]
                        [--remove_words WORDS[,WORD,...]]
                        [--remove_pattern REGEX_OR_FILE]
                        [--replace_pattern_file FILE]
                        [--extend_word_file FILE]
                        [--retrieve_terms_status FIELD]
                        [--analysis_terms ANA_TERMS]
                        [CSV_FILE [CSV_FILE ...]]

positional arguments:
  CSV_FILE              csv files to read. if empty, stdin is used

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --init_sample         make sample script and definition of fields
  --host IP_ADDRESS     ip address of host, default=127.0.0.1
  --port PORT           port number, default=8983
  --core NAME           name of core of solr
  --key COLUMN          name of column as key
  --column_map COLUMN:FIELD[,COLUMN:FIELD...]
                        mapping table between column and field
  --auto_add_fields     to add fields, that are in column map but not in solr,
                        when '--column_map' is used..
  --add_fields JSON_FILE
                        path of json file to define fields
  --show_fields_information
                        retrieve information of fields
  --fields_type FIELD:TYPE[,FIELD:TYPE..]
                        list of type of each field, those was used to convert
                        data type when adding ones to solr.
  --datetime_format datetime_format
                        format of datetime in columns whose type is 'pdate' in
                        '--fields_type', default='%Y-%m-%d %H:%M:%S'
  --extend_columns FIELD[,FIELD..]
                        decompose sentence and extend words
  --user_dictionary FILE
                        path of user dictionary
  --minmum_length_of_word INT
                        minimum length of word, default=2
  --remove_words WORDS[,WORD,...]
                        list of words to remove, default='。,、,？,.,\,,?'
  --remove_pattern REGEX_OR_FILE
                        regex pattern to remove before analyzing or file
  --replace_pattern_file FILE
                        path of file that has regex pattern to replace before
                        analyzing
  --extend_word_file FILE
                        path of file that has regex pattern and word to add at
                        deriving words
  --retrieve_terms_status FIELD
                        retrieve df(docFreq) and ttf(totalTermFreq) about
                        terms from solr, top100, with csv format
  --analysis_terms ANA_TERMS
                        retrieve analyzed result for string

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
  csv_text_solr.py --core wagahaiwa_nekodearu --datetime_format="%Y-%m-%d" --key=date --extend_columns=content \
                   --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt wagahaiwa_nekodearu.csv

  csv_text_solr.py --core wagahaiwa_nekodearu --retrieve_terms


</pre>
## csv_text_solr_search.py
<pre>
usage: csv_text_solr_search.py [-h] [-v] [--host IP_ADDRESS] [--port PORT]
                               --core NAME [--extend_columns FIELD[,FIELD..]]
                               [--user_dictionary FILE]
                               [--minmum_length_of_word INT]
                               [--remove_words WORDS[,WORD,...]]
                               [--remove_pattern REGEX_OR_FILE]
                               [--replace_pattern_file FILE]
                               [--extend_word_file FILE]
                               [--search STR [STR ...]] [--search_field FIELD]
                               [--search_ex STR [STR ...]]
                               [--search_limit INT]
                               [--search_operator {OR,AND}]
                               [--search_detail FIELD:KEYWORD[,FIELD:KEYWORD..]]
                               [--search_detail_file FILE]
                               [--search_output CSV_FILE]
                               [CSV_FILE [CSV_FILE ...]]

positional arguments:
  CSV_FILE              csv files to read. if empty, stdin is used

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --host IP_ADDRESS     ip address of host, default=127.0.0.1
  --port PORT           port number, default=8983
  --core NAME           name of core of solr
  --extend_columns FIELD[,FIELD..]
                        decompose sentence and extend words
  --user_dictionary FILE
                        path of user dictionary
  --minmum_length_of_word INT
                        minimum length of word, default=2
  --remove_words WORDS[,WORD,...]
                        list of words to remove, default='。,、,？,.,\,,?'
  --remove_pattern REGEX_OR_FILE
                        regex pattern to remove before analyzing or file
  --replace_pattern_file FILE
                        path of file that has regex pattern to replace before
                        analyzing
  --extend_word_file FILE
                        path of file that has regex pattern and word to add at
                        deriving words
  --search STR [STR ...]
                        sentence(s) to search
  --search_field FIELD  field to search
  --search_ex STR [STR ...]
                        sentence(s) to search with exntending by '--
                        user_ditctionary','--remove_words','--
                        remove_pattern','--replace_pattern_File','--
                        extended_word_file'
  --search_limit INT    limit of search results, default=50
  --search_operator {OR,AND}
                        operator to search, default='OR'
  --search_detail FIELD:KEYWORD[,FIELD:KEYWORD..]
                        detail search for each field
  --search_detail_file FILE
                        path of file that have detail search queriesfor each
                        field
  --search_output CSV_FILE
                        path of csv file to store result, default=stdout

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
  csv_text_solr_search.py --core wagahaiwa_nekodearu --extend_word_file=tfidf_extend_word.txt --search_field=content --search "主人" "学生" \
                   --search_ex "よく主人の所へ遊びに来る馬鹿野郎"
  csv_text_solr_search.py --core wagahaiwa_nekodearu --search_detail="date_local:"01-12""
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


</pre>
## csv_text_tfidf.py
<pre>
usage: csv_text_tfidf.py [-h] [-v] [--index COLUMN]
                         [--additional_columns COLUMN[,COLUMN...]]
                         [--user_dictionary FILE]
                         [--minmum_length_of_word INT]
                         [--remove_words WORDS[,WORD,...]]
                         [--remove_pattern REGEX_OR_FILE]
                         [--replace_pattern_file FILE]
                         [--extend_word_file FILE]
                         [--output_mode {simple,json,dot}] [--only_learn FILE]
                         [--model_file FILE]
                         [--sequential_learn IN_FILE,OUT_FILE]
                         [--dot_cut_off FLOAT] [--check_frequency FLOAT]
                         [--use_tf] [--use_idf] [--status] [--debug]
                         FILE COLUMN

positional arguments:
  FILE                  csv file to read, if empty, stdin is used
  COLUMN                column nameto process

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --index COLUMN        column name for index, default=row number
  --additional_columns COLUMN[,COLUMN...]
                        list of column names for additional columns into
                        simlarity table, default=None
  --user_dictionary FILE
                        path of user dictionary
  --minmum_length_of_word INT
                        minimum length of word, default=2
  --remove_words WORDS[,WORD,...]
                        list of words to remove, default='。,、,？,.,\,,?'
  --remove_pattern REGEX_OR_FILE
                        regex pattern to remove before analyzing or file
  --replace_pattern_file FILE
                        path of file that has regex pattern to replace before
                        analyzing
  --extend_word_file FILE
                        path of file that has regex pattern and word to add at
                        deriving words
  --output_mode {simple,json,dot}
                        format of output, default=simple
  --only_learn FILE     path of file to store model
  --model_file FILE     path of file to load model
  --sequential_learn IN_FILE,OUT_FILE
                        path of file that has result of learned model and new
                        one
  --dot_cut_off FLOAT   threshold for cutting off, only available with '--
                        output_mode=dot'
  --check_frequency FLOAT
                        ratio for checking frequency of words, only available
                        with '--debug', default=0.2
  --use_tf              use term frequency
  --use_idf             use inverse document frequency
  --status              print status
  --debug               debug mode

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
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0.3  --use_tf - content |\
      perl -ne "if(/label="([.\d]+)"/ && \$1>${THR}){s/\]/ color="red"]/} print;" > test.dot
  head wagahaiwa_nekodearu.csv | csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 --remove_pattern='《[^》]*》' - content > test.dot

  cat <<EOF > rep_pat.txt
  s/《[^》]*》//
  s/(?i)GHI/KLM/
  EOF
  head -40 wagahaiwa_nekodearu.csv |\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 --check_frequency=0.1 --replace_pattern_file=rep_pat.txt --debug - content> test.dot

  cat <<EOF > tfidf_extend_word.txt
  # regexp word1,word2,...
  書生	学生,学校
  原$	野原
  EOF
  head -40 wagahaiwa_nekodearu.csv |\
       csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \
                         --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt - content> test.dot

  # using learned model
  csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \
                    --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --only_learn=tfidf.db wagahaiwa_nekodearu.csv content
  echo 吾輩《わがはい》は猫である。名前はまだ無い。 |\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \
                        --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --model_file=tfidf.db wagahaiwa_nekodearu.csv content

  # sequential learning
  echo -e "date,content\n2021-07-22,引き止めて、一ヶ月経って、裏の書斎にこもっている。" > new_data.csv
  csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \
                    --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --sequential_learn=tfidf.db,tfidf2.db new_data.csv content
  echo -e "引き止めて、一ヶ月経って、裏の書斎にこもっている。" |\
      csv_text_tfidf.py --index date --output_mode dot --dot_cut_off=0 \
                        --replace_pattern_file=rep_pat.txt --extend_word_file=tfidf_extend_word.txt --model_file=tfidf2.db wagahaiwa_nekodearu.csv content


</pre>
## ja_depndency_analysis.py
<pre>
usage: ja_depndency_analysis.py [-h] [-v] [--word WORD] [--read_csv] [--print]
                                [--print_svg SVG_FILE] [--print_dot DOT_FILE]
                                [--print_subject TEXT_FILE] [--output FILE]
                                FILE

positional arguments:
  FILE                  file to read

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --word WORD           word to make dependency
  --read_csv            read csv file as analized result
  --print               printdisplacy as text
  --print_svg SVG_FILE  path of displacy as svg
  --print_dot DOT_FILE  path of displacy as dot
  --print_subject TEXT_FILE
                        path of subject/dependency
  --output FILE         path of output csv :default=stdout

example:
  ja_depndency_analysis.py --output=wagahaiwa_nekodearu_ana.csv wagahaiwa_nekodearu_utf8.txt
  ja_depndency_analysis.py --read_csv --word=吾輩 wagahaiwa_nekodearu_ana.csv
  ja_depndency_analysis.py --read_csv --print_subject=test.txt wagahaiwa_nekodearu_ana.csv


</pre>
## extract_words_table.py
<pre>
usage: extract_words_table.py [-h] [-v] [--sort COLUMN]
                              [--user_dict DCITIONARY]
                              [--unicode_normalized_mode {NFD,NFC,NFCD,NFKC}]
                              [--with_neologdn] [--for_mecab FOR_MECAB]
                              [--without_kansuji] [--output FILE]
                              [FILE [FILE ...]]

語句テーブルを生成する

positional arguments:
  FILE                  files to read, if empty, stdin is used

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --sort COLUMN         column name to sort
  --user_dict DCITIONARY
                        path of user dictionary
  --unicode_normalized_mode {NFD,NFC,NFCD,NFKC}
                        mode of unicode normalization, default=NFKC
  --with_neologdn       processing with neologdn
  --for_mecab FOR_MECAB
                        MECABで記号とする半角記号文字列,ex:'()[]{}'
  --without_kansuji     漢数字の半角数字への変換を行わない
  --output FILE         path of output file : default=stdout

remark:
  PYTHONPATH must be defined as path of directory where threr is 'textmining_lib.py'.

example:


</pre>
## make_mecab_dict.py
<pre>
usage: make_mecab_dict.py [-h] [-v] [--pos1 COLUMN] [--user_dict DCITIONARY]
                          [--header]
                          FILE COLUMN COLUMN

CSV形式の語句リストからMeCab用辞書生成のためのCSVファイルを生成する

positional arguments:
  FILE                  files to read, if empty, stdin is used
  COLUMN                column name for word
  COLUMN                column name for yomigana

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --pos1 COLUMN         column name for part of speech subdivision
  --user_dict DCITIONARY
                        path of user dictionary
  --header              with column header

remark:
  語句と読みのカラムを持つCSVからMeCab用辞書を生成する。
  読みが未指定の場合は、内部で読みを生成する。
  品詞は名詞で固定指定となっており、名詞細分類1はデフォルトは固有名詞である。'--pos1'により名詞細分類列が指定された場合はその値が用いられる。

example:


</pre>
## search_propr_nouns.py
<pre>
usage: search_propr_nouns.py [-h] [-v] [--output FILE] [--with_symbol]
                             [--with_number] [--start_with_number]
                             [--user_dict DCITIONARY]
                             [FILE [FILE ...]]

名詞が連続するものを固有名詞の候補として抽出する。

positional arguments:
  FILE                  files to read, if empty, stdin is used

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --output FILE         path of output file : default=stdout
  --with_symbol         treating symbol as word
  --with_number         treating numbr as word
  --start_with_number   treating word starting with number
  --user_dict DCITIONARY
                        path of user dictionary

remark:
  品詞が名詞のものが連続するものを固有名詞候補として抽出
  ただし、品詞細分類が以下に含まれるものは、連続語句として扱わない。
    ['*', '読点', '句点', '係助詞', '格助詞', '自立', '非自立', '接続助詞', '括弧開', '括弧閉', '副詞可能', '連体化', '空白']
  数字で始まるもの(例：1装置)を出力に含めたい場合は、'--start_with_number'を指定する。
  数字を含むもの(例：〇〇１装置)を出力に含めたい場合は、'--with_number'を指定する。
  記号を含むもの(例：装置[A])を出力に含めたい場合は、'--with_symbol'を指定する。

example:
</pre>
