## solr.sh
<pre>
convinient command wrapper of solr
Usage: solr.sh [-d dir] [-p port] start
Usage: solr.sh reset_core core_name
Usage: solr.sh command
arguments:
  command: start, stop, restart, status, healthcheck, create, create_core, create_collection, 
           delete, version, zk, auth, assert, config, autoscaling, export
  core_name : core name to delete and create.

options:
  -d dir : path of directory of data, default is current directory
  -p port: port number, default=8983
           for start, unused port is automaticaly scaned.

example:
  solr.sh start  # current directory is used as directory for data of solr.
  



</pre>
## make_mecab_dict.sh
<pre>
CSV形式の辞書登録用語句リストからMecabのユーザー辞書ファイルを生成する
Usage: make_mecab_dict.sh [-m] csv_file user_dic_file system_dicdir
arguments:
  csv_file: make_mecab_dict.py で生成された辞書登録用語句リストを持つCSVファイル
  user_dic_file: ユーザー辞書ファイル名、上書きされるので注意
  system_dicdit: Mecabのシステムディレクトリ

options:
  -m  : use ipa model to calcurate cost

remark:
  model file:
    https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7bnc5aFZSTE9qNnM
  system dicdir:
     'mecab --dictionary-info' or 'mecab-config --dicdir'
     'sudo update-alternatives --config mecab-dictionary'

example:
  python make_mecab_dict.py test_dict.csv 語句 読み > dictionary.csv
  make_mecab_dict.sh dictionary.csv user.dic /var/lib/mecab/dic/ipadic-utf8/

</pre>
