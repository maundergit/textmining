## make_mecab_dict.sh
<pre>
Usage: make_mecab_dict.sh [-m] csv_file user_dic_file system_dicdir
options:
  -m  : use ipa model to calcurate cost

remark:
  model file:
    https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7bnc5aFZSTE9qNnM
  dic dir:
     'mecab --dictionary-info' or 'mecab-config --dicdir'
     'sudo update-alternatives --config mecab-dictionary'

example:
  python make_mecab_dict.py test_dict.csv 語句 読み > dictionary.csv
  make_mecab_dict.sh dictionary.csv user.dic /var/lib/mecab/dic/ipadic-utf8/

</pre>
