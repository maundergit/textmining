#!/bin/bash
# -*- mode: sh;-*-
#----------------------------------------------------------------------
# Author:       m.akei
# Copyright:    (c)2021 , m.akei
# Time-stamp:   <2021-04-10 10:43:37>
#----------------------------------------------------------------------
DTSTMP=$(date +%Y%m%dT%H%M%S)

DDIR=$(dirname $0)
SNAME=$(basename $0)
#DDIR_ABS=$(cd $(dirname ${DDIR}) && pwd)/$(basename $DDIR)
DDIR_ABS=$(realpath ${DDIR})
TMPDIR=/tmp
# TMPDIR=/tmp/${SNAME}.$$


#---- 
remove_tmpdir(){
    if [[ "${TMPDIR}" != "" && "${TMPDIR}" =~ ${SNAME} && -e "${TMPDIR}" ]]; then
	rm -rf "${TMPDIR}"
    fi
}
make_tmpfile(){
    ID=$1
    if [[ "${TMPDIR}" != "" && ! -e "${TMPDIR}" ]]; then
	mkdir "${TMPDIR}"
    fi
    FN="${TMPDIR}/${ID}_$$.tmp"
    echo "${FN}"
}
check_and_get_stdin(){
    if [ "${INPUT}" = "-" ]; then
	if [ ! -e "${TMPDIR}" ]; then
	    mkdir "${TMPDIR}"
	fi
	TMPFILE_INPUT="${TMPDIR}/${SNAME}_input.tmp"
	cat > ${TMPFILE_INPUT}
	INPUT=${TMPFILE_INPUT}
    elif [ ! -e "${INPUT}" ]; then
	echo "??Error: file not found: ${INPUT}" 1>&2
	exit 1
    fi
    echo ${INPUT}
}
check_commands() {
    # check_commands ls dataflow find
    # usage: check_commands "${array[@]}"
    CHK_CMDS=("$@")
    unset MES
    for c in "${CHK_CMDS[@]}"; do
        if [ "$(which ${c})" = "" ]; then
            MES="${MES}-- error: not found $c\n"
        fi
    done
    if [ "${MES}" != "" ]; then
        echo -e "${MES}" 1>&2
        exit 1
    fi
}

usage_exit() {
    cat <<EOF 2>&1
CSV形式の辞書登録用語句リストからMecabのユーザー辞書ファイルを生成する
Usage: $SNAME [-m] csv_file user_dic_file system_dicdir
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
  ${SNAME} dictionary.csv user.dic /var/lib/mecab/dic/ipadic-utf8/

EOF
    exit 1
}

while getopts amh OPT
do
    case $OPT in
        a)  FLAG_A=1
            ;;
        m)  F_IPA_MODEL=1
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done
shift $((OPTIND - 1))

if [ "$1" = "" ];  then
    usage_exit
fi


INPUT=$1
INPUT_BASE=$(basename ${INPUT})

PREFIX=${INPUT_BASE%.*}
SUFFIX=${INPUT_BASE#${INPUT_BASE%.*}.}
OUTPUT=${PREFIX}.raw

# INPUT=$(check_and_get_stdin "${INPUT}")

UDIC=$2
DICDIR=$3

#----

USERRC=~/.mecabrc

if [ "${F_IPA_MODEL}" != "" ]; then
    IPA_MODEL=${DDIR}/mecab-ipadic_utf8.model
    if [ ! -e ${IPA_MODEL} ]; then
	echo "??error:${SNAME}: ${IPA_MODEL} was not found." 1>&2
	exit 1
    fi
fi

LIBEXECDIR=$(mecab-config --libexecdir)
# DICDIRROOT=$(mecab-config --dicdir)
SYSCONFDIR=$(mecab-config --sysconfdir)

MECABDICINDEX=${LIBEXECDIR}/mecab-dict-index

if [ -e ${IPA_MODEL} ]; then
    MOPT="${MOPT} -m ${IPA_MODEL}"
fi
echo ${MECABDICINDEX} ${MOPT} -d${DICDIR} -u ${UDIC} -f utf8 -t utf8 ${INPUT}
${MECABDICINDEX} ${MOPT} -d${DICDIR} -u ${UDIC} -f utf8 -t utf8 ${INPUT}

echo "-- ${UDIC} was created."

if [ ! -e ${USERRC} ]; then
    MECABRC=${SYSCONFDIR}/mecabrc
    cp ${MECABRC} ${USERRC}
    echo "-- copy ${MECABRC} to ${USERRC}" 1>&2
fi
cat <<EOF 1>&2
 edit 'userdic' in ${USERRC}.
 ex.
 userdic = ${UDIC}
EOF


#----
# remove_tmpdir
#-------------
# Local Variables:
# mode: sh
# coding: utf-8-unix
# End:

