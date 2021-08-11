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

check_os () {
    OSNAME=$(uname -o 2>/dev/null || echo others)
    OSVER=$(uname -a 2>/dev/null)   # -r or -v
    if [ "${OSNAME}" = "GNU/Linux" ]; then
        if [[ "${OSVER}" =~ microsoft ]]; then
            echo something to do for WSL
        else
            echo something to do for linux other than WSL
        fi
        echo something to do for linux
    else
        echo something to do for others
    fi
}
check_xwindow() {
    if [ "$DISPLAY" = "" ]; then
        echo "-- error: 'DISPLAY' environment was not definded." >&2
        exit 1
    fi
    if [ "$(xset q 2>/dev/null)" = "" ]; then
        echo "-- error: No X server at \$DISPLAY [$DISPLAY]" >&2
        exit 1
    fi
}
check_do() {
    MODE=$1
    CHK_CMD=$2
    shift
    CMD="$@"
    unset MES
    if [ "$(which ${CHK_CMD})" = "" ]; then
        if [ "${MODE}" = "0" ]; then
            echo "-- warning: not found ${CHK_CMD}"
        else
            echo "-- error: not found ${CHK_CMD}"
            exit 1;
        fi
    else
        ${CMD}
    fi
}
change_stdout(){
    F=$@
    exec 1<&-     # close stdout
    exec 1<>${F}  # reopen stdout as ${F} for read and write
    # exec 2<&-   # close stderr
    # exec 2>&1   # reopen 
}
check_commands_and_install() {
    # usage: check_commands_and_install "${array[@]}"
    CHK_CMDS=("$@")
    unset MES
    for c in "${CHK_CMDS[@]}"; do
        if [ "$(which ${c})" = "" ]; then
            if [ -x /usr/lib/command-not-found ]; then
                APT_CMD=$(/usr/lib/command-not-found -- "$c" 2>&1 | awk '/^(sudo|apt)/ {print $0}')
                ${APT_CMD}
            elif [ -x /usr/share/command-not-found/command-not-found ]; then
                APT_CMD=$(/usr/share/command-not-found -- "$c" 2>&1 | awk '/^(sudo|apt)/ {print $0}')
                ${APT_CMD}
            else
                MES="${MES}-- error: not found $c\n"
            fi
        fi
    done
    if [ "${MES}" != "" ]; then
        echo -e "${MES}" 1>&2
        exit 1
    fi
}
get_absolutepath() {
    P=$1
    RES=$(cd $(dirname ${P}) && pwd)/$(basename ${P})
    echo ${RES}
}
print_array(){
    # bash - How to pass an array as function argument? - Ask Ubuntu https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
    # usage: print_array "${array[@]}"
    local A
    A=("$@")
    for v in "${A[@]}"; do
        echo "'$v'"
    done
}
function stderr_out(){
    MESSAGES=("$@")
    for m in "${MESSAGES[@]}"; do
	echo "${m}" 1>&2
    done
}

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
Usage: $SNAME [-m] csv_file user_dic_file system_dicdir
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

