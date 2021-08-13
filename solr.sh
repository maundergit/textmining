#!/bin/bash
# -*- mode: sh;-*-
#----------------------------------------------------------------------
# Author:       m.akei
# Copyright:    (c)2021 , m.akei
# Time-stamp:   <2021-08-01 11:32:16>
#----------------------------------------------------------------------
DTSTMP=$(date +%Y%m%dT%H%M%S)

DDIR=$(dirname $0)
SNAME=$(basename $0)
#DDIR_ABS=$(cd $(dirname ${DDIR}) && pwd)/$(basename $DDIR)
DDIR_ABS=$(realpath ${DDIR})
TMPDIR=/tmp
# TMPDIR=/tmp/${SNAME}.$$

unused_port(){
    # usage: unused_port star_port [end_port]
    SP=$1
    EP=${2:-$((${SP}+10))}
    PS=($(nc -v -z 127.0.0.1 ${SP}-${EP} 2>&1 | awk '/failed/ {print $6}'))
    if (( ${#PS[@]} > 0)); then
	echo "${PS[0]}"
    else
	echo "??error:unused_port: unused port was not found in ${SP}-${EP}" 1>&2
	exit 1
    fi
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

check_commands solr nc

usage_exit() {
    cat <<EOF 2>&1
convinient command wrapper of solr
Usage: $SNAME [-d dir] [-p port] start
Usage: $SNAME reset_core core_name
Usage: $SNAME command
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
  

EOF
    exit 1
}

while getopts ad:p:h OPT
do
    case $OPT in
        a)  FLAG_A=1
            ;;
        d)  SOLR_DATADIR=$OPTARG
            ;;
        p)  PORT=$OPTARG
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
OSNAME=$(uname -o 2>/dev/null || echo others)
if [ "${OSNAME}" = "GNU/Linux" ]; then
    IPADDR=($(hostname -I))
    IPADDR=$(echo ${IPADDR[0]} | tr -d ' ')
else
    # Cygwin list - Utility to get IP address of the machine http://cygwin.1069669.n5.nabble.com/Utility-to-get-IP-address-of-the-machine-td3359.html
    IPADDR=$(perl -MSocket -MSys::Hostname -wle 'print inet_ntoa(scalar gethostbyname(hostname || "localhost"))')
fi

S_CMD=$1
shift 

PORT=${PORT:-8983}
SOLR_DATADIR=${SOLR_DATADIR:-$(pwd)}

#----
SOLR_ABS=$(readlink -f $(which solr))
SOLR_SERVER_HOME=$(readlink -f $(dirname ${SOLR_ABS})/..)/server

case ${S_CMD} in
    start)
	PORT2=$(unused_port ${PORT})
	if [ "${PORT}" != "${PORT2}" ]; then
	    echo -e "\n#warn:${SNAME}: port was changed to unused number: ${PORT} -> ${PORT2}\n" 1>&2
	    PORT=${PORT2}
	fi
	SOPTS="-p ${PORT} -d ${SOLR_SERVER_HOME}"
	if [ "${SOLR_DATADIR}" != "" ]; then
	    SOLR_DATADIR=$(realpath ${SOLR_DATADIR})
	    SOPTS="${SOPTS} -s ${SOLR_DATADIR}"
	fi
	echo "${SOLR_ABS} start ${SOPTS} $*" 1>&2
	${SOLR_ABS} start ${SOPTS} $*
	echo -e "-- url: http://${IPADDR}:${PORT}/" 1>&2
	;;
    reset)
	CORE_NAME=$1
	SOPTS="-p ${PORT}"
	${SOLR_ABS} delete ${SOPTS} -c ${CORE_NAME}
	${SOLR_ABS} create_core ${SOPTS} -c ${CORE_NAME}
	;;
    *)
	SOPTS="-p ${PORT}"
	${SOLR_ABS} ${S_CMD} ${SOPTS} $*
	;;
esac

#----
# remove_tmpdir
#-------------
# Local Variables:
# mode: sh
# coding: utf-8-unix
# End:

