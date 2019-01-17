#!/usr/bin/env bash
PROGNAME=$(basename $0)
VERSION="1.0"
#bash local2workstation.sh 4 -i
#bash local2workstation.sh 14 -a 2d-data 2d-video -e .csv video/2018/
usage() {
    echo "Usage: $PROGNAME [OPTIONS]"
    echo "  This script is ~."
    echo
    echo "Options:"
    echo "  -h, --help"
    echo "      --version"
    echo "  -s, --set MILPath misvmPath"
    echo
    echo "example: bash reinit.sh"
    echo "example: bash reinit.sh -s ../../MIL ../../../misvm"
    exit 1
}


SET=false
declare -a SETPATH=()

#shift 1

for OPT in "$@"
do
    case "$OPT" in
        '-h'|'--help' )
            usage
            exit 1
            ;;
        '--version' )
            echo $VERSION
            exit 1
            ;;
        '-s'|'--set' )
            if [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                echo "$PROGNAME: -s requires two arguments -- $1" 1>&2
                exit 1
            fi
            shift 1
            cnt=0
            while [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]
            do
                cnt=$((++cnt))
                SETPATH+=( "$1" )
                echo "$1"
                shift 1
            done
            if [ $cnt -ne 2 ]; then
                echo "$PROGNAME: -s requires two arguments -- $1" 1>&2
                exit 1
            fi
            SET=true
            ;;
        '--'|'-' )
            shift 1
            param+=( "$@" )
            break
            ;;
        -*)
            echo "$PROGNAME: illegal option -- '$(echo $1 | sed 's/^-*//')'" 1>&2
            exit 1
            ;;
        *)
            if [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]; then
                #param=( ${param[@]} "$1" )
                param+=( "$1" )
                shift 1
            fi
            ;;
    esac
done


if [[ -L ../MIL ]]; then
   echo "remove MIL file"
   unlink ../MIL
fi
if [[ -L ../misvm ]]; then
   echo "remove misvm file"
   unlink ../misvm
fi

echo $SET
if $SET; then
   echo "make a symboliclink to MIL"
   ln -s -r ${SETPATH[0]} ../
   echo "make a symboliclink to misvm"
   ln -s -r ${SETPATH[1]} ../
else
   echo "make a symboliclink to MIL"
   ln -s -r ../../MIL/MIL ../
   echo "make a symboliclink to misvm"
   ln -s -r ../../misvm ../
fi

exit 1
