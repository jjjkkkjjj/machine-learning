PROGNAME=$(basename $0)
VERSION="1.0"
#bash workstation2local.sh -w 14 -a __old-weight/img25/g0.001c1
usage() {
    echo "Usage: $PROGNAME [OPTIONS] -- OriginalFilePath1 OrigFilePath2 ..."
    echo "  This script is ~."
    echo
    echo "Options:"
    echo "  -h, --help"
    echo "      --version"
    echo "  -e, --extension extension(.mp4) path [path2...]"
    echo "  -w, --workstation [1~14] (default 2)"
    echo "  -a, --all path"
    echo
    exit 1
}

if [ $# -lt 2 ]; then
  echo "argument:$#" 1>&2
  echo "requirements: more than 2(detail is below)" 1>&2
  echo "bash workstation2local.sh 4 -a 2d-data 2d-video -e .py ./" 1>&2
  exit 1
fi

declare -i WS=2
EXTENSION="FALSE"
declare -a EXTENSIONPATH=()
declare -a ALLPATH=()

WS=$1
shift 1

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
        '-e'|'--extension' )
            if [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                echo "$PROGNAME: option requires an argument -- $1" 1>&2
                exit 1
            fi
            EXTENSION="$2"
            shift 2

            if [[ -z "$1" ]] || [[ "$1" =~ ^-+ ]]; then
                echo "$PROGNAME: option requires an argument -- $1" 1>&2
                exit 1
            fi
            while [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]
            do
                EXTENSIONPATH+=( "$1" )
                shift 1
            done
            ;;
        '-a'|'--all' )
            if [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
                echo "$PROGNAME: option requires an argument -- $1" 1>&2
                exit 1
            fi
            shift 1
            while [[ ! -z "$1" ]] && [[ ! "$1" =~ ^-+ ]]
            do
                ALLPATH+=( "$1" )
                shift 1
            done
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

WS=WS+10
misvm_dir=/home/junkado/Desktop/ubuntu_project/python_ubuntu/machine-learning/

if [ ! ${#param[@]} = 0 ]; then
    for file in ${param[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' kado@192.168.1.$WS:/home/kado/machine-learning/${file} ${misvm_dir}/${file}
    done
fi

if [ ! ${#EXTENSIONPATH[@]} = 0 ]; then
    for file in ${EXTENSIONPATH[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' kado@192.168.1.$WS:/home/kado/machine-learning/${file} ${misvm_dir}/${file}
    done
fi

if [ ! ${#ALLPATH[@]} = 0 ]; then
    for file in ${ALLPATH[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' kado@192.168.1.$WS:/home/kado/machine-learning/${file} ${misvm_dir}/${file}
    done
fi

exit 1
