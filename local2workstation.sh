PROGNAME=$(basename $0)
VERSION="1.0"
#bash local2workstation.sh 4 -i
#bash local2workstation.sh 14 -a 2d-data 2d-video -e .csv video/2018/
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
    echo "example: bash local2workstation.sh 14 -a 2d-data 2d-video -e .py ./"
    exit 1
}

if [ $# -lt 2 ]; then
  echo "argument:$#" 1>&2
  echo "requirements: more than 2(detail is below)" 1>&2
  echo "bash local2workstation.sh 4 -a 2d-data 2d-video -e .py ./" 1>&2
  exit 1
fi

declare -i WS=2
EXTENSION="FALSE"
INIT=false
REINIT=false
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
        '-i'|'--init' )
            INIT=true
            break
            ;;
        '-r'|'--reinit' )
            REINIT=true
            break
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

if $INIT; then
    # copy directory structure only
    scp ./setup-server.sh ~/.ssh/id_rsa.pub kado@192.168.1.$WS:~/
    ssh kado@192.168.1.$WS bash setup-server.sh
    ssh kado@192.168.1.$WS rm setup-server.sh
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --include "*/" --exclude "*" /home/junkado/Desktop/ubuntu_project/python_ubuntu/machine-learning kado@192.168.1.$WS:/home/kado/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' /home/junkado/Desktop/ubuntu_project/python_ubuntu/MIL/MIL kado@192.168.1.$WS:/home/kado/machine-learning/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/misvm/* kado@192.168.1.$WS:/home/kado/machine-learning/misvm/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/2d-data/* kado@192.168.1.$WS:/home/kado/machine-learning/2d-data/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/2d-video/*  kado@192.168.1.$WS:/home/kado/machine-learning/2d-video/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/video/*  kado@192.168.1.$WS:/home/kado/machine-learning/video/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/*.py kado@192.168.1.$WS:/home/kado/machine-learning/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/Dockerfile* kado@192.168.1.$WS:/home/kado/machine-learning/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}/*.sh kado@192.168.1.$WS:/home/kado/machine-learning/
    exit 1
fi

if $REINIT; then
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/misvm/* kado@192.168.1.$WS:/home/kado/machine-learning/misvm/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/2d-data/* kado@192.168.1.$WS:/home/kado/machine-learning/2d-data/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/2d-video/*  kado@192.168.1.$WS:/home/kado/machine-learning/2d-video/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/video/*  kado@192.168.1.$WS:/home/kado/machine-learning/video/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/*.py kado@192.168.1.$WS:/home/kado/machine-learning/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/Dockerfile* kado@192.168.1.$WS:/home/kado/machine-learning/
    rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' --delete ${misvm_dir}/*.sh kado@192.168.1.$WS:/home/kado/machine-learning/
fi

if [ ! ${#param[@]} = 0 ]; then
    for file in ${param[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}${file} kado@192.168.1.$WS:/home/kado/machine-learning/
    done
fi

if [ ! ${#EXTENSIONPATH[@]} = 0 ]; then
    for file in ${EXTENSIONPATH[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}${file}/*${EXTENSION} kado@192.168.1.$WS:/home/kado/machine-learning/${file}/
    done
fi

#no need?
if [ ! ${#ALLPATH[@]} = 0 ]; then
    for file in ${ALLPATH[@]}; do
        rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' ${misvm_dir}${file} kado@192.168.1.$WS:/home/kado/machine-learning/
    done
fi

exit 1
