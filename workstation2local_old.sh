echo "using workstation..."

COMMAND=`basename $0`

while getopts :a OPTION
do
  case $OPTION in
    a) OPTION_a="TRUE" ;;
#    s ) OPTION_s="TRUE" ; VALUE_s="$OPTARG" ;;
    \?) echo "Usage: $COMMAND [-a] ..." 1>&2
        exit 1 ;;
  esac
done

shift $(($OPTIND - 1))

misvm_dir=/home/junkado/Desktop/ubuntu_project/python_ubuntu/machine-learning/


if [ $# = 2 ]; then
   file=$1
   if [ "$OPTION_a" = "TRUE" ]; then
       file=*$1
   fi
   rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' kado@192.168.1.12:/home/kado/machine-learning/${file} ${misvm_dir}/$2
elif [ $# = 1 ]; then
   file=$1
   if [ "$OPTION_a" = "TRUE" ]; then
       file=*$1
   fi
   rsync -arv -e 'ssh -i ~/.ssh/id_rsa.pub' kado@192.168.1.12:/home/kado/machine-learning/${file} ${misvm_dir}
else
    echo "2(1) argument are required" 1>&2
    echo "local_relative_path [ws_dir_name]" 1>&2
    exit 1
fi
