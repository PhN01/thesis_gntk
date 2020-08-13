
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--n_samples)
    n_samples="${2}"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    positional+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${positional[@]}" # restore positional parameters

echo "${n_samples}"
python ./src/experiments/profiling_experiment.py --n_samples ${n_samples}