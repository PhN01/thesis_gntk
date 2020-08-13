
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--dataset)
    dataset="${2}"
    shift # past argument
    shift # past value
    ;;
    -e|--kernel)
    kernel="${2}"
    shift # past argument
    shift # past value
    ;;
    -e|--job_type)
    job_type="${2}"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    positional+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

# set experiment parameters
niters=500
job_mem=64000
job_time=119:59

if [ "${job_type}" = "p" ]; then
  python ./src/model_selection/gk_kernel_cv.py ./data/02_matrices/Graphkernels/${dataset}/${kernel}.npz -n ${dataset} -o ./data/03_experiments/Graphkernels/${dataset}_${kernel}.json -I ${niters}
elif [ "${job_type}" = "b" ]; then
  echo "bsub -W ${job_time} -R \"rusage[mem=${job_mem}]\" python ./src/model_selection/gk_kernel_cv.py ./data/02_matrices/Graphkernels/${dataset}/${kernel}.npz -n ${dataset} -o ./data/03_experiments/Graphkernels/${dataset}_${kernel}.json -I ${niters}" | bsub
fi