# example:
# ./single_gram_v2.sh --data_dir ../../data/TU_DO --dataset PROTEINS --exp_name standard_params_new
# --n_jobs 40 --job_type b --scratch 1 --n_cores 20 --mem 8096 --exp a.1

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
    -e|--exp)
    exp="${2}"
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
set -- "${positional[@]}" # restore positional parameters

if [ "${job_type}" = "b" ]; then
  job_cores=40
  job_mem=4096
  case ${dataset} in
  "MUTAG" | "IMDBBINARY" | "IMDBMULTI" | "PTC")
    job_time="23:59"
  ;;
  "NCI1" | "PROTEINS")
    job_time="119:59"
  ;;
  esac
fi

# set experiment parameters
if [ "${exp}" = "a.1" ]; then
  n_iter=1
elif [ "${exp}" = "a.2" ]; then
  n_iter=1
elif [ "${exp}" = "b.1" ]; then
  n_iter=10
else
  echo "Unknown experiment code (allowed: a.1, a.2, b.1). Exiting."
  exit 1
fi

iteration=0
while [[ ${iteration} -lt ${n_iter} ]]; do
  case ${job_type} in
    "p")
      python ./src/model_selection/gntk_kernel_cv.py --dataset ${dataset} --experiment ${exp} --iteration_idx ${iteration} --verbose 1
    ;;
    "b")
      echo "bsub -n ${job_cores} -W ${job_time}" -R "rusage[mem=${job_mem}] python ./src/model_selection/gntk_kernel_cv.py --dataset ${dataset} --experiment ${exp} --iteration_idx ${iteration} --verbose 1" | bash
    ;;
  esac
  iteration=$((iteration+=1))
done

if [ "${job_type}" = "b" ]; then
  echo "${iteration} jobs submitted."
fi