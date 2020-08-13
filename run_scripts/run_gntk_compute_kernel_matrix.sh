# example:
# ./single_gram_v2.sh --data_source TU_DO --dataset PROTEINS --job_type b

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--data_source)
    data_source="${2}"
    shift # past argument
    shift # past value
    ;;
    -e|--dataset)
    dataset="${2}"
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

# set batch job parameters
if [ "${job_type}" = "b" ]; then
  job_cores=40
  job_mem=8096
  case ${dataset} in
  "MUTAG" | "IMDBBINARY" | "IMDBMULTI" | "PTC" | "PROTEINS")
    job_time="3:59"
  ;;
  "NCI1")
    job_time="23:59"
  ;;
  esac
fi

# set lists of hyperparameters
# jk_list=(1 0)
# L_list=(1)
# R_list=(1)
# scale_list=(uniform)

jk_list=(1 0)
L_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
R_list=(1 2 3)
scale_list=(uniform degree)

n_jobs="$((${#L_list[@]} * ${#R_list[@]} * ${#scale_list[@]} * ${#jk_list[@]}))"
i=0

for L in ${L_list[@]}; do
for R in ${R_list[@]}; do
for scale in ${scale_list[@]}; do
for jk in ${jk_list[@]}; do

if [ "${job_type}" = "p" ]; then
  i=$((i+=1))
  echo "Starting job ${i}/${n_jobs}."
  python ./src/gntk/gntk_compute_kernel_matrix.py --data_source "${data_source}" --dataset "${dataset}" --n_blocks "${L}" --n_fc_layers "${R}" --scale "${scale}" --jumping_knowledge "${jk}"
elif [ "${job_type}" = "b" ]; then
  echo "bsub -n ${job_cores} -W ${job_time} -R \"rusage[mem=${job_mem}]\" python ./src/gntk/gntk_compute_kernel_matrix.py --data_source ${data_source} --dataset ${dataset} --n_blocks ${L} --n_fc_layers ${R} --scale ${scale} --jumping_knowledge ${jk}" | bash
else
  echo "unknown job_type. select 'p' for python and 'b' for batch submission."
fi

done
done
done
done

if [ "${job_type}" = "b" ]; then
  echo "${n_jobs} batch jobs submitted."
fi