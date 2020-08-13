# example: ./single_gram_v2.sh --data_dir ../../data/GNTK_paper --dataset PROTEINS --exp_name standard_params --n_threads 40 --job_type b

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
    -e|--job_type)
    job_type="${2}"
    shift # past argument
    shift # past value
    ;;
    -e|--gpu)
    gpu="${2}"
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

# setting the CV parameters
nrep=10
kfold=10

# iterating over repetitions and folds of the CV and submitting a separate job for each fold
rep=0
while [[ ${rep} -lt ${nrep} ]]
do
    fold=0
    while [[ ${fold} -lt ${kfold} ]]
    do
        if [ "${job_type}" = "p" ]; then
            python ./src/gin/main.py --dataset ${dataset} --rep_idx ${rep} --fold_idx ${fold}
        elif [ "${job_type}" = "b" ]; then
            if [ "${gpu}" = "1" ]; then
                echo "bsub -n 4 -W 2:00 -R \"rusage[mem=8096,ngpus_excl_p=1]\" python ./src/gin/main.py --dataset ${dataset} --rep_idx ${rep} --fold_idx ${fold}" | bash
            elif [ "${gpu}" = "0" ]; then
                echo "bsub -n 4 -W 4:00 -R \"rusage[mem=8096]\" python ./src/gin/main.py --dataset ${dataset} --rep_idx ${rep} --fold_idx ${fold}" | bash
            fi
        fi
        fold=$((fold+=1))
    done
    rep=$((rep+=1))
done