if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "here"
    if [ -d "./env" ]; then
        source env/bin/activate
    else
        echo "Virtual environment not found. Run make setup first."
        exit 1
    fi
fi

datasets=("MUTAG" "PTC" "NCI1" "IMDBBINARY" "IMDBMULTI" "PROTEINS")
for dataset in ${datasets[@]}; do
    if [ ! -d "./data/03_experiments/GIN/${dataset}" ]; then
        echo "Results in ./data/03_experiments/GIN/${dataset} missing."
        exit 1
    elif [ ! -d "./data/03_experiments/GNTK/a.1/${dataset}" ]; then
        echo "Results in ./data/03_experiments/GNTK/a.1/${dataset} missing."
        exit 1
    elif [ ! -d "./data/03_experiments/GNTK/a.2/${dataset}" ]; then
        echo "Results in ./data/03_experiments/GNTK/a.2/${dataset} missing."
        exit 1
    elif [ ! -d "./data/03_experiments/GNTK/b.1/${dataset}" ]; then
        echo "Results in ./data/03_experiments/GNTK/b.1/${dataset} missing."
        exit 1
    elif [ ! -f "./data/03_experiments/Graphkernels/${dataset}_VH.json" ]; then
        echo "Results in ./data/03_experiments/Graphkernels missing for dataset ${dataset}."
        exit 1
    fi
done
if [ ! -f "./data/03_experiments/timing/gram_timing_50samples.txt" ]; then
    echo "Results from timing experiment missing."
    exit 1
elif [ ! -f "./data/03_experiments/GNTK/profiling/results_10samples.txt" ]; then
    echo "Results from profiling experiment missing."
    exit 1
fi

python ./src/eval/evaluate_exp_a.py
python ./src/eval/evaluate_exp_b.py
python ./src/figures/create_plots.py

echo "Evaluation finished. Results are located in './reporting'."