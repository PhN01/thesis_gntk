
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "here"
    if [ -d "./env" ]; then
        source env/bin/activate
    else
        echo "Virtual environment not found. Run make setup first."
        exit 1
    fi
fi
python ./src/data/data_statistics.py --n_samples ${n_samples}