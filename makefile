.ONESHELL:

update_reqs:
	pipreqs --force --savepath ./requirements.txt
	cat ./requirements.txt

setup:
	python -m venv env
	source env/bin/activate
	pip install -r requirements.txt

stats:
	./run_scripts/run_compute_statistics.sh

evaluation: stats
	./run_scripts/run_evaluation.sh