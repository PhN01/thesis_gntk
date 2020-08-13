.ONESHELL:

reqs:
	pipreqs --force --savepath ./requirements.txt
	cat ./requirements.txt

stats:
	./run_scripts/run_compute_statistics.sh

