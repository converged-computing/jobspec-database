.PHONY: make_conda update_conda remove_conda update_data update_data_dry update_data_dry

.ONESHELL:

PROJECT_NAME = string_sims
BACK_UP_REPO = tcblab:~/my_volumes/KcsA_string_method_swarms/

make_conda:
	conda deactivate
	conda env create -f environment.yml
	conda activate $(PROJECT_NAME)
	ipython kernel install --user --name=$(PROJECT_NAME)
	pip install -e .

compress_data:
	cd data/raw/
	for i in */md/; do cd $$i; cd ..; echo $$i; tar cfz md.tar.gz md/ ; cd ..; done

update_conda:
	conda env update --file environment.yml --prune
	ipython kernel install --user --name=$(PROJECT_NAME)

remove_conda:
	conda remove --name=$(PROJECT_NAME) --all

update_data:
	 grep -v "^#" data/raw/dir_list.txt | while read a b;do rsync -rauLih   --progress  --exclude-from=data/raw/exclude_list.txt   $$a $$b;done
	 python  src/data/get_external_data.py

update_data_dry:
	 grep -v "^#" data/raw/dir_list.txt | while read a b; do echo $$a $$b; rsync -raunLih  --exclude-from=data/raw/exclude_list.txt   $$a $$b;done

update_data_dry_verbose:
	 grep -v "^#" data/raw/dir_list.txt | while read a b;do rsync -raunLivvvh   --exclude-from=data/raw/exclude_list.txt   $$a $$b;done

clean:
	find . -iname \#* -exec rm {} \;
	find . -iname "slurm*err" -exec rm {} \;
	find . -iname "slurm*out" -exec rm {} \;

format:
	black -l 79 .
	isort .

back_up_repo:
	rsync --del -rauLih --exclude md/ /home/sperez/Projects/string_sims/ tcblab:~/my_volumes/KcsA_string_method_swarms/string_sims/

back_up_repo_dry:
	rsync --del -rnauLih --exclude md/ /home/sperez/Projects/string_sims/ tcblab:~/my_volumes/KcsA_string_method_swarms/string_sims/

help:
	@echo "Possible options:"
	@echo "make_conda"
	@echo "update_conda"
	@echo "remove_conda"
	@echo "update_data"
	@echo "update_data_dry"
	@echo "back_up_repo"
	@echo "back_up_repo_dry"
	@echo "compress_data"
