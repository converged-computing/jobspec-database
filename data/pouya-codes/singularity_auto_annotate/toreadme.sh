echo """# Auto Annotate

### Development Information ###

\`\`\`
Date Created: 22 July 2020
Last Update: 1 Sep 2021 2021 by Amirali
Developer: Colin Chen
Version: 1.6.1
\`\`\`

**Before running any experiment to be sure you are using the latest commits of all modules run the following script:**
\`\`\`
(cd /projects/ovcare/classification/singularity_modules ; ./update_modules.sh --bcgsc-pass your/bcgsc/path)
\`\`\`

### Usage ###

\`\`\`""" > README.md

python app.py -h >> README.md
echo >> README.md
python app.py from-experiment-manifest -h >> README.md
echo >> README.md
python app.py from-arguments -h >> README.md
echo >> README.md
python app.py from-arguments use-manifest -h >> README.md
echo >> README.md
python app.py from-arguments use-directory -h >> README.md
echo >> README.md
echo """\`\`\`
""" >> README.md

echo """**Caution: you need to check the error log constantly to monitor _CUDA out of memory_ error. You need to use a smaller number for _--num_patch_workers_ if you get this error.**""" >> README.md

echo """

In order to increase the speed of auto_annotate, We should run parralel jobs. In order to achieve this, you should use this bash script file:
\`\`\`
#!/bin/bash
##!/bin/bash
#SBATCH --job-name annotate
#SBATCH --cpus-per-task 1
#SBATCH --array=1-793
#SBATCH --output path/to/folder/%a.out
#SBATCH --error path/to/folder/%a.err
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_auto_annotate
#SBATCH --mem=6G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<email>
#SBATCH -p upgrade

singularity run -B /projects/ovcare/classification -B /projects/ovcare/WSI singularity_auto_annotate.sif from-arguments
 --log_file_location path/to/file
 --log_dir_location path/to/folder
 --patch_location path/to/folder
 --slide_location path/to/folder
 --slide_pattern=""
 --patch_size 1024
 --resize_sizes 512
 --evaluation_size 512
 --is_tumor True
 --store_extracted_patches True
 --classification_threshold 0.9
 --num_patch_workers 1
 --slide_idx \$SLURM_ARRAY_TASK_ID
 --maximum_number_patches Tumor=400 Stroma=400

\`\`\`
""" >> README.md

echo "The number of arrays should be set to value of \`num_slides / num_patch_workers\`." >> README.md
echo "For fastest way, set the \`num_patch_workers=1\`, then number of arrays is \`num_slides\`." >> README.md
echo "If you want to extracted tumor patches with probability between 0.4 and 0.6, you should set \`classification_threshold=0.4\`, \`classification_max_threshold=0.6\`, and \`label=Tumor\`." >> README.md
