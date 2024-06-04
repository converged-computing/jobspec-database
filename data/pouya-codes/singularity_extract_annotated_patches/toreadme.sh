echo """# Extract Annotated Patches

### Development Information ###

\`\`\`
Date Created: 22 July 2020
Last Update: 26 Aug 2021 by Amirali
Developer: Colin Chen
Version: 1.6.4
\`\`\`

**Before running any experiment to be sure you are using the latest commits of all modules run the following script:**
\`\`\`
(cd /projects/ovcare/classification/singularity_modules ; ./update_modules.sh --bcgsc-pass your/bcgsc/path)
\`\`\`

### Usage ###
\`\`\`
""" > README.md

python app.py -h >> README.md
echo >> README.md
python app.py from-experiment-manifest -h >> README.md
echo >> README.md
python app.py from-arguments -h >> README.md
echo >> README.md
python app.py from-arguments use-manifest -h >> README.md
echo >> README.md
python app.py from-arguments from-hd5-files -h >> README.md
echo >> README.md
python app.py from-arguments use-directory -h >> README.md
echo >> README.md
python app.py from-arguments use-directory use-slide-coords -h >> README.md
echo >> README.md
python app.py from-arguments use-directory use-annotation -h >> README.md
echo >> README.md
python app.py from-arguments use-directory use-entire-slide -h >> README.md
echo >> README.md
python app.py from-arguments use-directory use-mosaic -h >> README.md
echo """\`\`\`
""" >> README.md

echo """

In order to increase the speed of extract_annotated_patches, We should run parallel jobs. In order to achieve this, you should use this bash script file:
\`\`\`
#!/bin/bash
#SBATCH --job-name Patch Extraction
#SBATCH --cpus-per-task 1
#SBATCH --array=1-<num_slides>
#SBATCH --output path/to/folder/%a.out
#SBATCH --error path/to/folder/%a.err
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_extract_annotated_patches
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<email>
#SBATCH -p upgrade

singularity run -B /projects/ovcare/classification -B /projects/ovcare/WSI singularity_extract_annotated_patches.sif from-arguments \\
 --hd5_location path/to/folder \\
 --num_patch_workers 1 \\
 use-directory \\
 --patch_location path/to/folder \\
 --slide_location path/to/folder \\
 --slide_pattern subtype \\
 --slide_idx \$SLURM_ARRAY_TASK_ID \\
 --store_extracted_patches \\
 use-entire-slide \\
 --slide_coords_location path/to/file \\
 --patch_size 2048 \\
 --stride 2000 \\
 --resize_sizes 512 \\
 # --max_slide_patches 10


\`\`\`

1. The number of arrays should be set to value of \`num_slides / num_patch_workers\`.
2. For fastest way, set the \`num_patch_workers=1\`, then number of arrays is \`num_slides\`.
""" >> README.md
