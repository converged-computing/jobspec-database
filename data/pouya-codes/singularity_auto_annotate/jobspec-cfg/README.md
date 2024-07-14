# Auto Annotate
Auto Annotate is a tool for extracting foreground patches from Whole Slide Images (WSIs), predicting whether a patch is a Tumor or Normal (i.e., non-tumor) patch, and then extracting and downsampling the patches. This tool is highly configurable with several options for customization to suit your requirements.

### Usage ###

```
usage: app.py [-h] {from-experiment-manifest,from-arguments} ...

If the slide dataset has a slide_pattern of 'subtype'. Each patch dataset has the pattern 'annotation/subtype/slide/patch_size/magnification'. For example, if the slide paths look like:

/path/to/slides/MMRd/VOA-1099A.tiff
/path/to/slides/POLE/VOA-1932A.tiff
/path/to/slides/p53abn/VOA-3088B.tiff
/path/to/slides/p53wt/VOA-3266C.tiff

And we use '--resize_sizes 512' then the extracted patch paths will look something like:

/path/to/patches/Tumor/MMRd/VOA-1099A/512/20/30140_12402.png
/path/to/patches/Normal/MMRd/VOA-1099A/512/20/42038_12402.png
...
/path/to/patches/Tumor/POLE/VOA-1932A/512/20/42038_12402.png
/path/to/patches/Normal/POLE/VOA-1932A/512/20/30140_12402.png
...
/path/to/patches/Tumor/p53abn/VOA-3088B/512/20/42038_12402.png
/path/to/patches/Normal/p53abn/VOA-3088B/512/20/30140_12402.png
...
/path/to/patches/Tumor/p53wt/VOA-3266C/512/20/30140_12402.png
/path/to/patches/Normal/p53wt/VOA-3266C/512/20/42038_12402.png
...

This component looks in the YAML section of the training log to obtain the trained model and it's hyperparameters.

positional arguments:
  {from-experiment-manifest,from-arguments}
                        Choose whether to use arguments from experiment manifest or from commandline
    from-experiment-manifest
                        Use experiment manifest

    from-arguments      Use arguments

optional arguments:
  -h, --help            show this help message and exit

usage: app.py from-experiment-manifest [-h] [--component_id COMPONENT_ID]
                                       experiment_manifest_location

positional arguments:
  experiment_manifest_location

optional arguments:
  -h, --help            show this help message and exit

  --component_id COMPONENT_ID

usage: app.py from-arguments [-h] --log_file_location LOG_FILE_LOCATION
                             --log_dir_location LOG_DIR_LOCATION
                             [--store_extracted_patches] [--store_thumbnail]
                             [--generate_annotation] [--skip_area SKIP_AREA]
                             [--patch_location PATCH_LOCATION]
                             [--patch_overlap PATCH_OVERLAP] --hd5_location
                             HD5_LOCATION [--generate_heatmap]
                             [--heatmap_location HEATMAP_LOCATION]
                             [--classification_threshold CLASSIFICATION_THRESHOLD]
                             [--classification_max_threshold CLASSIFICATION_MAX_THRESHOLD]
                             [--label LABEL] --patch_size PATCH_SIZE
                             [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
                             [--evaluation_size EVALUATION_SIZE] [--is_tumor]
                             [--num_patch_workers NUM_PATCH_WORKERS]
                             [--gpu_id GPU_ID] [--num_gpus NUM_GPUS]
                             [--old_version] [--slide_idx SLIDE_IDX]
                             [--maximum_number_patches MAXIMUM_NUMBER_PATCHES [MAXIMUM_NUMBER_PATCHES ...]]
                             [--use_radius] [--radius RADIUS]
                             [--use_color_norm] [--method METHOD]
                             [--reference_image REFERENCE_IMAGE]
                             [--use_standarizer]
                             {use-manifest,use-directory} ...

positional arguments:
  {use-manifest,use-directory}
                        Specify how to load slides to annotate.
                            There are 2 ways: by manifest and by directory.
    use-manifest        Use manifest file to locate slides.
                                a CSV file with minimum of 4 column and maximum of 6 columns. The name of columns
                                should be among ['origin', 'patient_id', 'slide_id', 'slide_path', 'annotation_path', 'subtype'].
                                origin, slide_id, patient_id must be one of the columns.

    use-directory       Use a rootdir to locate slidesIt is expected that slide paths have the structure '/path/to/rootdir/slide_pattern/slide_name.extension' where slide_pattern is usually 'subtype'. Patient IDs are extrapolated from slide_name using known, hardcoded regex.

optional arguments:
  -h, --help            show this help message and exit

  --log_file_location LOG_FILE_LOCATION
                        Path to the log file produced during training.
                         (default: None)

  --log_dir_location LOG_DIR_LOCATION
                        Path to log directory to save testing logs (i.e. /path/to/logs/testing/).
                         (default: None)

  --store_extracted_patches
                        Store extracted patches. Default does not store extracted patches.
                         (default: False)

  --store_thumbnail     Whether or not save thumbnail with showing the position of extracted patches. If yes, it will be stored at a folder called Thumbnails in HD5 folder.
                         (default: False)

  --generate_annotation
                        Whether or not save annotation for slide If yes, it will be stored at a folder called Annotation in HD5 folder. Also a folder called Thubmnails will be created in Annotation that shows the annotation on thumbnails. Only works for Tumor.
                         (default: False)

  --skip_area SKIP_AREA
                        If this flag is set, when the final annotation is created, polygons (areas) with less than area of determined will be skipped. note that smallest area is patch_size*patch_size.
                         (default: None)

  --patch_location PATCH_LOCATION
                        Path to root directory to extract patches into.
                         (default: ./)

  --patch_overlap PATCH_OVERLAP
                        Overlap between extracted patches.
                         (default: 0)

  --hd5_location HD5_LOCATION
                        Path to root directory to save hd5 into.
                         (default: None)

  --generate_heatmap    Generate heatmaps. Default does not generate heatmap.
                         (default: False)

  --heatmap_location HEATMAP_LOCATION
                        Path to directory to save the heatmap H5 files (i.e. /path/to/heatmaps/).
                         (default: ./)

  --classification_threshold CLASSIFICATION_THRESHOLD
                        Minimum obtained probability for the most probable class
                         (default: 0)

  --classification_max_threshold CLASSIFICATION_MAX_THRESHOLD
                        Maximum obtained probability for the most probable class
                         (default: 1.0)

  --label LABEL         Only search for this label in output probability of the modeluseful when you set the --classification_threshold threshold and you wantconsider only one of the labels such as tumor
                         (default: None)

  --patch_size PATCH_SIZE
                        Patch size in pixels to extract from slide to use in evaluation.
                         (default: 1024)

  --resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]
                        List of patch sizes in pixels to resize the extracted patchs and save. Each size should be at most patch_size. Default does not resize.
                         (default: None)

  --evaluation_size EVALUATION_SIZE
                        The size in pixel to resize patch before passing to model for evaluation. evaluation_size should be one of resize_sizes or set to patch_size. Default uses patch of patch_size for evaluation.
                         (default: None)

  --is_tumor            Only extract tumor patches. Default extracts tumor and normal patches.
                         (default: False)

  --num_patch_workers NUM_PATCH_WORKERS
                        Number of loader worker processes to multi-process data loading. Default uses single-process data loading.
                         (default: 0)

  --gpu_id GPU_ID       The ID of GPU to select. Default uses GPU with the most free memory.
                         (default: None)

  --num_gpus NUM_GPUS   The number of GPUs to use. Default uses a GPU with the most free memory.
                         (default: 1)

  --old_version         Convert trained model on previous version to the current one
                         (default: False)

  --slide_idx SLIDE_IDX
                        Select a specif slide from all the slides in that directory (usefull for running multiple jobs).
                         (default: None)

  --maximum_number_patches MAXIMUM_NUMBER_PATCHES [MAXIMUM_NUMBER_PATCHES ...]
                        Caution: when you use this flag the code while shuffles the extracted patches from each slide.space separated words describing subtype=maximum_number_of_extracted_patches pairs for each slide. Example: if want to extract 500 Tumor, 0 Normal patches and unlimited POLE patches then the input should be 'Tumor=500 Normal=0 POLE=-1'
                         (default: {})

  --use_radius          Activating this subparser will enable extracting all patches within radius of the coordinate.
                         (default: False)

  --radius RADIUS       From each selected coordinate, all its neighbours will be extracted. This number will be multiplied by the patch size.Note: In use-annotation, the number will be multiplied*stride.
                         (default: 1)

  --use_color_norm      Whether use normlization of patches before feeding to the model or not.
                         (default: False)

  --method METHOD       The Normalization method.
                         (default: vahadane)

  --reference_image REFERENCE_IMAGE
                        The path to reference image for normalization.
                         (default: None)

  --use_standarizer     Whether to apply brighness standarizer on the images.
                         (default: False)

usage: app.py from-arguments use-manifest [-h] --manifest_location
                                          MANIFEST_LOCATION

optional arguments:
  -h, --help            show this help message and exit

  --manifest_location MANIFEST_LOCATION
                        Path to manifest CSV file.
                         (default: None)

usage: app.py from-arguments use-directory [-h] --slide_location
                                           SLIDE_LOCATION
                                           [--slide_pattern SLIDE_PATTERN]
                                           [--mask_location MASK_LOCATION]

optional arguments:
  -h, --help            show this help message and exit

  --slide_location SLIDE_LOCATION
                        Path to root directory containing all of the slides.
                         (default: None)

  --slide_pattern SLIDE_PATTERN
                        '/' separated words describing the directory structure of the slide paths. Normally slides paths look like /path/to/slide/rootdir/subtype/slide.svs and if slide paths are /path/to/slide/rootdir/slide.svs then simply pass ''.
                         (default: subtype)

  --mask_location MASK_LOCATION
                        Path to root directory which contains mask for tissue selection. It should contain png files or annotation file with label clear_area.
                         (default: None)

```

**Caution: you need to check the error log constantly to monitor _CUDA out of memory_ error. You need to use a smaller number for _--num_patch_workers_ if you get this error.**


In order to increase the speed of auto_annotate, We should run parralel jobs. In order to achieve this, you should use this bash script file:
```
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
 --slide_pattern=
 --patch_size 1024
 --resize_sizes 512
 --evaluation_size 512
 --is_tumor True
 --store_extracted_patches True
 --classification_threshold 0.9
 --num_patch_workers 1
 --slide_idx $SLURM_ARRAY_TASK_ID
 --maximum_number_patches Tumor=400 Stroma=400

```

The number of arrays should be set to value of `num_slides / num_patch_workers`.
For fastest way, set the `num_patch_workers=1`, then number of arrays is `num_slides`.
If you want to extracted tumor patches with probability between 0.4 and 0.6, you should set `classification_threshold=0.4`, `classification_max_threshold=0.6`, and `label=Tumor`.
