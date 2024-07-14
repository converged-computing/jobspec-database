# Extract Annotated Patches

The Extract Annotated Patches project is a powerful tool for extracting specific patches from large biomedical images. This tool is highly configurable and can be used in a variety of contexts, including the extraction of patches from HD5 files, directories, or manifest files. The tool supports multiple methods for specifying the coordinates of the patches to be extracted, including slide coordinates, annotations, or the entire slide.

The tool is command-line based and has different subcommands for different use cases. The tool uses a main argument to specify whether the command line arguments are sourced from an experiment manifest or from the command line. Each subcommand has its own set of arguments.

## Running in Parallel

To increase the speed of the `extract_annotated_patches`, parallel jobs can be run. This can be achieved by using the provided bash script file. This script uses SLURM for parallel job scheduling. The `--array` argument should be set to the value of `num_slides / num_patch_workers`. For the fastest execution, set `num_patch_workers=1`, then the number of arrays is `num_slides`.

### Usage ###
```

usage: app.py [-h] {from-experiment-manifest,from-arguments} ...

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

usage: app.py from-arguments [-h] --hd5_location HD5_LOCATION [--seed SEED]
                             [--num_patch_workers NUM_PATCH_WORKERS]
                             [--store_thumbnail]
                             {from-hd5-files,use-manifest,use-directory} ...

positional arguments:
  {from-hd5-files,use-manifest,use-directory}
                        Specify how to load slides to extract.
                            There are 3 ways of extracting slides: from hd5 files, by manifest and by directory.
    from-hd5-files      uses pre created hd5 files located at hd5_location to create and store images in patch_location

    use-manifest        Use manifest file to locate slides.
                                a CSV file with minimum of 4 column and maximum of 6 columns. The name of columns
                                should be among ['origin', 'patient_id', 'slide_id', 'slide_path', 'annotation_path', 'subtype'].
                                origin, slide_id, patient_id must be one of the columns.

    use-directory       Use a rootdir to locate slides.
                            It is expected that slide paths have the structure '/path/to/rootdir/slide_pattern/slide_name.extension' where slide_pattern is usually 'subtype'. Patient IDs are extrapolated from slide_name using known, hardcoded regex.

optional arguments:
  -h, --help            show this help message and exit

  --seed SEED           Seed for random shuffle.
                         (default: 256)

  --num_patch_workers NUM_PATCH_WORKERS
                        Number of worker processes to multi-process patch extraction. Default sets the number of worker processes to the number of CPU processes.
                         (default: None)

  --store_thumbnail     Whether or not save thumbnail with showing the position of extracted patches. If yes, it will be stored at a folder called Thumbnails in HD5 folder.
                         (default: False)

required arguments:
  --hd5_location HD5_LOCATION
                        Path to root directory to save hd5 into.
                         (default: None)

usage: app.py from-arguments use-manifest [-h] --patch_location PATCH_LOCATION
                                          --manifest_location
                                          MANIFEST_LOCATION
                                          [--slide_idx SLIDE_IDX]
                                          [--store_extracted_patches]
                                          {use-slide-coords,use-annotation,use-entire-slide,use-mosaic}
                                          ...

positional arguments:
  {use-slide-coords,use-annotation,use-entire-slide,use-mosaic}
                        Specify which coordinates in the slides to extract.
                                There are 3 ways of extracting patches: by slide_cords, by annotation, and by entire_slide.
    use-slide-coords    Specify patches to extract using slide coordinates.
                                A slide coords JSON file containing keys 'patch_size' and 'coords'.
                                The key 'patch_size' gives the pixel width, height of the patch.
                                Coordinates are a list of size 2 lists of numbers representing the x, y pixel coordinates.
                                The [x, y] list represents the coordinates of the top left corner of the patch_size * patch_size extracted patch.
                                Coordinates are indexed by slide name for the slide the patches are from, and annotation the patches are labeled with.
                        
                                {
                                    patch_size: int,
                                    coords: {
                                        [slide name]: {
                                            [label]: [
                                                [int|float, int|float],
                                                ...
                                            ],
                                            ...
                                        },
                                        ...
                                    }
                                }

    use-annotation      Specify patches to extract by annotation.
                                If a slide is named 'VOA-1823A' then the annotation file for that slide is a text file named 'VOA-1823A.txt' with each line containing (i.e.):
                        
                                Tumor [Point: 84332.8046875, 68421.28125, Point: 84332.8046875, 68421.28125,...]
                                Stroma [...]

    use-entire-slide    Extracting patches from the whole slide. In this way,
                                both tumor and normal areas will be extracted.
                                The label is called Mix.

    use-mosaic          Selecting patches from the whole slide in an efficient way.
                                In this way, first all patches will be clustered based on their RGB histogram.
                                Then, in each cluster, another clustering will be applied for their coordiantes.

optional arguments:
  -h, --help            show this help message and exit

  --slide_idx SLIDE_IDX
                        Positive Index for selecting part of slides instead of all of it. (useful for array jobs)
                         (default: None)

  --store_extracted_patches
                        Whether or not save extracted patches as png files on the disk.
                         (default: False)

required arguments:
  --patch_location PATCH_LOCATION
                        Path to root directory to extract patches into.
                         (default: None)

  --manifest_location MANIFEST_LOCATION
                        Path to manifest CSV file.
                         (default: None)

usage: app.py from-arguments from-hd5-files [-h] --slide_location
                                            SLIDE_LOCATION
                                            [--slide_pattern SLIDE_PATTERN]
                                            [--slide_idx SLIDE_IDX]
                                            [--resize RESIZE [RESIZE ...]]
                                            [--max_num_patches MAX_NUM_PATCHES]

optional arguments:
  -h, --help            show this help message and exit

  --slide_location SLIDE_LOCATION
                        Path to slide rootdir.
                         (default: None)

  --slide_pattern SLIDE_PATTERN
                        '/' separated words describing the directory structure of the slide paths. Normally slides paths look like /path/to/slide/rootdir/subtype/slide.svs and if slide paths are /path/to/slide/rootdir/slide.svs then simply pass ''.
                         (default: subtype)

  --slide_idx SLIDE_IDX
                        Positive Index for selecting part of slides instead of all of it. (useful for array jobs)
                         (default: None)

  --resize RESIZE [RESIZE ...]
                        List for determining desired resize. For example, if the HDF5 file has [256, 512, 1024] patches, and we are only interested in 256, we set this flag. [256]
                         (default: None)

  --max_num_patches MAX_NUM_PATCHES
                        Maximum number of patches we want to extract. For example, if there are 2000 patches from each HDF5, and we only need first 500 ones, we set this flag to 500. NOTE: The patches are same order that was supposed to be extracted.
                         (default: None)

usage: app.py from-arguments use-directory [-h] --patch_location
                                           PATCH_LOCATION --slide_location
                                           SLIDE_LOCATION
                                           [--store_extracted_patches]
                                           [--slide_idx SLIDE_IDX]
                                           [--slide_pattern SLIDE_PATTERN]
                                           [--mask_location MASK_LOCATION]
                                           {use-slide-coords,use-annotation,use-entire-slide,use-mosaic}
                                           ...

positional arguments:
  {use-slide-coords,use-annotation,use-entire-slide,use-mosaic}
                        Specify which coordinates in the slides to extract.
                                There are 3 ways of extracting patches: by slide_cords, by annotation, and by entire_slide.
    use-slide-coords    Specify patches to extract using slide coordinates.
                                A slide coords JSON file containing keys 'patch_size' and 'coords'.
                                The key 'patch_size' gives the pixel width, height of the patch.
                                Coordinates are a list of size 2 lists of numbers representing the x, y pixel coordinates.
                                The [x, y] list represents the coordinates of the top left corner of the patch_size * patch_size extracted patch.
                                Coordinates are indexed by slide name for the slide the patches are from, and annotation the patches are labeled with.
                        
                                {
                                    patch_size: int,
                                    coords: {
                                        [slide name]: {
                                            [label]: [
                                                [int|float, int|float],
                                                ...
                                            ],
                                            ...
                                        },
                                        ...
                                    }
                                }

    use-annotation      Specify patches to extract by annotation.
                                If a slide is named 'VOA-1823A' then the annotation file for that slide is a text file named 'VOA-1823A.txt' with each line containing (i.e.):
                        
                                Tumor [Point: 84332.8046875, 68421.28125, Point: 84332.8046875, 68421.28125,...]
                                Stroma [...]

    use-entire-slide    Extracting patches from the whole slide. In this way,
                                both tumor and normal areas will be extracted.
                                The label is called Mix.

    use-mosaic          Selecting patches from the whole slide in an efficient way.
                                In this way, first all patches will be clustered based on their RGB histogram.
                                Then, in each cluster, another clustering will be applied for their coordiantes.

optional arguments:
  -h, --help            show this help message and exit

  --store_extracted_patches
                        Whether or not save extracted patches as png files on the disk.
                         (default: False)

  --slide_idx SLIDE_IDX
                        Positive Index for selecting part of slides instead of all of it. (useful for array jobs)
                         (default: None)

  --slide_pattern SLIDE_PATTERN
                        '/' separated words describing the directory structure of the slide paths. Normally slides paths look like /path/to/slide/rootdir/subtype/slide.svs and if slide paths are /path/to/slide/rootdir/slide.svs then simply pass '' (using bash script, it is impossible to pass '', therefore in that case, use "'".).
                         (default: subtype)

  --mask_location MASK_LOCATION
                        Path to root directory which contains mask for tissue selection. It should contain png files or annotation file with label clear_area.
                         (default: None)

required arguments:
  --patch_location PATCH_LOCATION
                        Path to root directory to extract patches into.
                         (default: None)

  --slide_location SLIDE_LOCATION
                        Path to slide rootdir.
                         (default: None)

usage: app.py from-arguments use-directory use-slide-coords
       [-h] --slide_coords_location SLIDE_COORDS_LOCATION

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --slide_coords_location SLIDE_COORDS_LOCATION
                        Path to slide coords JSON file.
                         (default: None)

usage: app.py from-arguments use-directory use-annotation [-h]
                                                          --annotation_location
                                                          ANNOTATION_LOCATION
                                                          --slide_coords_location
                                                          SLIDE_COORDS_LOCATION
                                                          [--patch_size PATCH_SIZE]
                                                          [--stride STRIDE]
                                                          [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
                                                          [--is_tumor]
                                                          [--is_TMA]
                                                          [--patch_overlap PATCH_OVERLAP]
                                                          [--annotation_overlap ANNOTATION_OVERLAP]
                                                          [--max_slide_patches MAX_SLIDE_PATCHES]
                                                          [--use_radius]
                                                          [--radius RADIUS]

optional arguments:
  -h, --help            show this help message and exit

  --patch_size PATCH_SIZE
                        Patch size in pixels to extract from slide.
                         (default: 1024)

  --stride STRIDE       Stride in pixels which determines the gap between each two extracted patches. NOTE: This value will be added with the patch_size for actual stride.For example, if patch_size is 2048 and stride is 2000, the actual stride is 2000+2048=4048.
                         (default: 0)

  --resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]
                        List of patch sizes in pixels to resize the extracted patches and save. Each size should be at most patch_size. Default simply saves the extracted patch.
                         (default: None)

  --is_tumor            Only extract tumor patches. Default extracts tumor and normal patches.
                         (default: False)

  --is_TMA              TMA cores are simple image instead of slide.
                         (default: False)

  --patch_overlap PATCH_OVERLAP
                        Overlap between extracted patches.
                         (default: 0)

  --annotation_overlap ANNOTATION_OVERLAP
                        Patches having overlapp above this value with the annotated pixels will be extracted.
                         (default: 1.0)

  --max_slide_patches MAX_SLIDE_PATCHES
                        Select at most max_slide_patches number of patches from each slide.
                         (default: None)

  --use_radius          Activating this subparser will enable extracting all patches within radius of the coordinate.
                         (default: False)

  --radius RADIUS       From each selected coordinate, all its neighbours will be extracted. This number will be multiplied by the patch size.Note: In use-annotation, the number will be multiplied*stride.
                         (default: 1)

required arguments:
  --annotation_location ANNOTATION_LOCATION
                        Path to immediate directory containing slide's annotation TXTs.
                         (default: None)

  --slide_coords_location SLIDE_COORDS_LOCATION
                        Path to slide coords JSON file to save extracted patch coordinates.
                         (default: None)

usage: app.py from-arguments use-directory use-entire-slide
       [-h] --slide_coords_location SLIDE_COORDS_LOCATION
       [--patch_size PATCH_SIZE] [--stride STRIDE]
       [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
       [--max_slide_patches MAX_SLIDE_PATCHES] [--use_radius]
       [--radius RADIUS]

optional arguments:
  -h, --help            show this help message and exit

  --patch_size PATCH_SIZE
                        Patch size in pixels to extract from slide.
                         (default: 1024)

  --stride STRIDE       Stride in pixels which determines the gap between each two extracted patches. NOTE: This value will be added with the patch_size for actual stride.For example, if patch_size is 2048 and stride is 2000, the actual stride is 2000+2048=4048.
                         (default: 0)

  --resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]
                        List of patch sizes in pixels to resize the extracted patches and save. Each size should be at most patch_size. Default simply saves the extracted patch.
                         (default: None)

  --max_slide_patches MAX_SLIDE_PATCHES
                        Select at most max_slide_patches number of patches from each slide.
                         (default: None)

  --use_radius          Activating this subparser will enable extracting all patches within radius of the coordinate.
                         (default: False)

  --radius RADIUS       From each selected coordinate, all its neighbours will be extracted. This number will be multiplied by the patch size.Note: In use-annotation, the number will be multiplied*stride.
                         (default: 1)

required arguments:
  --slide_coords_location SLIDE_COORDS_LOCATION
                        Path to slide coords JSON file to save extracted patch coordinates.
                         (default: None)

usage: app.py from-arguments use-directory use-mosaic [-h]
                                                      --slide_coords_location
                                                      SLIDE_COORDS_LOCATION
                                                      [--patch_size PATCH_SIZE]
                                                      [--evaluation_size EVALUATION_SIZE]
                                                      [--n_clusters N_CLUSTERS]
                                                      [--percentage PERCENTAGE]
                                                      [--stride STRIDE]
                                                      [--resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]]
                                                      [--use_radius]
                                                      [--radius RADIUS]

optional arguments:
  -h, --help            show this help message and exit

  --patch_size PATCH_SIZE
                        Patch size in pixels to extract from slide.
                         (default: 1024)

  --evaluation_size EVALUATION_SIZE
                        Patch size in pixels to calculate clusters based on that. This should be at lower resolution (e.g. 5x) since it has more contexual information.
                         (default: 128)

  --n_clusters N_CLUSTERS
                        Number of color clusters. This value should be selected based on the slide.
                         (default: 9)

  --percentage PERCENTAGE
                        Percentage of patches to build the mosaic.
                         (default: 0.05)

  --stride STRIDE       Stride in pixels which determines the gap between each two extracted patches. NOTE: This value will be added with the patch_size for actual stride.For example, if patch_size is 2048 and stride is 2000, the actual stride is 2000+2048=4048.
                         (default: 0)

  --resize_sizes RESIZE_SIZES [RESIZE_SIZES ...]
                        List of patch sizes in pixels to resize the extracted patches and save. Each size should be at most patch_size. Default simply saves the extracted patch.
                         (default: None)

  --use_radius          Activating this subparser will enable extracting all patches within radius of the coordinate.
                         (default: False)

  --radius RADIUS       From each selected coordinate, all its neighbours will be extracted. This number will be multiplied by the patch size.Note: In use-annotation, the number will be multiplied*stride.
                         (default: 1)

required arguments:
  --slide_coords_location SLIDE_COORDS_LOCATION
                        Path to slide coords JSON file to save extracted patch coordinates.
                         (default: None)
```



In order to increase the speed of extract_annotated_patches, We should run parallel jobs. In order to achieve this, you should use this bash script file:
```
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

singularity run -B /projects/ovcare/classification -B /projects/ovcare/WSI singularity_extract_annotated_patches.sif from-arguments \
 --hd5_location path/to/folder \
 --num_patch_workers 1 \
 use-directory \
 --patch_location path/to/folder \
 --slide_location path/to/folder \
 --slide_pattern subtype \
 --slide_idx $SLURM_ARRAY_TASK_ID \
 --store_extracted_patches \
 use-entire-slide \
 --slide_coords_location path/to/file \
 --patch_size 2048 \
 --stride 2000 \
 --resize_sizes 512 \
 # --max_slide_patches 10


```

1. The number of arrays should be set to value of `num_slides / num_patch_workers`.
2. For fastest way, set the `num_patch_workers=1`, then number of arrays is `num_slides`.

