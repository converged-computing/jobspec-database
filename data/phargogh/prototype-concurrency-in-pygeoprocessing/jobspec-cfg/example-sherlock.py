"""Demonstrate multi-machine warping.

To execute:
    * Log into sherlock
    * Install mamba
    * Create an environment with dependencies:
        $ mamba create -p $SCRATCH/daskenv-310 -y python=3.10 pygeoprocessing
        $ mamba activate $SCRATCH/daskenv-310
        $ python -m pip install -r requirements.txt
    * run `sh_dev` to enter an interactive session
    * $ mamba activate $SCRATCH/daskenv-310
    * $ python example-sherlock.py

During execution, you can see that dask workers have spun up with
`squeue -u <username>`, and you can SSH into the node to see that the job is
indeed working.  The workspace will write to the CWD.

Future work with dask could include:
    * reporting progress bars with dask's progress utilities
    * being able to run this program from within a singularity container (the
      only way I've been able to get this to work so far has been within a
      mamba/conda environment, not within Singularity containers because of the
      lack of slurm CLI tools ... see
      https://github.com/ExaESM-WP4/Batch-scheduler-Singularity-bindings for a
      possible approach to setting up shims).
    * integrating further with the dask ecosystem to support parallelism across
      more parts of pygeoprocessing.
"""
import json
import logging
import os

import pygeoprocessing
from dask_jobqueue import SLURMCluster

logging.basicConfig(level=logging.DEBUG)


def main():
    # Defaults can be written to ~/.config/dask/jobqueue.yml
    # System-wide defaults can also live at /etc/dask/jobqueue.yml
    cluster = SLURMCluster(
        cores=1, memory="4GB", queue="normal,hns", walltime="00:10:00",
        job_script_prologue=[
            'export GDAL_CACHEMAX=2048',
        ],
    )
    cluster.scale(2)  # 2 workers with the above specs

    # at this point, all jobs have been submitted.

    from dask.distributed import Client  # how dask will connect to cluster

    client = Client(cluster)

    print(cluster.job_script())  # slurm launches a individual dask workers

    # use ssh port forwarding to access the dask dashboard
    # ssh -N -L 8787:<login node hostname> -L 8888:<login note hostname>:8888 login@<login address of cluster>
    # then go to localhost:8787/status to see the hosted dashboard

    workspace = 'workspace_dir_dask'
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    with open('input_data.json') as input_json:
        data = json.load(input_json)
        data_files = data['data']
        bbox = data['bbox_stanford']

    source_path_list = [
        f'/vsicurl/{url}' for url in data_files]
    target_path_list = [
        os.path.join(workspace, os.path.basename(url))
        for url in data_files]
    pixel_size = pygeoprocessing.get_raster_info(
        source_path_list[0])['pixel_size']

    pygeoprocessing.align_and_resize_raster_stack(
        base_raster_path_list=source_path_list,
        target_raster_path_list=target_path_list,
        resample_method_list=['near'] * len(source_path_list),
        target_pixel_size=pixel_size,
        bounding_box_mode=bbox,
        raster_align_index=0,
        executor=client.get_executor()
    )
    client.shutdown()


if __name__ == '__main__':
    main()
