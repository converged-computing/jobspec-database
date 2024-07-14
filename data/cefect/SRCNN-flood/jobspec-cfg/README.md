# SRCNN

Implementation of Dong (2014)'s ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092) to flood depth rasters.

Model structure [is here](\cnnf\models) and has been modified from Dong (2014)'s 3x2D convolution layer 

NOTE: test, train, and validation data is stored separately

## Related projects
[2307_super](https://github.com/cefect/2307_super): analysis/prep of RIM2D model outputs.

[SRCNN-pytorch](https://github.com/cefect/SRCNN-pytorch): Dong (2014)'s original implementation (my fork).

## Install 
1) create a ./definitions.py file as shown below
2) build environment (conda) using environmnet.yml. if only training a model (e.g., HPC applications), a more lightweight install is provided in environment_train.yml

### definitions.py
```
import os
wrk_dir = r'l:\10_IO\2307_super' #default working directory
test_data_dir = os.path.join(wrk_dir, 'tests') #directory containing test data
```

## Structure
- workflows: scripts for executing specific workflows of the project (e.g., train, test, report)
    - unix: bash scripts for SLURM cluster runs
- cnnf: python source code
- implementations: project workflows for specific case studies
- tests: pytest scripts
- env (or env2): machine environment (.gitignored)


## Basic Use
see [implementations/ahr_1206.py](/implementations/ahr_1206.py) for typical implementation.

### Compile training data
1) `prepare.pipeline_prep_multiband()` for each scenario, construct a multiband GeoTiff (b1:target, b2:input, b3:DEM)

2) `prepare.port_to_h5()` compile all scenarios into h5 database (grouped by GeoTiff chunks)


### Compile evaluation data
same as above but for a single scenario

### Train CNN weights
1) `train.train_model()` compute model (`models.SRCNN`) weights

### Report on training run
1) `report.plot_metric_v_backprops()` plot key metrics vs. backproagation 
2) `eval_calibrated.apply_all_epochs()` construction prediction (downscaled) WSH for each epoch
3) `report.plot_output_matrix()` plot matrix of epochs showing prediction rasters


## Cluster Use
bash scripts for cluster runs are provided in [\workflows\unix\train_test.sh](\workflows\unix\train_test.sh).
For cluster training, the lightweight environment is recommended.




 



 