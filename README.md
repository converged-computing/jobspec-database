# Job Specification Database

This database is 🚧️ under developement! 🚧️

It will eventually be added to 🥑️ [Dinosaur Datasets](https://vsoch.github.io/datasets/). 🥑️

## Usage

The data files are organized by repository in [data](data). These instructions are for generation. Create a python environment and install dependencies:

```bash
pip install -r requirements.txt
```

You'll need to make a "drivers" directory and download the chromedriver (matching your browser) to it inside of scripts. Then, run the parsing script, customizing the matrix of search terms. You should have a chromedriver installed, all browsers closed, and be prepared to login to GitHub.

```bash
cd scripts/
python search.py
```

Then download files, from the root, targeting the output file of interest.

```bash
python scripts/get_jobspecs.py ./scripts/data/raw-links-may-23.json --outdir ./data
```

Note that the data now is just a trial run! We have 11k+ unique results from just that, wowza.
We will want to optimize our search term strategy to get across a set of applications and scopes. After this we currently have `19544` scripts.

## Analysis

You can run:

```bash
python scripts/word2vec.py --input ./data
```

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614

