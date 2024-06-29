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

Note that the data now is just a trial run! For the first run, we had 11k+ unique results from just a trial run.
For the second run, that went up to `19544`. When I added more applications, for half of the run it was 25k.
The current total is `31932` scripts. I didn't add the last run of flux because I saw what I thought were false positives.

## Analysis

### 1. Word2Vec

Word2Vec is a little old, and I think a flaw is that it is combining jobspecs. But if we have the window the correct size, we can make associations between close terms.
The space I'm worried about is the beginning of one script and the end of another, and maybe a different approach or strategy could help with that.
To generate the word2vec embeddings you can run:

```bash
python scripts/word2vec.py --input ./data
```

Updates to the above on June 9th:

- Better parsing to tokenize 
  - we combine by space instead of empty space so words at end are not combined (this was a bug)
  - punctuation that should be replaced by space instead of empty space honored (dashes, underscore, etc)
  - hash bangs for shell parsed out
  - better tokenization and recreation of content
  - each script is on one line (akin to how done for word2vec)

I think it would be reasonable to create a similarity matrix, specifically cosine distance between the vectors.
This will read in the metadata.tsv and vectors.tsv we just generated.

```bash
python scripts/vector_matrix.py --vectors ./scripts/data/combined/vectors.tsv --metadata ./scripts/data/combined/metadata.tsv
```

The above does the following:

1. We start with our jobspecs that are tokenized according to the above.
2. We further remove anything that is purely numerical
3. We use TF-IDF to reduce the feature space to 300 terms
4. We do a clustering of these terms to generate the resulting plot.

The hardest thing is just seeing all the terms. I messed with JavaScript for a while but gave up for the time being, the data is too big for the browser
and likely we need to use canvas.

### 2. Directive Counts

I thought it would be interesting to explicitly parse the directives. That's a bit hard, but I took a first shot:

```bash
python scripts/parse_directives.py --input ./data
```
```console
Assessing 33851 conteder jobscripts...
Found (and skipped) 535 duplicates.
```

You can find tokenized lines (with one jobspec per line), the directive counts, and the dictionary and skips in [scripts/data/combined/](scripts/data/combined/)

### 3. Adding Topics or More Structure

I was thinking about adding doc2vec, because word2vec is likely making associations between terms in different documents,
but I don't think anyone is using doc2vec anymore, because the examples I'm finding using a deprecated version of tensorflow that
has functions long removed. We could use the old gensim version, but I think it might be better to think of a more modern approach.
I decided to try [top2vec](https://github.com/ddangelov/Top2Vec).

```bash
# Using pretrained model (not great because not jobscript terms)
python scripts/run_top2vec.py

# Build with doc2vec - be careful we set workers and learn mode (slower) here
# started at 7pm
python3 scripts/run_top2vec_with_doc2vec.py --speed learn
python3 scripts/run_top2vec_with_doc2vec.py --speed deep-learn
```

And then to explore (finding matches for a subset of words):

```
python3 scripts/explore_top2vec.py
python3 scripts/explore_top2vec.py --outname top2vec-jobspec-database-learn.md --model ./scripts/data/combined/wordclouds/top2vec-with-doc2vec-learn.model 

# Deep learn (highest quality vectors), takes about 6-7 hours to run 128 GB ram CPU instance
python3 scripts/explore_top2vec.py --outname top2vec-jobspec-database-deep-learn.md --model ./scripts/data/combined/wordclouds/top2vec-with-doc2vec-deep-learn.model 
```

For word2vec:

 - continuous bag of words: we create a window around the word and predict the word from the context
 - skip gram: we create the same window but predict the context from the word (supposedly slower but better results)

I had to run this on a large VM for it to work. See the topics in [scripts/data/combined/wordclouds](scripts/data/combined/wordclouds). We can likely tweak everything but I like how this tool is approaching it (see docs in [ddangelov/Top2Vec](https://github.com/ddangelov/Top2Vec)).

### 4. Gemini

We can run Gemini across our 33K jobspecs to generate a templatized output for each one:

```bash
python scripts/classify-gemini.py 
```

That takes a little over a day to run, and it will cost about 25-$30 per run. I did two runs for about $55.
Then we can both check the model, normalize and visualize our resources (that we parsed) and compare to what Gemini says. 

```bash
python scripts/process-gemini.py
```

You can then see the data output in [scripts/data/gemini-with-template-processed](scripts/data/gemini-with-template-processed) or use this script to visualize results that are filtered to those with all, missing, or some wrong values:

```bash
# pip install rich
python scripts/inspect-gemini.py

# How to customize
python scripts/inspect-gemini.py --type missing
python scripts/inspect-gemini.py --type wrong

# Print more than 1
python scripts/inspect-gemini.py --type all --number 3
```

#### 4. LC Jobspec Database

This database is kind of messy - not sure I like it as much as the one I generated. Someone else can deal with it :)

- Total unique jobspec jsons: 210351
- Total with BatchScript: 116117

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614

