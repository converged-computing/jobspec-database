# Contributing

## Introduction
<!--  Contributions welcome.-->
We encourage contributions to DeepArchitect.
If DeepArchitect has been useful for your work, please cite it and/or contribute to the codebase.
We encourage everyone doing research in architecture search to implement their algorithms in DeepArchitect to make them widely available to other researchers and the machine learning community at large.
This will significantly improve reproducibility and reusability of architecture search research.

<!-- Information that you will find in this document. -->
Contributions can be a result of your own research or from implementations of existing algorithms.
If you have developed a searcher, search space, evaluator, or any other component or functionality that would be useful to include in DeepArchitect, please make a pull request that follows the guidelines described in this document.

After reading this document, you will understand:

* what are the different types of contributions that we identify;
* what is the folder structure for contributions;
* what is required in terms of tests and documentation for different types of contributions;
* what are the different levels of conformity that we require for different types of contributions.

<!-- How to decide exactly what to contribute. -->
If you have a feature that you would like to add to DeepArchitect but you are unsure about its suitability, open a [GitHub issue](https://github.com/negrinho/deep_architect/issues) for discussion.
This guarantees that your efforts are well-aligned with the project direction and needs.
Consider including a code snippet or pseudo-code illustrating a useful use case for the feature.

## Types of contributions
<!-- The contrib and dev folders and their semantics. -->
Most contributions will live in the contrib folder.
The contrib folder is used for functionality that is likely useful, but that won't necessarily be maintained over time.
While code lies in the contrib folder, the code owners are responsible for its maintenance.
If code in the contrib folder breaks and the code owner does not fix it in a timely manner, we reserve the right to move the code to the dev folder.
The dev folder contains code sketching interesting functionality that may not be fully functional or it has not been refactored well enough to be integrated in contrib.
Unmaintained code will be moved to dev upon breakage. Code in dev should not be used directly, but it can inspire further development.

<!-- How code evolves between the different folders. -->
Code that is part of the contrib folder may eventually be refactored into code that is part of the deep_architect folder. Similarly, code in the dev folder may be refactored in code that goes in the contrib folder.
If code becomes part of the deep_architect folder, it becomes the responsibility of the developers of DeepArchitect to maintain it.
To create a new contrib folder, it is best to first discuss its scope.
We do not impose these restrictions for the dev folder.
The dev folder should be used lightly though.
We will only accept contributions to the dev folder if it they showcase important functionality and there is sufficient reason to justify their incompleteness.
If the functionality is complete, we advise the contributor to refactor it into the contrib folder.
Including the contribution in the contrib folder can be done either by adding it to an existing contrib subfolder, or by creating a new well-scoped contrib subfolder.

<!-- Contributions that are generated by extensively adapting existing code. -->
Cross-pollination between contrib and dev folders is expected and encouraged.
For example, a few subcontrib folders already contain useful functionality, but a contributor may want to extend it and encapsulate it in a more coherent contrib subfolder.
This scheme allows DeepArchitect to evolve without committing to major refactoring decisions upfront.
If the foreseen contribution is better seen as an extension or a fix to an existing contrib folder, please open an issue or a pull request to discuss with the most active contributors on how to best incorporate the contribution in the existing files and folders.
We may ask for refactoring changes or additional tests.

## Required documentation and tests
<!-- Folder structure for contrib contributions. -->
Your new library in contrib should be placed in `deep_architect/contrib/$YOUR_LIBRARY_NAME`.
New folders in contrib should include a `README.md` file providing information about the functionality that the library seeks to implement, the features that are implemented in the folder contributed, and an explanation about how the implementation is split between the different files and folders.
Also include an explanation about when would it be natural to use the code in this library.
This guarantees that a new user will quickly get a reasonable grasp of how to use the library and what files to look at for specific desired functionality.
Comments for each major class and function are also recommended but not mandatory.
Check the comments in `deep_architect/core.py` to get a sense of the style and format used for comments.
It is also convenient to include in `README.md`, a roadmap for missing functionality that would be nice to include in the future.
This informs future contributors about where the contributed project is going and compels them to help, e.g., if they believe that the feature is important.

<!-- README file and its Structure. -->
The following is a typical structure for `README.md`:
explanation of the problem that the contributed code tries to solve,
some example code, a brief description of the high-level organization of the
contributed library, and a roadmap for future work items and nice-to-haves
and how other people can contribute to it, additional comments, GitHub handles
of the code owners.
If another contributor would like to extend an existing contributed library,
it is best to reach out to the appropriate owner by writing an issue and
mentioning the appropriate owner.
The addition of significant new functionality requires adding more tests to
exercise the newly developed code.

<!-- Test and examples. -->
In addition to `README.md`, it is convenient to add tests and examples.
The contributor should place tests in `tests/contrib/$YOUR_LIBRARY_NAME` and
examples in `examples/contrib/$YOUR_LIBRARY_NAME`.
Both `tests/contrib` and `examples/contrib` are meant to mostly reproduce the
folder structure in `deep_architect/contrib`.
This guarantees that removing a contributed library can be done easily by
removing the corresponding folders in `deep_architect/contrib`, `tests/contrib`,
and `examples/contrib`.
While an example is not required, we do require a few tests to exercise the
contributed code and have some guarantee that specific features remain correct
as the contributed code and the development environment change.

## Folder structure for contributions
<!-- Motivation for the design of the contrib folder structure,
and more details about its structure. -->
For minimizing coupling between contributions of different people, we adopt a
design similar to the one used in
[Tensorflow](https://github.com/tensorflow/tensorflow).
Namely, we have a contrib folder where each new sufficiently
different well-scoped contribution gets assigned a folder in `deep_architect/contrib`.
The name of the folder should be chosen to reflect the functionality that
lies within.
All the library code contributed by the developer will be placed in this folder.
Main files that are meant to be run should be placed in `examples/contrib`
rather than in `deep_architect/contrib`.
The same name should be used for both the folder in `deep_architect/contrib` and
in `examples/contrib`.
The subfolder in `examples/contrib` is meant for runnable code related to
or making extensive use of the library code in the `deep_architect/contrib` subfolder.
We recommend checking existing examples in the
[repo](https://github.com/negrinho/deep_architect) for determining how to
structure and document a new example appropriately.

<!-- storing configurations for running examples. -->
Each configuration to run the example should be placed in a JSON configuration
file `$CONFIG_NAME.json` in a folder named `configs` living in the same folder
of the main file of the example.
JSON configuration files guarantee that the options that determine the behavior
of running the code can be kept separated from the code itself.
This is more manageable, programmable, and configurable than having a command line interface.
This guarantees that it is easy to maintain and store many different configurations,
e.g., one configuration where the code is exercised with
few resources and another configuration where the code is exercised in a
longer run, e.g., see [here](https://github.com/negrinho/deep_architect/tree/master/examples/mnist_with_logging).
Each JSON file corresponds to a different configuration.
We suggest including a `debug.json` to run a quick experiment to
validate the functionality of both the code under `contrib/examples` and
`deep_architect/contrib`.
We recommend the use of configuration files for all but the most trivial examples.
We often use the signature `python path/to/example/main.py -- config_filepath /path/to/config.json`
for running examples, where we put all the configuration information in the JSON file.

<!-- Separating the contribution according to the different modular components
identified in the framework. -->
Whether contributing examples or libraries, we recommend identifying the
search spaces, searchers, evaluators, and datasets and splitting them into
different files, e.g., [see](https://github.com/negrinho/deep_architect/tree/master/deep_architect/searchers).
Having these components into multiple files makes the dependencies more
explicit and improves the reusability of the components.
The framework is developed around these modular components.
We recommend creating the following files when appropriate: `evaluators.py`,
`search_spaces.py`, `searchers.py`, `main.py`, and `config.json`.

## Development environment
<!-- Visual Studio as the recommended code editor to use. -->
The recommended code editor is [Visual Studio Code](https://code.visualstudio.com/)
with recommended plugins `ms-python.python`, `donjayamanne.githistory`,
`eamodio.gitlens`, `donjayamanne.jupyter`, `yzhang.markdown-all-in-one`,
`ban.spellright`. These can be installed through the extension tab or in the
command line (after Visual Studio Code has been installed) with
`code --install-extension $EXTENSION_NAME` where `EXTENSION_NAME` should be
replaced by the name of each of the extensions.

We include VS Code settings with the repo which makes uses of [yapf](https://github.com/google/yapf) to automatically format the code on save. This will allow the contributor to effortlessly maintain formatting consistency with the rest of DeepArchitect.

<!-- Singularity containers for easy running. -->
We provide Singularity and Docker containers recipes for the development environment. These can found in `containers` along with additional information on how to build them.

<!-- Python 2 and Python 3 cross compatibility. -->
We have attempted to maintain compatibility with both Python 2 and Python 3. There might exist places in the code base where this is not verified. If you find a place in the codebase that is not simultaneously Python 2 and Python 3 compatible, please issue a pull request fixing the problem.

## Code style
<!-- Guidelines on the code style to use. -->
All contributions should follow the code style used in most of the code base. When in doubt, mimic the style of `deep_architect`. Code in `deep_architect` is the most carefully designed. Getting the general gist of the design decisions that went in writing this code will help you write code that fits well with the existing code. We provide an autoformatter configuration for VS Code.

<!-- Naming guidelines for variables, functions, and files. -->
Readable variable names are preferred for function names, function arguments, class names, object attributes, object attributes, and dictionary keys. Names for iterator variables or local variables with a short lifespan can be shorter and slightly less readable. `deep_architect/core.py` (and the code in `deep_architect` in general) is a good place to get the gist of how these decisions influenced the naming conventions of the code. Function signatures should be readable without much documentation. Use four spaces for indentation. Upon submission of a pull request, some of these aspects will be reviewed to make sure that the level of conformity is appropriate for the type of contribution.

## Examples of contributions
<!-- Identification of a number of general types of contributions. -->
In this section, we identify the most natural contributions for DeepArchitect. These were identified to guarantee that DeepArchitect covers existing architecture search algorithms well. Other contributions are also very much encouraged.

### Contributing a searcher
<!-- What does a saercher do in the most widely applicable case. -->
Searchers interact with the search space through a very simple interface: the searcher can ask if all the hyperparameters are specified (and therefore, if the specified search space can be compiled to a single model that can be evaluated); if the search space is not specified, the searcher can ask for a single unspecified hyperparameter and assign a value to it. When a value is assigned to an unspecified hyperparameter, the search space transitions, which sometimes gives rise to additional unspecified hyperparameters, e.g., after choosing the number of repetitions for a repetition substitution module.

<!-- General and specific searchers. -->
The most general searchers rely solely on this simple interface. Good examples of general searchers implemented can be found [here](https://github.com/negrinho/deep_architect/tree/master/deep_architect/searchers). In more specific cases, namely in reimplementations of searchers proposed in specific architecture search papers, there is some coupling between the search space and the searcher. In this case, the developed searcher expects the search space to have certain structure or properties. We recommend these types of searchers and search spaces to be kept in a contrib folder dedicated to the specific pair.

<!-- Preferences about general versus specific models. -->
Searchers that work with arbitrary search space are preferred. Searchers that require specific properties from the search space are also often easily implemented in the framework. If the searcher requires specific search space properties, document this, e.g., by including example of search spaces that the searcher operates on, by discussing how do these differences compare with the most general case, and by discussing how are these differences supported by the DeepArchitect framework. All searchers should be accompanied by documentation, at the very least a docstring, and ideally both a docstring and an example exercising the searcher.

### Contributing a search space
<!-- What is the goal of the search space. -->
A search space encodes the set of architecture that will be under consideration by the searcher. Due to the compositionality properties of search spaces, e.g., through the use of substitution modules, or simply via the use of functions that allow us to create a larger search space from a number of smaller search spaces, new search spaces can be reused for defining yet other search spaces. A search space is as an encoding for the set of architectures that the expert finds reasonable, i.e., encodes the expert's inductive bias about the problem under consideration.

<!-- Framework independent search spaces. -->
For certain search spaces, it may make sense to develop them in a framework independent way. For example, all substitution modules are framework independent. Certain search space functionality that takes other smaller search spaces and put them together into a larger search space are also often framework independent.

Due to the flexibility of the domain specific language in DeepArchitect, it is possible to have search spaces over structures different than deep architectures, e.g., it is possible to have search spaces over scikit-learn pipelines or arithmetic circuits. Due to the generic interface that the searchers use to interface with the search space, any existing general searchers can be directly applied to the problem at hand.

The goal of introducing new search spaces may be to explore new interesting
structures and to make them available to other people that want to use them.

### Contributing an evaluator
<!-- equate the evaluator with training -->
Evaluators determine the function that we are optimizing over the search space. If the evaluator does a poor job identifying the models that we in fact care about, i.e., the models that achieve high performance when trained to convergence, then the validity of running architecture search is undermined.

It is worth to consider the introduction of new evaluators for specific tasks. For example, if people in the field have found that specific evaluators (i.e., specific ways of training the models) are necessary to achieve high-performance, then it is useful replicate them.

### Contributing a surrogate model
<!-- On the importance of surrogate models. -->
Sequential model-based optimization (SMBO) searchers use surrogate models extensively. Given a surrogate function predicting a quantity related to performance, a SMBO searcher optimizes this function (often approximately) to pick the architecture to evaluate next. The quantity predicted by the surrogate model does not need to be the performance metric of interest, it can simply be a score that preserves the ordering of the models in the space according to the performance metric of interest. Surrogate models also extend naturally to multi-objective optimization.

The quality of a surrogate function can be evaluated both by the quality of the search it induces, and by how effective it is in determining the relative ordering of models in the search space. A good surrogate functions should be able to embed the architecture to be evaluated and generate accurate predictions. It is unclear which surrogate models predict performance well. We ask contributors to explore different structures and validate their performance. Existing implementations of surrogate functions can be found [here](https://github.com/negrinho/deep_architect/tree/master/deep_architect/surrogates).

## Conclusion
<!-- What were the topics that were addressed in this document. -->
This document outlines guidelines for contributing to DeepArchitect. Having these guidelines in place guarantees that the focus is placed on the functionality developed, rather than on the specific arbitrary decisions taken to implement it. Please make sure that you understand the main points of this document, e.g., in terms of folder organization, documentation, code style, test requirements, and different types of contributions.