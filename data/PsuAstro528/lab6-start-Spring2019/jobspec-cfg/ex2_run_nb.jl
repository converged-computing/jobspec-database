# First, make sure that all package cahces are up-to-date on master process
import Pkg
Pkg.activate(".")
# Now make sure tha all workers have access to packages
using Distributed
@everywhere import Pkg
@everywhere Pkg.activate(".")
using NBInclude
println("# About to run notebook")
flush(stdout)
@nbinclude("ex2.ipynb")
println("# Completed running notebook")
flush(stdout)

