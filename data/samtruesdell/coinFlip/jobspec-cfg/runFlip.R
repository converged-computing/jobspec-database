
### The code below simulates a coin flip (or multiple flips). Set nFlip to the
### number of flips you want to record. If running on multiple processors, 
### you can set nFlip to a lower number and later combine he results to plot 
### the figure. If running on the HPCC set nFlip to 1.


#### Simulation Parameters ####

# How many times to flip the code in this local simulation
nFlip <- 75

# Extra sleep time to emulate more complicated simulation (larger for
# HPCC example)
sleepTime <- 1

# Identify which folder to store the results in. Assuming runPre was just
# executed (it should have been!), identify the results folder that was
# most recently created.
resNames <- grep(pattern = 'results', # look for files containing this
                 x = list.files(),    # all files in the working directory
                 value = TRUE)        # pring out the actual folder names

folder2use <- tail(sort(resNames), 1) # ID the most recent folder

#### Coin flip loop ####

# Loop not the most efficient approach, but helps show the process
# for a more complicated simulation.

for(i in 1:nFlip){

  #### Flip the coin ####
  
  # Changing n will repeat the experiment n times (the equivolent of
  # parallel processing). Size is just the number of times to flip the
  # coin -- once for bernoulli distribution. Prob is the probability of
  # flipping a head.
  isHead <- rbinom(n = 1, 
                   size = 1, 
                   prob = 0.5)
  
  # SystemSleep to emulate longer running script...
  Sys.sleep(time = sleepTime)
  
  
  #### Generate Output File Info ####
  
  # Generate name for output file; use date & time + random number so that
  # the file paths from multiple simulations are not overwritten. Can't use
  # i because of parallel processing overwrites.
  
  # get and format the system time
  time <- Sys.time()
  timeString <- format(x = time, 
                       format = "%Y-%m-%d-%H-%M-%S")
  
  # generate a (rounded) random number to append to reduce overwriting data
  randomNumber <- round(runif(n = 1, 
                              min = 1e3, 
                              max = 1e4-1),
                        digits = 0)
  
  # Generate a file name and output file path for the result text file
  outFileName <- paste0(timeString, '_', randomNumber, '.txt')
  outPath <- file.path(folder2use, outFileName)
  
  
  #### Create output file ####
  
  cat(isHead, file = outPath)
  
  #### Track progress ####
  
  # print(paste('i =', i))
  cat('Completed rep', i, '\n')
  
}


