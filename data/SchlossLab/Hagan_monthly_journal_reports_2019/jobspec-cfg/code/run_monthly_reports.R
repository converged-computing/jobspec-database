# An R-Script to gather data from xml files, compile into dataset, merge with impact data
# and generate Rmd reports

# load packages
library(knitr)
library(kableExtra)
library(markdown)
library(rmarkdown)
library(tidyverse)
library(lubridate)

this_ym <- today() %>% str_extract(., "\\d{4}-\\d{2}") #the month & year reports generated

#parse data from xml
source("code/report_parse.R")

#load, clean, & join manuscript data w. impact data
source("code/merge_clean_report_data.R")

# load dataset & functions for report generation
source("code/load_report_data_functions.R")

#get preferred plot settings for ggplot
source("code/plot_options.R")


# create ASM report (this is outside the loop so it is only called once)
rmarkdown::render('code/monthly_report.Rmd',  # template file
                  output_file =  paste0("ASM_journals_report_", this_ym, ".html"), 
                  output_dir = paste0("reports/", today() %>% str_replace_all(., "-", "_")))

# for each journal in the data create a report
# these reports are saved in output_dir with the name specified by output_file
journals_list <- data %>% pull(journal) %>% unique() #list of journals on which to run reports

for (each_journal in journals_list){ #loop through the reports for each journal
  rmarkdown::render('code/each_journal_report.Rmd',  # template file for individual reports
                    output_file =  paste0(each_journal, "_report_", this_ym, ".html"), 
                    output_dir = paste0("reports/", today() %>% str_replace_all(., "-", "_")))
}
