#!/usr/bin/Rscript
script_options = docopt::docopt(
    stringr::str_glue("Usage:
    create-distance-data.R [options] 

    Takes RCT data and creates distance matrix from PoTs to villages.


    Options:
        --input-path=<path>  Path to find results [default: {file.path('data', 'stan_analysis_data')}]
        --output-path=<path>  Path to find results [default: {file.path('optim', 'data')}]
        --output-name=<output-name>  Prepended to output file
        --num-extra-pots=<num-extra-pots>  How many extra PoTs to sample per village above experiment [default: 2]
        --county-subset=<county-subset>  Which county to subset OA to
        --distance-cutoff=<distance-cutoff>  How close PoTs have to be to sample extra. [default: 3500]
    "),
    args = if (interactive()) "
                            --output-name=SIAYA-experiment.rds
                            --num-extra-pots=4
                            --county-subset=SIAYA
                              " 
           else commandArgs(trailingOnly = TRUE)
)

set.seed(19484)

library(tidyverse)
library(sf)
library(econometr)
source(file.path("rct-design-fieldwork", "takeup_rct_assign_clusters.R"))
source(file.path("analysis_util.R"))
source(file.path( "dist_structural_util.R"))


script_options$distance_cutoff = as.numeric(script_options$distance_cutoff)

source(file.path("multilvlr", "multilvlr_util.R"))

full_experiment = ifelse(is.null(script_options$county_subset) | script_options$county_subset == "full", TRUE, FALSE)


wgs.84 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
kenya.proj4 <- "+proj=utm +zone=36 +south +ellps=clrk80 +units=m +no_defs"

rct.schools.data <- read_rds(file.path("data", "takeup_rct_schools.rds"))
rct.cluster.selection <- read_rds(file.path("data", "rct_cluster_selection_2.0.rds"))
cluster.strat.data <- read_rds(file.path("data", "takeup_processed_cluster_strat.rds"))

load(file.path("data", "takeup_village_pot_dist.RData"))

load(file.path("data", "analysis.RData"))

standardize <- as_mapper(~ (.) / sd(.))
unstandardize <- function(standardized, original) standardized * sd(original)

nosms_data <- analysis.data %>% 
  filter(sms.treatment.2 == "sms.control") %>% 
  left_join(village.centers %>% select(cluster.id, cluster.dist.to.pot = dist.to.pot),
            by = "cluster.id") %>% 
  mutate(standard_cluster.dist.to.pot = standardize(cluster.dist.to.pot)) %>% 
  group_by(cluster.id) %>% 
  mutate(cluster_id = cur_group_id()) %>% 
  ungroup()

monitored_nosms_data <- analysis.data %>% 
  filter(mon_status == "monitored", sms.treatment.2 == "sms.control") %>% 
  left_join(village.centers %>% select(cluster.id, cluster.dist.to.pot = dist.to.pot),
            by = "cluster.id") %>% 
  mutate(standard_cluster.dist.to.pot = standardize(cluster.dist.to.pot)) %>% 
  group_by(cluster.id) %>% 
  mutate(cluster_id = cur_group_id()) %>% 
  ungroup()

analysis_data <- monitored_nosms_data

sd_of_dist <- analysis.data %>% 
  filter(mon_status == "monitored", sms.treatment.2 == "sms.control") %>% 
  left_join(village.centers %>% select(cluster.id, cluster.dist.to.pot = dist.to.pot),
            by = "cluster.id") %>% 
    summarise(ed_sd = sd(cluster.dist.to.pot)) %>%
    pull()


## Now extract distance data
## Grabbing Data
cluster_ids_actually_used = analysis_data %>%
  select(cluster.id) %>%
  pull() %>%
  unique()

rct_cluster_df = st_as_sf(rct.cluster.selection) 

rct_pot_cluster_ids = rct_cluster_df %>%
  filter(selected == TRUE) %>%
  filter(cluster.id %in% cluster_ids_actually_used) %>%
  pull(cluster.id)

n_orig_pots = length(rct_pot_cluster_ids)

rct_school_df = st_as_sf(rct.schools.data)

# If subsettting to a county, use PoTs on both sides of market 
if (!full_experiment) {

    subset_rct_school_df = rct_school_df %>%
        mutate(County = str_to_lower(County)) %>%
        filter(County == script_options$county_subset)

    # pot and village same here
    village_df = subset_rct_school_df %>%
        mutate(id = 1:n())  
    pot_df = subset_rct_school_df %>%
        mutate(id = 1:n())  

    distance_matrix = st_distance(
        subset_rct_school_df
    )

}

if (full_experiment) {


    unselected_pots = setdiff(
    rct_school_df$cluster.id,
    rct_pot_cluster_ids)


    # Adding PoTs not used in experiment but we know exist
    # due to computational constraints we choose a random sample within 2.5km of 
    # a village
    add_pot_df = rct_school_df %>%
        filter(cluster.id %in% unselected_pots) 


    rct_pot_df = rct_school_df %>%
        filter(cluster.id %in% rct_pot_cluster_ids)



    village_df = village.centers %>%
    st_as_sf(coords = c("lon", "lat"), crs = wgs.84) %>%
    mutate(id = 1:n()) 
    # filter(!(id %in% c(40, 45, 116))) %>% # Basically impossible to serve these
    # mutate(id = 1:n()) 


    add_distance_matrix = st_distance(
        village_df, 
        add_pot_df
    )

    long_add_distance_mat = add_distance_matrix %>%
        as_tibble() %>%
        mutate(index_i = 1:n()) %>%
        gather(variable, dist, -index_i) %>%
        mutate(index_j = str_extract(variable, "\\d+") %>% as.numeric())  %>%
        select(index_i, index_j, dist) %>%
        mutate(
            dist = as.numeric(dist),
            dist_km = as.numeric(dist/1000))


    extra_pot_ids = long_add_distance_mat %>%
        mutate(close = dist <= script_options$distance_cutoff) %>%
        group_by(index_i) %>%
        filter(close == TRUE) %>%
        sample_n(script_options$num_extra_pots, replace = TRUE) %>%
        pull(index_j) %>%
        unique()


    pot_df = rct_school_df %>%
        filter(cluster.id %in% c(rct_pot_cluster_ids, extra_pot_ids)) %>%
        mutate(id = 1:n())




    distance_matrix = st_distance(
    village_df,
    pot_df
    )


}


long_distance_mat = distance_matrix %>%
    as_tibble() %>%
    mutate(index_i = 1:n()) %>%
    gather(variable, dist, -index_i) %>%
    mutate(index_j = str_extract(variable, "\\d+") %>% as.numeric())  %>%
    select(index_i, index_j, dist) %>%
    mutate(
        dist = as.numeric(dist),
        dist_km = as.numeric(dist/1000))


brms_long_distance_mat = long_distance_mat %>%
    left_join(
        village_df %>% as_tibble() %>% select(cluster.id, id),
        by = c("index_i" = "id")
    )  %>%
    mutate(
        sd_dist = dist / sd_of_dist, 
        assigned.treatment = "bracelet"
    ) 

pot_df = pot_df %>%
    mutate(
        lon = sf::st_coordinates(.)[, 1],
        lat = sf::st_coordinates(.)[, 2]) %>%
    as_tibble() 

village_df = village_df %>%
    mutate(
        lon = sf::st_coordinates(.)[, 1],
        lat = sf::st_coordinates(.)[, 2]) %>%
    as_tibble() 

distance_data = lst(
    long_distance_mat,
    brms_long_distance_mat,
    pot_df, 
    village_df,
    sd_of_dist
)


saveRDS(
    distance_data,
    file = file.path(
        script_options$output_path,
        script_options$output_name
    )
)
