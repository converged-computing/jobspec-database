#!/usr/bin/Rscript
script_options <- docopt::docopt(
    stringr::str_glue("Usage:
        postprocess_allocation.R  [options] [--min-cost | --max-takeup]

        Options:
          --min-cost  Flag to minimise programme cost for a given takeup level.
          --max-takeup  Flag to maximise vaccine takeup for a given cost level.
          --optim-input-path=<optim-input-path>  Path where input data is stored.
          --optim-input-a-filename=<optim-input-a-filename>  Optim input a filename.
          --optim-input-b-filename=<optim-input-b-filename>  Optim input b filename.
          --comp-output-basename=<comp-output-basename>  Comparison output basename
          --output-path=<output-path>  Path where output should be saved.
          --output-basename=<output-basename>  Output basename.
          --map-plot  Plot map without adding lines for closest villages and active PoTs 
          --posterior-median  If posterior median over takeup demand is used.
          --cutoff-type=<cutoff-type>  Which cutoff type to use if using full posterior draws [default: no-cutoff]
          --constraint-type=<constraint-type>  Aggregate or individual village welfare
          --welfare-function=<welfare-function>  Log or identity utility
          --data-input-path=<data-input-path>  Where village and PoT data is stored [default: {file.path('optim', 'data')}]
          --data-input-name=<data-input-name>  Filename of village and PoT data [default: full-experiment-4-extra-pots.rds]
          --pdf-output-path=<pdf-output-path>  Output path for PDF plots 
"),
  args = if (interactive()) "
                            --constraint-type=agg \
                            --welfare-function=log \
                            --min-cost \
                            --optim-input-path=optim/data/STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP/agg-log-full-many-pots \
                            --optim-input-a-filename=target-rep-suppress-rep-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP-post-draws-optimal-allocation.rds \
                            --output-path=optim/plots/agg-log-full-many-pots \
                            --output-basename=target-rep-agg-log-suppress-rep-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP-median \
                            --cutoff-type=cutoff
                            --data-input-name=full-many-pots-experiment.rds \
                            --pdf-output-path=presentations/takeup-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP-fig/
                             " else commandArgs(trailingOnly = TRUE)
) 


library(tidyverse)
library(sf)
library(data.table)
library(latex2exp)



source("optim/optim-functions.R")
# script_options = script_options %>%
#   modify_at(numeric_options, as.numeric)

optim_type = if_else(script_options$min_cost, "min_cost", "max_takeup")
stat_type = if_else(script_options$posterior_median, "median", "post-draws")

swf = eval(parse(text = script_options$welfare_function))

## Input Paths
optim_input_a_filepath = file.path(script_options$optim_input_path, 
                                script_options$optim_input_a_filename)
optim_input_b_filepath = file.path(script_options$optim_input_path,
                                script_options$optim_input_b_filename)
## Input Data
optimal_df = read_rds(optim_input_a_filepath)

dist_data = read_rds(
  file.path(
    script_options$data_input_path,
    script_options$data_input_name
  )
)

village_data = dist_data$village_df
pot_data = dist_data$pot_df

n = nrow(village_data)
m = nrow(pot_data)

## Output Filepaths
map_plot_path = file.path(
    script_options$output_path,
    paste0(
        script_options$output_basename,
        "-map-plot.png"
    )
)


optimal_allocation_demand_comp_plot_path = file.path(
    script_options$output_path,
    str_glue(
        "{script_options$output_basename}-optimal-allocation-demand-comp-plot.png"
    )
)



wgs.84 = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
## Loading Data
data = list(
    n = n,
    m = m,
    pot_locations = st_as_sf(pot_data, coords = c("lon", "lat"), crs = wgs.84),
    village_locations = st_as_sf(village_data, coords = c("lon", "lat"), crs = wgs.84)
) 



library(scales)
#extract hex color codes for a plot with three elements in ggplot2 
hex <- ggthemes::canva_pal("Primary colors with a vibrant twist")(2)

if (script_options$map_plot & is.null(script_options$optim_input_b_filename)){
    data$village_locations %>%
        ggplot() +
        geom_sf() +
        geom_sf(
            data = data$pot_locations,
            colour = hex[1],
            shape = 17,
            # size = 2, 
            alpha = 0.7) +
        theme_bw() +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
        labs(
            title = "PoT Allocation Problem",
            subtitle = "Black dots indicate villages. Triangles indicate potential clinic locations."
        )  +
        labs(x = "", y = "") + 
        # theme(
        # axis.title.x=element_blank(),
        # axis.text.x=element_blank(),
        # axis.ticks.x=element_blank(),
        # axis.title.y=element_blank(),
        # axis.text.y=element_blank(),
        # axis.ticks.y=element_blank()) +
        NULL

    ggsave(map_plot_path, width = 8, height = 6, dpi = 500)
}


if (stat_type == "median" & is.null(script_options$optim_input_b_filename)) {



    plot_optimal_allocation = function(village_data,
                                       pot_data,
                                       optimal_data){

        oa_stats = gen_oa_stats(
            village_data,
            pot_data,
            optimal_data, 
            welfare_function = swf
        )

        assigned_pots = oa_stats$assigned_pot
        n_pots_used = oa_stats$n_pots_used
        takeup_hit = oa_stats$takeup_hit
        util_hit = oa_stats$util_hit
        util_target = oa_stats$util_target
        overshoot = oa_stats$overshoot
        mean_dist = oa_stats$mean_dist

        pot_data$assigned_pot = pot_data$id %in% assigned_pots

    title_str = str_glue(
        "Takeup: {takeup_hit}%"
    )

    subtitle = str_glue(
        "Average Distance: {round(mean_dist/1000, 2)}km, Policymaker's Utility: {round(util_hit/144, 2)}"
    )

        if (script_options$constraint_type == "agg") {
            sub_str = str_glue("Assigned PoTs: {n_pots_used}")
        }

        if (script_options$constraint_type == "indiv") {
            sub_str = str_glue("Assigned PoTs: {n_pots_used}")
        }

        plot_data = bind_rows(
            village_data %>% mutate(type = "village") %>% select(-cluster.id), 
            pot_data %>% mutate(type = paste0("pot_", assigned_pot)) %>% select(-cluster.id)
        )


        optimal_pot_plot = plot_data %>%
            ggplot() +
            geom_sf(aes(
                color = type, 
                shape = type, 
                size = type, 
                alpha = type
            )) +
            theme_bw() +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_segment(
                data = optimal_data, 
                aes(x = village_lon, 
                    y = village_lat, 
                    xend = pot_lon, 
                    yend = pot_lat
                    ))  +
            annotate(
                "text",
                x = 34.8, 
                y = 0.75,
                label = sub_str
            ) +
            labs(x = "", y = "", title = title_str, subtitle = subtitle) +
            scale_size_manual(
                values = c("pot_TRUE" = 1.5, "pot_FALSE" = 1 , "village" = 1),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_alpha_manual(
                values = c("pot_TRUE" = 1, "pot_FALSE" = 0.3, "village" = 0.8),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_color_manual(
                values = c("pot_TRUE" = hex[2], "pot_FALSE" = hex[1], "village" = "black"),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_shape_manual(
                values = c("pot_TRUE" = 17, "pot_FALSE" = 17, "village" = 16),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            theme(legend.title = element_blank()) +
            # theme(legend.position = "bottom") +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            guides(alpha = guide_legend(override.aes = list(alpha = 1)))
        return(optimal_pot_plot)
    }


    allocation_plots = map(
        optimal_df$model_output,
        ~ plot_optimal_allocation(
            village_data = data$village_locations,
            pot_data = data$pot_locations,
            optimal_data = .x
        )
    )


    allocation_stats = map(
        optimal_df$model_output,
        ~gen_oa_stats(
            village_data = data$village_locations,
            pot_data = data$pot_locations,
            optimal_data = .x, 
            welfare_function = swf
        )
    )

    if (!is.null(script_options$pdf_output_path)) {
        map(
            allocation_plots,
            ~ .x + labs(caption = NULL)
        ) %>%
        iwalk(
            ., 
            ~ggsave(
                plot = .x, 
                filename = file.path(
                    script_options$pdf_output_path, 
                    str_glue(
                        "{script_options$output_basename}-optimal-allocation-plot.pdf"
                    )
                ), 
                width = 8, 
                height = 6
            )
        )
    }


    imap(
        allocation_stats,
        ~ .x %>%
            enframe() %>%
            spread(name, value) %>%
            unnest(-assigned_pots) %>%
            saveRDS(
                file.path(
                    script_options$output_path,
                    str_glue(
                        "{script_options$output_basename}-optimal-allocation-stats.rds"
                    )
                )
            )
    )

    iwalk(
        allocation_plots,
        ~ggsave(
            plot = .x,
            filename = file.path(
                script_options$output_path,
                str_glue(
                    "{script_options$output_basename}-optimal-allocation-plot.png"
                )
            ),
        width = 10,
        height = 10,
        dpi = 500
        )
    )
}

# if input b passed, create comp plot
if (!is.null(script_options$optim_input_b_filename)) {
    print(optim_input_b_filepath)
    optimal_b_df = read_rds(optim_input_b_filepath)
    comp_optimal_df = rbindlist(
        list(
            optimal_df, 
            optimal_b_df
        )
    )
    # if any suppress rep, the first input is assumed to have rep suppressed
    suppress_rep = c(
        script_options$optim_input_a_filename,
        script_options$optim_input_b_filename) %>%
    str_detect(., "suppress") %>%
    any()
    if (suppress_rep == TRUE) {
        comp_optimal_df$suppress_rep = c(TRUE, FALSE)
    } else {
        comp_optimal_df$suppress_rep = c(FALSE, FALSE)
    }


    comp_optimal_df[, n_pots_used := lapply(model_output, function(x){length(unique(x$j))})]

    comp_optimal_df[, demand_data := NULL]

    comp_optimal_df[, pot_data := list(data$pot_locations)]
    comp_optimal_df[suppress_rep == TRUE, visibility_z := "No Visibility"]
    plot_comp_function = function(meta_df){
        meta_df[, pot_data := map2(
            model_output, 
            pot_data, 
            ~mutate(
                .y, 
                assigned_pot = id %in% unique(.x$j)
            )
            )]

        meta_df[, mu_z := factor(visibility_z, levels = c("No Visibility", "control", "ink", "calendar", "bracelet"))]
        meta_df[, B_z := factor(private_benefit_z, levels = c("control", "ink", "calendar", "bracelet"))]
        
        pot_lhs_data = meta_df[1, pot_data][[1]]
        pot_rhs_data = meta_df[2, pot_data][[1]]


        pot_comp_df = bind_rows(
            pot_lhs_data %>% 
                mutate(
                    B_z = meta_df[1, private_benefit_z], 
                    mu_z = meta_df[1, visibility_z]
                ), 
            pot_rhs_data %>% 
                mutate(
                    B_z = meta_df[2, private_benefit_z], 
                    mu_z = meta_df[2, visibility_z] 
                )
        ) %>%
        mutate(type = paste0("pot_", assigned_pot))

        village_comp_df = bind_rows(
            village_data %>% mutate(type = "village") %>% select(-cluster.id) %>%
                mutate(
                    B_z = meta_df[1, private_benefit_z], 
                    mu_z = meta_df[1, visibility_z] 
                ),
            village_data %>% mutate(type = "village") %>% select(-cluster.id) %>%
                mutate(
                    B_z = meta_df[2, private_benefit_z], 
                    mu_z = meta_df[2, visibility_z] 
                )
        ) 



        optimal_comp_df = meta_df %>%
            select(private_benefit_z, visibility_z, model_output) %>%
            unnest(model_output)



        optimal_comp_df = optimal_comp_df %>%
            mutate(B_z = private_benefit_z, mu_z = visibility_z) %>%
            mutate(
                mu_z = factor(mu_z, levels = c("No Visibility", "control", "ink", "calendar", "bracelet")),
                cf_type = str_glue("B({B_z}, d),   mu({mu_z}, d)"), 
            ) 


        vil_pot_comp_df = bind_rows(
            pot_comp_df,
            village_comp_df
        ) %>%
        mutate(
            mu_z = factor(mu_z, levels = c("No Visibility", "control", "ink", "calendar", "bracelet")),
                cf_type = str_glue("B({B_z}, d),   mu({mu_z}, d)"), 
        ) 

        meta_df = meta_df %>%
            as_tibble() %>%
            mutate(
                cf_type = str_glue("B({B_z}, d),   mu({mu_z}, d)"), 
            ) 
        ed_label = function(string){paste0("Visibility: ", str_to_title(string))}
        vil_pot_comp_df %>%
            ggplot() +
            geom_sf(aes(
                color = type, 
                shape = type, 
                size = type, 
                alpha = type
            )) +
            geom_segment(
                data = optimal_comp_df, 
                aes(x = village_lon, 
                    y = village_lat, 
                    xend = pot_lon, 
                    yend = pot_lat
                    ))  +
            theme_bw()  +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            facet_wrap(~mu_z, labeller = as_labeller(ed_label)) +
            geom_text(
                 data = meta_df,
                 x = 34.8, 
                 y = 0.75,
                aes(label = paste0("Assigned PoTs: ", n_pots_used))
            ) +
            # annotate(
            #     "text",
            #     x = 34.9, 
            #     y = 0.75,
            #     label = sub_str
            # ) +
            labs(x = "", y = "") +
            scale_size_manual(
                values = c("pot_TRUE" = 1.5, "pot_FALSE" = 1 , "village" = 1),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_alpha_manual(
                values = c("pot_TRUE" = 1, "pot_FALSE" = 0.3, "village" = 0.8),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_color_manual(
                values = c("pot_TRUE" = hex[2], "pot_FALSE" = hex[1], "village" = "black"),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            scale_shape_manual(
                values = c("pot_TRUE" = 17, "pot_FALSE" = 17, "village" = 16),
                breaks = c("pot_TRUE", "pot_FALSE", "village"),
                labels = c("PoT Used", "PoT Unused", "Village")
            ) +
            theme(legend.title = element_blank()) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            theme(legend.position = "bottom") +
            guides(alpha = guide_legend(override.aes = list(alpha = 1)))

    }


    plot_comp = plot_comp_function(
            comp_optimal_df
    )


    ggsave(
        plot = plot_comp,
        filename = file.path(
            script_options$pdf_output_path, 
            str_glue(
                "{script_options$comp_output_basename}-compare-optimal-allocation-plot.pdf"
            )
        ), 
        width = 8, 
        height = 6
    )
}


if (stat_type == "post-draws") {

    long_optimal_df = optimal_df %>%
        select(
            draw,
            model,
            private_benefit_z,
            visibility_z, 
            model_output
            )  %>%
        unnest(model_output)

    cols_we_want = c(
        "draw", 
        "model", 
        "private_benefit_z",
        "visibility_z",
        "i", 
        "j", 
        "demand", 
        "dist", 
        "village_lon", 
        "village_lat",
        "pot_lon", 
        "pot_lat", 
        "target_optim"
    )


    long_optimal_df = long_optimal_df %>%
        select(any_of(cols_we_want))

    long_optimal_df %>%
        saveRDS(
            file.path(
                script_options$optim_input_path, 
                str_replace(script_options$optim_input_a_filename, "\\.rds", "-subset-long-data.rds")
            )
        )
}
