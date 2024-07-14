task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

library(abind)
library(TMB)
library(data.table)
library(tidyverse)
library(ktools)

options("mc.cores" = parallel::detectCores()-2)
dir.create("processed", FALSE)

message(task_id)

set.seed(123)
# short hand
X = attributes

file <- list.files(pattern = "*.rds")[task_id]
# if (file.exists(paste0("processed", "/", file))) {
#     next
# }

post <- readRDS(file)

params <- c()
params$nsv <- file %>% gsub(pattern = ".*nsv([0-9]{1})_.*", replacement = "\\1") %>% as.numeric()
params$sample_size <- file %>% gsub(pattern = ".*sample_size([0-9]{4})_.*", replacement = "\\1") %>% as.numeric()
params$bias <- file %>% gsub(pattern = ".*bias(.*?)_.*", replacement = "\\1")
params$trend <- file %>% gsub(pattern = ".*trend(.*?)_.*", replacement = "\\1")
params$theK <- file %>% gsub(pattern = ".*theK([0-9]{1,3}).*", replacement = "\\1") %>% as.numeric()

params

## Pool population
birth_cohorts <- 1940:2005
wanted_cohort <- 1985:2005

sk_scale <- function(median = 16, q = 0.5, shape = 10, skew = 1.5) {
    (0.5^((-1) / skew) - 1)^((-1) / shape) / median
}

# AFS parameters and sampling
# Following skew log-logistic distribution, at the begining, year 1900
ref <- list(scale = sk_scale(17), shape = 10, skew = 1.5)
(min_scale <- sk_scale(15))
(max_scale = sk_scale(19))

if (params$trend == "none") {
    pdata <- tibble(yob = birth_cohorts, scale = ref$scale, skew = ref$skew, shape = ref$shape)
} else if (params$trend == "increase") {
    scalev <- seq(min_scale, max_scale, length.out = length(birth_cohorts))
    pdata <- tibble(yob = birth_cohorts, scale = scalev, skew = ref$skew, shape = ref$shape)
} else if (params$trend == "decrease") {
    scalev <- seq(max_scale, min_scale, length.out = length(birth_cohorts))
    pdata <- tibble(yob = birth_cohorts, scale = scalev, skew = ref$skew, shape = ref$shape)
}

pdata %<>% mutate(median = qskewlogis(.5, scale, shape, skew))

pdata %>%
    mutate(med = qskewlogis(.5, scale, shape, skew)) %>%
    filter(yob %in% range(wanted_cohort)) %>%
    pull(med) %>%
    diff()

real_diff_1970_2005 <- tibble(
    trend = char(none, increase, decrease),
    real_diff = c(0, 1.440758, -1.038961)
)

# Get some stats
# apply to one scenario with 1000 posterior samples
stats = function(x, agref = 1) {
    # skew and shape 
    realskew = x[, "log_skew"] %>% exp()
    realshape = x[, "log_shape"] %>% exp()
    # scale depending on age reference
    ageref = tibble(age = X(x)$age, id = X(x)$age_id) %>% mutate(agr = findInterval(age, seq(15, 50, 5)))
    agecoef = x[, ageref %>% filter(agr == agref) %>% pull(id)] %>% rowMeans()
    realscale = exp(sweep(x[, X(x)$yob_id], 1, x[, "intercept"], "+") + agecoef)
    # median differences
    getmed = function(y, x) {
        tibble(
            med = qskewlogis(.5, realscale[y, ], realshape[y], realskew[y]),
            yob = X(x)$yob
        ) %>%
            left_join(pdata, "yob") %>%
            filter(yob %in% range(wanted_cohort)) %>%
            mutate(diff = med - median) %>%
            {
                tibble(
                    first_diff = .$diff[1],
                    last_diff = .$diff[2],
                    trend_diff = .$med[2] - .$med[1],
                    real_diff = .$median[2] - .$median[1]
                )
            }
    }    

    st = lapply(1:nrow(x), getmed, x = x) %>% bind_rows()

    st %>%
        mutate(
            up = quantile(trend_diff, prob = .975),
            lo = quantile(trend_diff, prob = .025),
            inside95 = real_diff <= up & real_diff >= lo,
            up = quantile(trend_diff, prob = .75),
            lo = quantile(trend_diff, prob = .25),
            insideIQR = real_diff <= up & real_diff >= lo,
            direction = case_when(
                params$trend == "increase" ~ trend_diff > 0,
                params$trend == "decrease" ~ trend_diff < 0,
                params$trend == "none" ~ abs(trend_diff) < 6/12,
            ),
        ) %>% summarise(
            ave_first = mean(first_diff),
            ave_last = mean(last_diff),
            ave_trend = mean(trend_diff),
            cov_95 = mean(inside95),
            cov_iqr = mean(insideIQR),
            direction = mean(direction)
        )
}

# parallel::detectCores()
# apply to the number of survey sample of a scenario
getstats = function(aref) {
    parallel::mclapply(post, stats, agref = aref) %>%
        bind_rows(.id = "sample") %>%
        mutate(
            ageref = aref, nsv = params$nsv, size = params$sample_size,
            trend = params$trend, bias = params$bias, nsample = params$theK
        )
}
	
out1 = getstats(1)
out2 = getstats(2)
out3 = getstats(3)
out4 = getstats(4)

out = bind_rows(out1, out2, out3, out4)

saveRDS(out, paste0("processed", "/", file))