args = commandArgs(trailingOnly = T)
name = args[1]

setwd(paste0('/fs/ess/PAS0854/Active_projects/', name))

master = read.csv('Master.csv')
tcg = master[grepl('TCGA', master$ID), ]
ori = master[!(rownames(master) %in% rownames(tcg)), ]

library(ggplot2)
library(gridExtra)
library(stringr)
library(dplyr)

pr = as.data.frame(cbind(tcg$ID, tcg$TissueOfOrigin))
pr = pr[!(is.na(pr$V2)), ]
pr$V2 = gsub(', NOS', "", pr$V2)

pr = pr %>%
  count(V2, sort = TRUE) %>%
  left_join(pr, by = 'V2')
pr = pr[pr$V2 %in% unique(pr$V2)[1:15], ]
pr = pr[, -3]
pr = pr[!duplicated(pr$V2), ]

tots = c()
for (x in pr$V2){
  pri = cases() %>%
    GenomicDataCommons::filter(primary_site == x) %>%
    GenomicDataCommons::count()
  tots = c(tots, pri)
}

pr$n = unlist(lapply(1:nrow(pr), function(x){
  return((as.numeric(as.character(pr$n[x]))/
            tots[x])*100)
}))

pr = pr[order(-pr$n), ]

ttg = ggplot(pr, aes(x=reorder(V2, -n), y = n)) + 
  geom_bar(stat = 'identity', aes(fill = V2)) + 
  xlab('Primary Site') +
  ylab('Percent Prevalence') +
  labs(fill = 'Primary Site') +
  ggtitle('TCGA Primary Site') + 
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10), 
                   expand =c(.01,.2)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
        legend.position = 'none')

pr = as.data.frame(cbind(ori$ID, ori$TissueOfOrigin))
pr = pr[!(pr$V2 == ""), ]
pr$V2 = gsub(', NOS', "", pr$V2)

pr = pr %>%
  count(V2, sort = TRUE) %>%
  left_join(pr, by = 'V2')
pr = pr[pr$V2 %in% unique(pr$V2)[1:15], ]
pr = pr[, -3]
pr = pr[!duplicated(pr$V2), ]

clin = read.csv('/fs/ess/PAS0854/Raven/fusionsClin.csv')
tot = clin$SpecimenSiteOfOrigin
tot = tot[!(tot == "")]
tot = gsub(', NOS', "", tot)
tot = table(tot)
tot = tot[names(tot) %in% pr$V2]
tot = tot[match(pr$V2, names(tot))]

pr$n = unlist(lapply(1:nrow(pr), function(x){
  return((as.numeric(as.character(pr$n[x]))/
            tots[x])*100)
}))

pr = pr[order(-pr$n), ]

otg = ggplot(pr, aes(x=reorder(V2, -n), y = n)) + 
  geom_bar(stat = 'identity', aes(fill = V2)) + 
  xlab('Primary Site') +
  ylab('Percent Prevalence') +
  labs(fill = 'Primary Site') +
  ggtitle('Orien Primary Site') +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10), 
                   expand =c(.01,.2)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
        legend.position = 'none')

pr = as.data.frame(cbind(tcg$ID, tcg$Histology))
pr = pr[!(is.na(pr$V2)), ]
pr$V2 = gsub(', NOS', "", pr$V2)

pr = pr %>%
  count(V2, sort = TRUE) %>%
  left_join(pr, by = 'V2')
pr = pr[pr$V2 %in% unique(pr$V2)[1:15], ]

thg = ggplot(pr, aes(x = reorder(V2,V2,function(x)-length(x)), fill = V2)) + 
  geom_histogram(stat = 'count') + 
  xlab('Histology') +
  ylab('Count') +
  labs('Histology') +
  ggtitle('TCGA Histology') +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10), 
                   expand =c(.01,.2)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), 
        legend.position = 'none')

pr = as.data.frame(cbind(ori$ID, ori$Histology))
pr = pr[!(pr$V2 == ""), ]
pr$V2 = unlist(lapply(pr$V2, function(x) substr(x, 7, nchar(x))))
pr$V2 = gsub(', NOS', "", pr$V2)
pr$V2 = gsub('of no special type', "", pr$V2)

pr = pr %>%
  count(V2, sort = TRUE) %>%
  left_join(pr, by = 'V2')
pr = pr[pr$V2 %in% unique(pr$V2)[1:15], ]

ohg = ggplot(pr, aes(x = reorder(V2,V2,function(x)-length(x)), fill = V2)) + 
  geom_histogram(stat = 'count') + 
  xlab('Histology') +
  ylab('Count') +
  labs('Histology') +
  ggtitle('Orien Histology') +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10), expand =c(.05,.2)) +
  theme(axis.text.x = element_text(size = 9, angle = 90, vjust = 0.5, hjust=1), 
        legend.position = 'none')

pdf('./CancerTypes.pdf', height = 10, width = 20)

grid.arrange(ttg, otg, thg, ohg, nrow = 2, ncol = 2)

dev.off()

