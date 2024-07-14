library(ggplot2)
library(dplyr)

### set the working directory
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else{
  # If running as a script, finding the file is harder
  # https://stackoverflow.com/a/55322344/10930590
  this_file <- commandArgs() %>%
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)

  setwd(dirname(this_file))
  setwd("../data")
}

LVs <- read.table("../output/reformated_NAc_PFC_VTA_Lvs.txt", header = TRUE, colClasses = c("factor", "factor", "factor", "factor", "numeric"))
df <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(df) <- c("LV_ID", "day", "region", "pvalue")

### perform one way anova
for (i in 1: length(levels(LVs$LV_ID))) {
  lv <- paste("LV", as.character(i), sep="")
  for (d in levels(LVs$day)) {
    for (r in levels(LVs$region)) {
      temp_lv <- subset(LVs, LVs$LV_ID == lv & LVs$day == d & LVs$region == r)

      ## one way anova
      one.way <- aov(lv_value ~ treatment, data = temp_lv)
      #summary(one.way)
      pvalue <- summary(one.way)[[1]][[1,"Pr(>F)"]]
      df[nrow(df)+1, ] <- c(lv, d, r, pvalue)
    }
  }
}

### BH correction and write the results
df$adjusted_p <- p.adjust(df$pvalue, "BH")
write.csv(df, "../output/LVs_pvalues.txt", row.names = FALSE)
sig_df <- subset(df, df$adjusted_p < 0.05)

### an example to show LV among treatments: LV135
LV135_day1_NAc <- subset(LVs, LVs$LV_ID == "LV135" & LVs$day == "day1" & LVs$region == "NAc")
LV135_day1_NAc$treatment <- factor(LV135_day1_NAc$treatment, levels = c("saline", "food", "cocaine"))
myplot <- ggplot(LV135_day1_NAc, aes(x=treatment, y=lv_value, fill=treatment)) + 
  geom_boxplot() +
  ggtitle("")+
  theme_classic()+
  theme(plot.title = element_text(size = 20, hjust = 0.5), line = element_blank(), plot.background = element_blank(), panel.grid.major = element_blank()) + #set the background
  theme(panel.border = element_blank()) +   #set the border
  theme(axis.text.x = element_text()) + 
  theme(axis.title = element_text(size = 20), axis.text.x = element_text(size=20, angle=0, hjust = 0.5), axis.text.y = element_text(size=20)) + #set the x and y lab
  ylab("LV135") + xlab("") +     #set the name of x-axis and y-axis
  theme( legend.title = element_blank(),  legend.position = "none", legend.text = element_text(size = 12), legend.key.width = unit(0.5, "cm")) + #set legend
  theme(axis.line.x = element_line(color="black", size = .6),  
        axis.line.y = element_line(color="black", size = .6),
        axis.ticks.x = element_line(size = 0.5),
        axis.ticks.y = element_line(size = 0.5),
        axis.ticks.length = unit(0.2, "cm")) 
outfile <- paste("../output/", lv, d, r, ".pdf", sep="_")
pdf(file = outfile,   # The directory you want to save the file in
    width = 4, # The width of the plot in inches
    height = 4)
print(myplot)
dev.off()
### End of LV135
