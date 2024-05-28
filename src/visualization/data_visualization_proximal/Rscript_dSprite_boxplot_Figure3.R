library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)
library(extrafont)
loadfonts()

# Set the working directory to results/
setwd("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/results")
df = read.csv("aggregated_results_for_figures/dsprite_boxplot_data.csv")

df$sample_size = as.factor(df$sample_size)

# ordering by descending MSE in first panel
methods = c('pmmr', 'kpv', 'naivenet_awzy', 'naivenet_awy', 'cevae', 'dfpv','nmmr_u', 'nmmr_v')
df_subset = df[which(df$method %in% methods),]

df_subset$method = factor(df_subset$method, levels=methods)

# sort rows according to vector `methods`
df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]

# create a `linewidth` column to highlight our method's results
df_subset = mutate(df_subset, linewidth=ifelse((method=="cevae" & sample_size %in% c("1000")) |
                                                 (method=="pmmr" & sample_size %in% c("5000", "7500")), "tiny",
                                               ifelse((method=="pmmr" & sample_size=="1000") |
                                                        (method=="kpv" & sample_size %in% c("1000", "5000", "7500")) |
                                                        (method=="cevae" & sample_size %in% c("5000", "7500")), "small", 
                                                        "normal")))

# Map sample sizes to desired representation
n_strings <- c("1000" = "1,000", "5000" = "5,000", "7500" = "7,500")

commafy <- function(n) {
  n <- format(as.numeric(n), big.mark=",")
  return(n)
}

p <- ggplot(df_subset, aes(x=sample_size, y=oos_mse, fill=method, size=linewidth)) + 
  geom_boxplot(fatten=1, outlier.size=0.3, outlier.stroke=0.2) +
  ylab(TeX(r'(Causal MSE ($log_{10}$))')) +
  xlab("Data Size") +  
  scale_y_continuous(trans='log10', labels=commafy) + 
  scale_x_discrete(labels=commafy) +
  coord_cartesian(ylim = c(1, 100)) +
  scale_fill_manual(labels=c("PMMR", "KPV", "Naive net Y~AWZ", "Naive net Y~AW", "CEVAE", "DFPV","**NMMR U (ours)**", "**NMMR V (ours)**"),
                    values=c("aquamarine4", "deeppink2", "lemonchiffon2", "blue",
                             "goldenrod2", "steelblue2", "darkorchid3", "magenta2")) +
  scale_size_manual(values=c(0.3, 0.15, 0.05), guide="none") +
  theme_bw() + 
  theme(legend.title=element_blank(),
        legend.text = ggtext::element_markdown(size=10),
        panel.spacing = unit(0, "lines"),
        strip.text.x = element_blank(),
        panel.grid.major.x = element_blank(),
        text = element_text(family = "Times", size=10)) +
  facet_grid(. ~ sample_size, scales = "free", space = "free")

p

ggsave("dsprite_boxplot.png", p, path="~/Desktop", dpi=320, width = 6, height = 2.65, units = "in")
