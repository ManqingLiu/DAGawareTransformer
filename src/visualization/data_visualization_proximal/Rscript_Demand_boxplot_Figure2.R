library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)
library(extrafont)
loadfonts()

# Set the working directory to results/
setwd("~/")
df = read.csv("aggregated_results_for_figures/demand_boxplot_data.csv")

df$sample_size = as.factor(df$sample_size)

# ordering by descending MSE in first panel
methods = c('pmmr', 'kpv', 'naivenet_awy', 'cevae', 'twosls', 'dfpv', 'linear_reg_awy', 'nmmr_u', 'nmmr_v')
df_subset = df[which(df$method %in% methods),]

df_subset$method = factor(df_subset$method, levels=methods)

# sort rows according to vector `methods`
df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]

# create a `linewidth` column to highlight our method's results
df_subset = mutate(df_subset, linewidth=ifelse((method=="twosls" & sample_size=="50000"), "tiny", 
                                               ifelse((method=="twosls" & sample_size %in% c("5000", "10000")) | 
                                                        (method=="kpv" & sample_size=="10000") |
                                                        (method=="pmmr" & sample_size %in% c("1000", "10000")) |
                                                        (method=="linear_reg_awy"), "small", "normal")))

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
  coord_cartesian(ylim = c(5, 4000)) +
  scale_fill_manual(labels=c("PMMR", "KPV", "Naive net", "CEVAE", "2SLS", "DFPV", "LS",
   "**NMMR U (ours)**", "**NMMR V (ours)**"),
                    values=c("aquamarine4", "deeppink2", "lemonchiffon2",
                             "goldenrod2", "red", "steelblue2", "yellow", "darkorchid3", "magenta2")) +
  scale_size_manual(values=c(0.3, 0.15, 0.05), guide="none") +
  theme_bw() + 
  theme(legend.title=element_blank(),
        legend.text = ggtext::element_markdown(size=8),
        panel.spacing = unit(0, "lines"),
        strip.text.x = element_blank(),
        panel.grid.major.x = element_blank(),
        text = element_text(family = "Times", size=8)) +
  facet_grid(. ~ sample_size, scales = "free", space = "free")

p

ggsave("demand_boxplot.png", p, path="~/Desktop", dpi=320, width = 6, height = 2.65, units = "in")
