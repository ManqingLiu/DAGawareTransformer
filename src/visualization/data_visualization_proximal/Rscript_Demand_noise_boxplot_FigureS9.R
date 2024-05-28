library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)
library(extrafont)
loadfonts()

# Set the working directory to results/
setwd("~/")
df = read.csv("aggregated_results_for_figures/demand_noise_boxplot_data.csv")

# ordering by descending MSE in first panel
methods = c('pmmr', 'kpv', 'naivenet_awy', 'cevae', 'twosls', 'dfpv', 'linear_reg_awy', 'nmmr_u', 'nmmr_v')
df_subset = df[which(df$method %in% methods),]

df_subset$method = factor(df_subset$method, levels=methods)

# sort rows according to vector `methods`
df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]

scyify <- function(n) {
  n <- format(as.numeric(n), scientific=TRUE)
  return(n)
}

p <- ggplot(df_subset, aes(y=oos_mse, fill=method)) + 
  geom_boxplot(size=0.2, outlier.size=0.3, outlier.stroke=0.2, position = position_dodge2(preserve = "total")) +
  ylab(TeX(r'(Causal MSE ($log_{10}$))')) +
  scale_y_continuous(trans='log10', labels=scyify, sec.axis = sec_axis(~ ., name=TeX(r'(W noise ($\sigma^2_W$))'), breaks=NULL, labels=NULL)) +
  scale_x_continuous(labels=NULL, sec.axis = sec_axis(~ ., name=TeX(r'(Z noise ($\sigma^2_Z$))'), breaks=NULL, labels=NULL)) +
  scale_fill_manual(labels=c("PMMR", "KPV", "Naive net", "CEVAE", "2SLS", "DFPV", "Least squares", "**NMMR U (ours)**", "**NMMR V (ours)**"),
                    values=c("aquamarine4", "deeppink2", "lemonchiffon2",
                             "goldenrod2", "red", "steelblue2",
                             "gray60", "darkorchid3", "magenta2")) +
  theme_bw() +
  theme(legend.title=element_blank(),
        legend.text=ggtext::element_markdown(size=10),
        panel.spacing = unit(0, "lines"),
        axis.ticks.x = element_blank(),
        text = element_text(family = "Times", size=10),
        axis.title = element_text(family = "Times", size=15),
        axis.text.y.left = element_text(family="Times", size=10),
        strip.text = element_text(family="Times", size=14)) +
  facet_grid(vars(W_noise), vars(Z_noise), scales = "free", space = "free")

p

ggsave("demand_noise_boxplot.png", p, path="~/Desktop", dpi=320, width = 11, height = 8, units = "in")
