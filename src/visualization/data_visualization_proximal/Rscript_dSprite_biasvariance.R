library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)

# Set the working directory to results/
setwd("~/")
df = read.csv("aggregated_results_for_figures/dsprite_biasvariance.csv")

# subset methods for the plot
methods = c('naivenet_awzy', 'cevae', 'kpv', 'pmmr', 'naivenet_ay', 'dfpv', 'nmmr_u', 'nmmr_v')

for (n in c(1000, 5000, 7500)) {
  
  df_subset = df[which(df$method %in% methods & df$sample_size==n),]
  
  df_subset$method = factor(df_subset$method, levels=methods)
  
  # sort rows according to vector `methods`
  df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]
  
  method_names <- c(naivenet_awzy = "Naive net Y~AWZ",
                    cevae = "CEVAE", 
                    kpv = "KPV", 
                    pmmr = "PMMR", 
                    naivenet_ay = "Naive net Y~A",
                    dfpv = "DFPV", 
                    nmmr_u = "**NMMR U (ours)**", 
                    nmmr_v = "**NMMR V (ours)**")
  
  p <- ggplot(df_subset) + 
    geom_histogram(aes(x=variance, color="var"), binwidth=1, fill="transparent", alpha=1, size=0.25) + 
    geom_histogram(aes(x=bias, color="bias"), binwidth=1, fill="transparent", alpha=0.5, size=0.25) +
    coord_cartesian(ylim = c(0, 600), xlim = c(-10, 80)) +
    scale_colour_manual(name="", values=c("var"="darkblue", "bias"="red"), labels=c("Variance", "Bias")) +
    xlab("Units of Bias or Variance") +
    ylab("Count") +
    ggtitle(sprintf("Bias-Variance trade-off on dSprite (n=%s)", n)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5),
          panel.spacing = unit(0, "lines"), 
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          text = element_text(family = "Times", size=20),
          strip.text = ggtext::element_markdown(size=16, margin=unit(c(7, 0, 4, 0), "pt")),
          axis.title.x = element_text(vjust=-1),
          plot.margin = margin(b=10, t=1, r=1, l=1)) +
    facet_wrap(~ method, ncol=4, labeller=labeller(method = method_names))
  
  p
  
  ggsave(sprintf("dsprite_biasvariance%i.png", n), p, path="~/Desktop", dpi=320, width = 11, height = 6, units = "in")
  
}

######### Boxplots of dSprite bias and variance

library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)
library(extrafont)
loadfonts()

# Set the working directory to results/
setwd("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/results/")

df = read.csv("aggregated_results_for_figures/dsprite_biasvariance.csv")

df$sample_size = as.factor(df$sample_size)

# ordering by descending MSE in first panel
methods = c('pmmr', 'kpv', 'naivenet_awzy', 'cevae', 'dfpv','nmmr_u', 'nmmr_v')
df_subset = df[which(df$method %in% methods),]

df_subset$method = factor(df_subset$method, levels=methods)

# sort rows according to vector `methods`
df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]

# create a `linewidth` column to highlight our method's results
# df_subset = mutate(df_subset, linewidth=ifelse((method=="cevae" & sample_size %in% c("1000")) |
#                                                  (method=="pmmr" & sample_size %in% c("5000", "7500")), "tiny",
#                                                ifelse((method=="pmmr" & sample_size=="1000") |
#                                                         (method=="kpv" & sample_size %in% c("1000", "5000", "7500")) |
#                                                         (method=="cevae" & sample_size %in% c("5000", "7500")), "small", 
#                                                       "normal")))

# Map sample sizes to desired representation
n_strings <- c("1000" = "1,000", "5000" = "5,000", "7500" = "7,500")

commafy <- function(n) {
  n <- format(as.numeric(n), big.mark=",")
  return(n)
}

p <- ggplot(df_subset, aes(x=sample_size, y=bias, fill=method)) + 
  geom_boxplot(fatten=1, outlier.size=0.3, outlier.stroke=0.2) +
  ylab("Bias") +
  xlab("Data Size") +  
  scale_y_continuous(labels=commafy) + 
  scale_x_discrete(labels=commafy) +
  coord_cartesian(ylim = c(-15, 15)) +
  scale_fill_manual(labels=c("PMMR", "KPV", "Naive net", "CEVAE", "DFPV","**NMMR U (ours)**", "**NMMR V (ours)**"),
                    values=c("aquamarine4", "deeppink2", "lemonchiffon2",
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

ggsave("dsprite_bias_boxplot.png", p, path="~/Desktop", dpi=320, width = 6, height = 2.65, units = "in")

p <- ggplot(df_subset, aes(x=sample_size, y=variance, fill=method)) + 
  geom_boxplot(fatten=1, outlier.size=0.3, outlier.stroke=0.2) +
  ylab("Variance") +
  xlab("Data Size") +  
  scale_y_continuous(labels=commafy) + 
  scale_x_discrete(labels=commafy) +
  coord_cartesian(ylim = c(0, 10)) +
  scale_fill_manual(labels=c("PMMR", "KPV", "Naive net", "CEVAE", "DFPV","**NMMR U (ours)**", "**NMMR V (ours)**"),
                    values=c("aquamarine4", "deeppink2", "lemonchiffon2",
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

ggsave("dsprite_variance_boxplot.png", p, path="~/Desktop", dpi=320, width = 6, height = 2.65, units = "in")

