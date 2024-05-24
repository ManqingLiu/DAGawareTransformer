library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)

# Set the working directory to results/
setwd("~/")

df = read.csv("aggregated_results_for_figures/demand_biasvariance.csv")

# subset methods for the plot
methods = c('linear_reg_awy', 'cevae', 'kpv', 'pmmr', 'twosls', 
            'linear_reg_aw2y', 'naivenet_awzy', 'dfpv', 'nmmr_u', 'nmmr_v')

for (n in c(1000, 5000, 10000, 50000)) {
  
  df_subset = df[which(df$method %in% methods & df$sample_size==n),]
  
  df_subset$method = factor(df_subset$method, levels=methods)
  
  # sort rows according to vector `methods`
  df_subset = df_subset[order(unlist(sapply(df_subset$method, function(x) which(methods == x)))),]
  
  method_names <- c(kpv = "KPV", 
                    pmmr = "PMMR", 
                    cevae = "CEVAE", 
                    dfpv = "DFPV", 
                    nmmr_u = "**NMMR U (ours)**", 
                    nmmr_v = "**NMMR V (ours)**", 
                    twosls = "2SLS", 
                    naivenet_awzy = "Naive net", 
                    linear_reg_awy = "LS AW", 
                    linear_reg_aw2y = "LS-QF AW2")
  
  p <- ggplot(df_subset) + 
    geom_line(aes(x=A, y=variance, color="var"), linetype="solid", alpha=1) + 
    geom_line(aes(x=A, y=bias, color="bias"), linetype="solid", alpha=1) +
    # geom_smooth(method='loess', formula=y ~ x, aes(x=A, y=true_EY_doA), color='black', size=0.65) +
    coord_cartesian(ylim = c(-50, 100), xlim = c(8, 32)) +
    scale_colour_manual(name="", values=c("var"="darkblue", "bias"="red"), labels=c("Variance", "Bias")) +
    xlab("Ticket price (A)") +
    ylab(TeX('Units of Bias or Variance')) +
    ggtitle(sprintf("Bias-Variance trade-off on Demand (n=%s)", n)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5),
          panel.spacing = unit(0, "lines"), 
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          text = element_text(family = "Times", size=20),
          strip.text = ggtext::element_markdown(size=16, margin=unit(c(7, 0, 4, 0), "pt")),
          axis.title.x = element_text(vjust=-1),
          plot.margin = margin(b=10, t=1, r=1, l=1)) +
    facet_wrap(~ method, ncol=5, labeller=labeller(method = method_names))
  
  p
  
  ggsave(sprintf("demand_biasvariance%i.png", n), p, path="~/Desktop", dpi=320, width = 11, height = 6, units = "in")
  
}
