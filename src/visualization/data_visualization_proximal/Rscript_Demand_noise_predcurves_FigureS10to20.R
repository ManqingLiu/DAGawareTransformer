library(ggplot2)
library(ggpp)
library(dplyr)
library(latex2exp)
library(extrafont)
loadfonts()

# Set the working directory to results/
setwd("~/")
df = read.csv("aggregated_results_for_figures/demand_noise_predcurve_data.csv")

method_names <- c(kpv = "KPV", 
                  pmmr = "PMMR", 
                  cevae = "CEVAE", 
                  dfpv = "DFPV", 
                  nmmr_u = "NMMR U", 
                  nmmr_v = "NMMR V", 
                  twosls = "2SLS", 
                  naivenet_awy = "Naive net", 
                  linear_reg_awy = "LS", 
                  linear_reg_aw2y = "LS-QF")

# ordering by descending MSE in first panel
# methods = c('pmmr', 'kpv', 'naivenet_awy', 'cevae', 'twosls', 'dfpv', 'linear_reg_awy', 'linear_reg_aw2y', 'nmmr_u', 'nmmr_v')
methods = c('naivenet_awy')
df_subset = df[which(df$method %in% methods),]

for (method in methods) {
  df_method = df_subset[which(df_subset$method == method),]

  p <- ggplot(df_method) +
    ggtitle(sprintf("%s noise sensitivity", method_names[method])) +
    geom_smooth(method='loess', formula=y ~ x, aes(x=A, y=true_EY_doA), color='black', size=0.4, alpha=0.75) +
    geom_line(aes(x=A, y=pred_EY_doA, group=rep), color="#438BC4", linetype='dotted', alpha=1, size=0.6) +
    coord_cartesian(xlim = c(8, 32), ylim = c(-70, 170)) +
    scale_y_continuous(breaks=c(-50, 0, 50, 100, 150), 
                       sec.axis = sec_axis(~ ., name=TeX(r'(W noise ($\sigma^2_W$))'), breaks=NULL, labels=NULL)) +
    scale_x_continuous(sec.axis = sec_axis(~ ., name=TeX(r'(Z noise ($\sigma^2_Z$))'), breaks=NULL, labels=NULL)) +
    scale_colour_distiller(palette='Set1') +
    xlab("Ticket price (A)") +
    ylab(TeX(r'(Ticket sales: $E \[ Y^a \] $)')) +
    theme_bw() +
    theme(legend.position="none",
          panel.spacing = unit(0, "lines"),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          text = element_text(family = "Times", size=20),
          axis.title.x = element_text(vjust=-1),
          plot.margin = margin(b=10, t=1, r=1, l=1),
          plot.title = element_text(hjust = 0.5)) +
    facet_grid(vars(W_noise), vars(Z_noise))

  if (method == "dfpv") {
    p <- p + geom_point(data = data.frame(method='dfpv', Z_noise=c(4, 8), W_noise=c(0.5, 0)), aes(x=20, y=c(-50, 140)), shape="*", size=8, inherit.aes=FALSE)
  }

  ggsave(sprintf("demand_noise_predcurve_%s.png", method), p, path="~/Desktop", dpi=320, width = 22, height = 12, units = "in")
}
