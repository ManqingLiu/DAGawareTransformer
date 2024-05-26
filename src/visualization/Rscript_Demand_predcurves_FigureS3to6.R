library(ggplot2)
library(ggtext)
library(dplyr)
library(latex2exp)

# Set the working directory to results/
df = read.csv("experiments/results/proximal/E_Y_givenA.csv")

p <- ggplot(df) + 
geom_smooth(method='loess', formula=y ~ x, aes(x=A, y=EY_givenA), color='green', size=0.65) +
geom_smooth(method='loess', formula=y ~ x, aes(x=A, y=EY_doA), color='black', size=0.65) +
coord_cartesian(ylim = c(0, 100), xlim = c(8, 32)) +
xlab("Ticket price (A)") +
ylab(TeX(r'(Ticket sales: $E \[ Y^a \] $ or $E \[ Y | A \] $)')) +
theme_bw() +
theme(legend.position="none", 
        panel.spacing = unit(0, "lines"), 
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        text = element_text(family = "Times", size=20),
        strip.text = ggtext::element_markdown(size=16, margin=unit(c(7, 0, 4, 0), "pt")),
        axis.title.x = element_text(vjust=-1),
        plot.margin = margin(b=10, t=1, r=1, l=1))

p
  
ggsave("demand_YdoA_vs_Yobs.png", p, path="~/Desktop", dpi=320, width = 11, height = 6, units = "in")
