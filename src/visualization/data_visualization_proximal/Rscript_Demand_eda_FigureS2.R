library(ggplot2)
library(cowplot)

# Set the working directory to results/
setwd("~/")

df = read.csv("aggregated_results_for_figures/demand_eda_data.csv")

# Scatter plot of U vs. Z
plot1 <- ggplot(df) +
  geom_point(aes(x=Z1, y=U), color="red", alpha=0.5, size=1) +
  geom_point(aes(x=Z2, y=U), color="orange", alpha=0.5, size=1) +
  ylab("Demand (U)") +
  xlab("Cost of Fuel (Z)") +
  theme_bw() + 
  theme(legend.title=element_blank(),
        legend.text=element_text(size=20),
        text = element_text(family = "Times", size=20),
        plot.margin = margin(0.1, 0.25, 0.25, 0.1, "cm"))

# Scatter plot of U vs. W
plot2 <- ggplot(df) +
  geom_point(aes(x=W, y=U), alpha=0.5, size=1, color="purple") +
  ylab("Demand (U)") +
  xlab("Page views (W)") +
  theme_bw() + 
  theme(legend.title=element_blank(),
        legend.text=element_text(size=20),
        text = element_text(family = "Times", size=20),
        plot.margin = margin(0.1, 0.25, 0.25, 0, "cm")) 

# Scatter plot of Y_obs vs. A_obs and Y_struct vs. doA
plot3 <- ggplot(df) + 
  geom_point(aes(x=A_obs, y=Y_obs, color="points"), alpha=0.5, size=1) +
  geom_smooth(method='loess', formula=y~x, aes(x=doA, y=Y_struct, color="line"), size=0.75) +
  scale_color_manual(values = c("line" = "black", "points" = "blue")) +
  ylab("Sales (Y)") +
  xlab("Price (A)") +
  theme_bw() + 
  theme(legend.title=element_blank(),
        legend.text=element_text(size=20),
        legend.position = "none",
        text = element_text(family = "Times", size=20),
        plot.margin = margin(0.1, 0.25, 0.25, 0, "cm"))

p <- plot_grid(plot1, plot2, plot3, nrow=1, labels = c("A", "B", "C"), 
               label_fontfamily="serif", label_fontface="plain", label_size=20)

p

ggsave("demand_eda.png", p, path="~/Desktop", dpi=320, width = 11, height = 5, units = "in")
