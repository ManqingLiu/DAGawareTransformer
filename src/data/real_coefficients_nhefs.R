
path2 <- "/Users/manqingliu/Dropbox/Harvard/Courses spring 2022/EPI289/NHEFS"
setwd(path2)

library("readxl")
nhefs <- read_excel("nhefs.xlsx")

#### no interactions in both models
## treatment model
dat <- subset(nhefs, !(is.na(wt82_71)))
fit_A <- glm(qsmk ~ sex + race + age +  education
           + smokeintensity 
           + smokeyrs + exercise
           + active + wt71, family=binomial(),
           data=dat)
summary_fit_A <- summary(fit_A)
coefficients_A <- coef(fit_A)
print(coefficients_A)

## outcome model
fit_Y <- glm(wt82_71 ~ qsmk + sex + race + age +  education
             + smokeintensity 
             + smokeyrs + exercise
             + active + wt71, data = dat)
summary_fit_Y <- summary(fit_Y)
coefficients_Y <- coef(fit_Y)
print(coefficients_Y)

