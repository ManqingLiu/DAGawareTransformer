#install.packages("geepack") # install package if required
library("geepack")
library("readxl")

nhefs <- read_excel("CHANGE DIRECTORY/nhefs.xlsx")
nhefs$cens <- ifelse(is.na(nhefs$wt82), 1, 0)

table(nhefs$cens)
prop.table(table(nhefs$cens))

################################################
# Adjusting for confounding and selection bias #
################################################

# estimation of denominator of treatment weights
denom.tx <- glm(qsmk ~ sex + race + age + I(age^2) + 
                   as.factor(education) + smokeintensity + 
                   I(smokeintensity^2) + smokeyrs + I(smokeyrs^2) + 
                   as.factor(exercise) + as.factor(active) + as.factor(hbp), 
                 family = binomial(), data = nhefs)
summary(denom.tx)
pd.qsmk <- predict(denom.tx, type = "response")

# estimation of numerator of treatment weights
numer.tx <- glm(qsmk~1, family = binomial(), data = nhefs)
summary(numer.tx)
pn.qsmk <- predict(numer.tx, type = "response")

# estimation of denominator of censoring weights
denom.cens <- glm(cens ~ qsmk + sex + race + age + I(age^2) + 
                    as.factor(education) + smokeintensity + 
                    I(smokeintensity^2) + smokeyrs + I(smokeyrs^2) + 
                    as.factor(exercise) + as.factor(active) + as.factor(hbp), 
                  family = binomial(), data = nhefs)
summary(denom.cens)

pd.cens <- 1-predict(denom.cens, type = "response")

# estimation of numerator of censoring weights
numer.cens <- glm(cens~qsmk, family = binomial(), data = nhefs)
summary(numer.cens)
pn.cens <- 1-predict(numer.cens, type = "response")

nhefs$sw.a <- ifelse(nhefs$qsmk == 0, ((1-pn.qsmk)/(1-pd.qsmk)),
                     (pn.qsmk/pd.qsmk))
nhefs$sw.c <- pn.cens/pd.cens
nhefs$sw <- nhefs$sw.c*nhefs$sw.a

summary(nhefs$sw.a)
#sd(nhefs$sw.a)
#summary(nhefs$sw.c)
#sd(nhefs$sw.c)
summary(nhefs$sw)
#sd(nhefs$sw)

msm.sw <- geeglm(wt82_71~qsmk, data=nhefs, 
                 weights=sw, id=seqn, corstr="independence")
summary(msm.sw)

beta <- coef(msm.sw)
SE <- coef(summary(msm.sw))[,2]
lcl <- beta-qnorm(0.975)*SE 
ucl <- beta+qnorm(0.975)*SE
cbind(beta, lcl, ucl)
