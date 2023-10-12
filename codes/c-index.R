#######################################################################
library(foreign)
library(rms)

data <- read.csv("test.csv",header=T,sep=",") 
head(data)


coxm2 <- cph(Surv(data$T,data$fustat==1)~ data$PH,
            x=T,y=T,data=data,surv=T,time.inc=48)

cal2 <- calibrate(coxm2, cmethod='KM', method='boot', u=48,m=20, B=1000)
plot(cal2,lwd=2,lty=1,errbar.col=c(rgb(255,255,255,maxColorValue=255)),
     xlim=c(0,1),ylim=c(0,1),xlab="Predicted Probabilityof 48 m Pregency",
     ylab="Actual 48 m Pregency",
     col=c(rgb(98,192,83,maxColorValue=255)))
lines(cal2[,c("mean.predicted","KM")],type="b",lwd=2,col=c(rgb(98,192,83,maxColorValue=255)),pch=16)
abline(0,1,lty=3,lwd=2,col=c(rgb(0,118,192,maxColorValue=255)))
###########################################################################
library(survcomp)
cindex <- concordance.index(data$PH,surv.time = data$T, surv.event = data$fustat,method = "noether")
cindex$c.index; cindex$lower; cindex$upper
