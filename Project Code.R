#Part 1
#How do the predictors influence the response?

#Libraries
library(lattice)
library(fastDummies)
library(caTools)
library(glmnet)
library(dplyr)
library(ggplot2)
library(GGally)
library(reshape)
library(car)
## Box-Cox Transformation


# library(MASS)
# fm1=lm()
# boxcox(fm1, lambda = seq(-0.1, 0.2, 0.01), plotit = TRUE,grid=TRUE)

#Reading the data
setwd("D:/")
data=read.csv("Diamonds.csv")
attach(data)
str(data)
View(data)


#Scatterplot of Continuous Predictors 
#vs Response

#carat
ggplot(NULL,aes(x=carat,y=price))+
  geom_point(size=1,col=2)+
  labs(title="Scatterplot",
       subtitle="Price vs Carat",
       x="\nCarat",
       y="Price",
       col="Index")

#depth
ggplot(NULL,aes(x=depth,y=price))+
  geom_point(size=1,col=3)+
  labs(title="Scatterplot",
       subtitle="Price vs Depth",
       x="\nDepth",
       y="Price",
       col="Index")

#table
ggplot(NULL,aes(x=table,y=price))+
  geom_point(size=1.5,col=5)+
  labs(title="Scatterplot",
       subtitle="Price vs Table",
       x="\nTable",
       y="Price",
       col="Index")

#x
ggplot(NULL,aes(x=x,y=price))+
  geom_point(size=1,col=4)+
  labs(title="Scatterplot",
       subtitle="Price vs Length",
       x="\nLength(in mm)",
       y="Price",
       col="Index")

#y
ggplot(NULL,aes(x=y,y=price))+
  geom_point(size=1,col=6)+
  labs(title="Scatterplot",
       subtitle="Price vs Width",
       x="\nWidth(in mm)",
       y="Price",
       col="Index")

#z
ggplot(NULL,aes(x=z,y=price))+
  geom_point(size=1,col=7)+
  labs(title="Scatterplot",
       subtitle="Price vs Depth",
       x="\nDepth(in mm)",
       y="Price",
       col="Index")

#Histogram of continuous predictors

#carat
ggplot(NULL,aes(x=log(carat)))+
  geom_histogram(fill=4,col=1,bins=25,
                 aes(y=..density..))+
  labs(title="Histogram of log(Carat)",
       x="\nCarat",
       col="Index")

#table
ggplot(NULL,aes(x=table))+
  geom_histogram(fill=3,col=1,bins=8,
                 aes(y=..density..))+
  labs(title="Histogram of table",
       x="\ntable",
       col="Index")

#depth
ggplot(NULL,aes(x=log(depth)))+
  geom_histogram(fill=2,col=1,bins=8,
                 aes(y=..density..))+
  labs(title="Histogram of Depth",
       x="\nDepth",
       col="Index")

#x
ggplot(NULL,aes(x=x))+
  geom_histogram(fill=5,col=1,bins=10,
                 aes(y=..density..))+
  labs(title="Histogram of Length(in mm)",
       x="\nLength(in mm)",
       col="Index")

#y
ggplot(NULL,aes(x=log(y)))+
  geom_histogram(fill=4,col=1,bins=20,
                 aes(y=..density..))+
  labs(title="Histogram of Width(in mm)",
       x="\nWidth(in mm)",
       col="Index")

#z
ggplot(NULL,aes(x=log(z)))+
  geom_histogram(fill=3,col=1,bins=20,
                 aes(y=..density..))+
  labs(title="Histogram of Depth(in mm)",
       x="\nDepth(in mm)",
       col="Index")

#Boxplot of Categorical Variables

#Clarity
ggplot(data=NULL,aes(x=as.factor(clarity),y=log(price)
                     ,fill=clarity))+
  geom_boxplot()+
  labs(title = 'Boxplot :: Price vs Clarity',
       x='Clarity',
       y='Price of Diamond')

#Cut
ggplot(data=NULL,aes(x=as.factor(cut),y=price
                     ,fill=cut))+
  geom_boxplot()+
  labs(title = 'Boxplot :: Price vs Cut',
       x='Cut',
       y='Price of Diamond')

#Color
ggplot(data=NULL,aes(x=as.factor(color),y=price
                     ,fill=color))+
  geom_boxplot()+
  labs(title = 'Boxplot :: Price vs Color',
       x='Color',
       y='Price of Diamond')

#Pair-Pair plot of continuous variables
#splom(data[,c(2,6,7,8,9,10,11)])

#Correlation Heat-map
data_con=data[,c(2,6,7,8,9,10,11)]
cor(data_con)
corr = data.matrix(cor(data_con[sapply(data_con,
                                           is.numeric)])) 
mel = melt(corr)
mel
ggplot(mel, aes(X1,X2))+geom_tile(aes(fill=value)) +
  geom_text(aes(label = round(value, 4)))+
  scale_fill_gradient2(low='#003300',mid = '#ffff99' ,high='#66b3ff') +
  labs(title = 'Correlation Heatmap')

#Observing the response
ggplot(NULL,aes(x=price))+
  geom_histogram(fill=2,col=1,bins=20,
                 aes(y=..density..))+
  labs(title="Histogram of Price",
       x="\nPrice",
       col="Index")

#Note : Positively skewed , so log transformation done

ggplot(NULL,aes(x=log(price)))+
  geom_histogram(fill=4,col=1,bins=20,
                 aes(y=..density..))+
  labs(title="Histogram of log(Price)",
       x="\nlog(Price)",
       col="Index")





#Part 2
#How to find a simple model for prediction of price of Diamond?

#Dummy variable creation for the Categorical Variables
data=data[,-1]
data1=dummy_cols(data,select_columns = c('cut','color','clarity'))
#Working data
data2=data1[,-c(2,3,4,7,15,22,30)]

#Note : From the correlation matrix carat, x , y, z seem to have multicollinearity
summary(lm(log(data$price)~data2$x))
summary(lm(data$price~data2$y)) #0.749
summary(lm(data$price~data2$z)) #0.7418
summary(lm(data$price~data2$carat)) #0.8493

#Ordinary Multiple Linear Regression
data3=data2[,-c(4,5,6)]
View(data3)
View(data1)
#Train Set and Test Set
y=data1$price
y
set.seed(seed=4567)
train=which(sample.split(y,0.6)==TRUE)
train
train_data=cbind(y_p=log(y[train]),data1[train,-c(2,3,4,7,8,9,10,15,22,30)])
test_data=cbind(y_p=log(y[-train]),data1[-train,-c(2,3,4,7,8,9,10,15,22,30)])
head(train_data)
View(train_data)
y1=train_data$y_p
model1=lm(y_p~.,train_data)
model1
summary(model1)$coefficients #0.8876
ei1=residuals(model1)
hii1=hatvalues(model1)
sum((ei1/(1-hii1))^2)  #4103.543

val=cbind(fitted=predict(model1,test_data),actual=test_data[,1])
res=rstandard(model1)
fit_tr=predict(model1,train_data)
#Plot of Residuals
ggplot(NULL,aes(x=fit_tr,y=res))+
  geom_point(size=1,col=3)+
  labs(title="Scatterplot of Residuals",
       subtitle="Multiple Linear Regression",
       x="Fitted values",
       y="Residuals",
       col="Index")



rSqr=sum((val[,1]-val[,2])^2);#2124.267

#actual and fitted value in test set
ggplot(NULL,aes())+
  geom_histogram(col=1,
                 aes(val[,1],fill=4,y=..density..))+
  geom_histogram(col=2,
                 aes(val[,2],fill=3,y=..density..))+
  labs(title="Histogram of Actual value and Fitted value of Price",
       x="\nPrice",
       col="Index") #sky=fitted ,blue=actual


meltdata=melt(val)
p1 = ggplot(data=meltdata,aes(value,fill=X2))+
  geom_density(alpha=.6)+
  labs(title="Density Plot of Actual value and Fitted value of Price",
       subtitle = "Test-Set",
       col="Index")
p1

#Lasso Regression
set.seed(seed=1234)
train=which(sample.split(y,0.6)==TRUE)
train
train_data=cbind(y_p=log(y[train]),data1[train,-c(2,3,4,7,15,22,30)])
test_data=cbind(y_p=log(y[-train]),data1[-train,-c(2,3,4,7,15,22,30)])
head(train_data)
View(data1)
X= model.matrix( ~ . - y_p - 1,train_data)
fm.lasso= glmnet(X, train_data$y_p, alpha = 1)
plot(fm.lasso, xvar = "lambda", label = TRUE)
plot(fm.lasso, xvar = "dev", label = TRUE)
cv.lasso <- cv.glmnet(X, train_data$y_p, alpha = 1, nfolds = 50)
plot(cv.lasso) #21 non- zero predictor ;log lambda= -5.5

s.cv <- c(lambda.min = cv.lasso$lambda.min, lambda.1se = cv.lasso$lambda.1se)
round(coef(cv.lasso, s = s.cv), 3) # corresponding coefficients

View(test_data)
fit_lasso=predict(cv.lasso,s="lambda.1se",newx=data.matrix(test_data[,-1]))

View(train_data)
#Least square using Lasso Model
data_new=train_data[,-c(2,3,4,8,9,10,11,15,22,23)]
model_lasso=lm(y_p~.,data_new)
summary(model_lasso)
res1=residuals(model_lasso)
hii2=hatvalues(model_lasso)
sum((res1/(1-hii2))^2) #1728.387


sum
fitt1=predict(model_lasso)

#Plot of Residuals
ggplot(NULL,aes(x=fitt1,y=res1))+
  geom_point(size=1,col=5)+
  labs(title="Scatterplot of Residuals",
       x="Fitted values",
       y="Residuals",
       col="Index")

#Detection of three outliers
outliers=which(res1>2|res1< -2)
outliers
out1=c(6750,7230,16739,17664,30325,31012)
train_data1=train_data[-out1,]


data_new1=train_data1[,-c(2,3,4,8,9,10,11,15,22,23)]
model_lasso=lm(y_p~.,data_new1)
summary(model_lasso)
res1=residuals(model_lasso)
hii2=hatvalues(model_lasso)
sum((res1/(1-hii2))^2) #1728.387

res2=resid(model_lasso)
fitt2=predict(model_lasso)

#Plot of Residuals
ggplot(NULL,aes(x=fitt2,y=res2))+
  geom_point(size=1,col=2)+
  labs(title="Scatterplot of Residuals",
       subtitle="Lasso Regression",
       x="Fitted values",
       y="Residuals",
       col="Index")

durbinWatsonTest(model_lasso)
ncvTest(model_lasso)




View(data_new1)
model_lasso1=lm(y_p~.,data_new1,weights = (1/res2)^2)
summary(model_lasso1)
res3=resid(model_lasso1)
fitt3=predict(model_lasso1)

ggplot(NULL,aes(x=fitt3,y=res3))+
  geom_point(size=1,col=3)+
  labs(title="Scatterplot of Residuals",
       x="Fitted values",
       y="Residuals",
       col="Index")
  



#Comparing Density Plot in Test Set
data_new_test=test_data[,-c(2,3,4,8,9,10,11,15,22,23)]
View(data_new_test)
val1=cbind(fitted=predict(model_lasso,data_new_test),actual=data_new_test[,1])

meltdata1=melt(val1)
p1 = ggplot(data=meltdata1,aes(value,fill=X2))+
  geom_density(alpha=.6)+
  labs(title="Density Plot of Actual value and Fitted value of Price",
       subtitle = "Test-Set",
       col="Index")

p1

rSqr=sum((val1[,1]-val1[,2])^2);#2124.267
rSqr