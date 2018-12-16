setwd("/Users/pratikeshsingh/Desktop/R git/")
test=read.csv("test.csv")
train=read.csv("train.csv")
test$work.experience=1
combi=rbind(train,test)
str(combi)
is.na(combi)
table(is.na(combi))
summary(combi)
summary(train$Salary)
str(train$Salary)##Salary is spread across a range of ???35k - ???40lac and is right

#skewed with the median at???3lac and the interquartile range from

#???1.8lac to ???3.7lac.
summary(train$X10percentage)

hist(train$X10percentage)

summary(train$X12percentage)
hist(train$X12percentage)

##The 10th and 12th percentage has no missing values and outliers.

#10th percentage is slightly left skewed and 12th percentage follows

#normal distribution.

summary(combi$collegeGPA)
combi$collegeGPA=ifelse(combi$collegeGPA<10,(combi$collegeGPA)*10,combi$collegeGPA)
# #College GPA has max value of 99.93 and minimum value of 6.45.
# This indicated that the values are not in the same scale. Some of
# the college GPA could be in percentile and some in absolute
# numbers. There was no indicator to distinguish those. All the
# values which were below 10 were converted to 100 point scale
summary(combi$Quant)
hist(combi$Quant)

summary(combi$Logical)
hist(combi$Logical)
summary(combi$English)
hist(combi$English)
library(psych)
pairs.panels(combi[24:26])

# Quant, Logical and English scores don't have any missing values.
# The data follows normal distribution with no skews. These
# AMCAT scores show a high correlation with the salary and Quant
# has the highest correlation.

hist(combi$Domain)

table(combi$CollegeTier)
median(train$Salary[which(train$CollegeTier==1)])
median(train$Salary[which(train$CollegeTier==2)])

# 90% of the students in this dataset are from Tier 2 colleges. In the
# data given that the median salary of the Tier 1 college graduates is
# higher than the Tier2 college graduates.

ifelse(train$DOL=='present',"sys.time",train$DOL)##as there were missing values in DOL COLUMN SO THEY WERE REPLACED BY PRESENT DATE

# The dataset contains graduates with different years of experience.
# As expected, the salary range is less for graduates with less years
# of experience ,MADE A NEW COLUMN WITH A NAME OF WORK EX BY SUBTRACTING DOL FROM DOJ

library(ggplot2)
boxplot(train$Salary)##TO CHeck the outliers
ggplot(train,aes(x=work.experience,y=Salary))+
  geom_point(size=2.5,color="red")+xlab("workex")+ylab("salary")+ggtitle("salary vs workex")+xlim(c(0,5))+
  ylim(c(0e+00,1e+06))##to remove value outlier values in grphs we set the x and y limit to concentrated points

table(train$Gender)
max(train$Salary[which(train$Gender=="m")])
max(train$Salary[which(train$Gender=="f")])
min(train$Salary[which(train$Gender=="m")])
min(train$Salary[which(train$Gender=="f")])
# Approximately 75% of the graduates in this dataset are males. The
# data indicates there is no salary variation based on gender.
##degree
summary(train$Degree)
train$Degree[train$Degree=='M.Sc. (Tech.)']='M.Tech./M.E.'
summary(train$Degree)
# Approximately 92% of the graduates in this dataset has
# B.Tech/BE degree. There are only 2 MSc. (Tech) degree holders
# which we merged with M. Tech. The data shows that the median
# salary of B.Tech / BE graduates is higher than MCA post
# graduates.
###specialization
library(plyr)
levels(train$Specialization)
train$Specialization=revalue(train$Specialization,
                             c("computer and communication engineering"="computer application","computer engineering"="computer networking","computer science"="computer science & engineering"))
train$Specialization=revalue(train$Specialization,c("computer application"="computer networking","computer science & engineering"="computer science and technology"))
train$Specialization=revalue(train$Specialization,c("computer science and technology"="computer networking"))                             
train$Specialization=revalue(train$Specialization,c('electrical and power engineering'="electrical engineering","electronics & instrumentation eng"= "electronics & telecommunications" ,"electronics and communication engineering"="electronics and computer engineering","electronics and electrical engineering"="electronics and instrumentation engineering","electronics engineering"="embedded systems technology"))
train$Specialization=revalue(train$Specialization,c("embedded systems technology"="electronics and instrumentation engineering","electronics and computer engineering"="electronics & telecommunications", "electronics" ="electrical engineering"))
train$Specialization=revalue(train$Specialization,c( "electronics and instrumentation engineering" ="electronics & telecommunications"))
train$Specialization=revalue(train$Specialization,c( "electrical engineering"="electronics & telecommunications"))
train$Specialization=revalue(train$Specialization,c("applied electronics and instrumentation"="control and instrumentation engineering","biomedical engineering" ="biotechnology","ceramic engineering"="civil engineering","industrial & production engineering"="industrial engineering"                 
                                                    ,"information & communication technology"  ="information science"                    
                                                    ,"information science engineering"         ="information technology"                 
                                                    ,"instrumentation and control engineering"= "instrumentation engineering"            
                                                    ,"internal combustion engine"   =           "mechanical & production engineering"    
                                                    ,"mechanical and automation"  =             "mechanical engineering"                 
                                                    ,"mechatronics"                =            "metallurgical engineering"              
                                                    ,"other"                          =         "polymer technology"))                     
train$Specialization=revalue(train$Specialization,c("aeronautical engineering"="automobile/automotive engineering",
                                                    
                                                    "polymer technology"        =               "chemical engineering"                   
                                                    ,"telecommunication engineering"=                    "electronics & telecommunications"       
                                                    ,"industrial & management engineering"  =   "industrial engineering"                 
                                                    ,"information science"                  =   "information technology"                 
                                                    ,"instrumentation engineering"           =  "mechanical & production engineering"    
                                                    ,"mechanical engineering"               =   "metallurgical engineering"     )) 
train$Specialization=revalue(train$Specialization,c(      
  "biotechnology"="chemical engineering",                         
  "electronics & telecommunications"="power systems and automation"  ,"industrial engineering" ="metallurgical engineering"))          
# ###[1] "automobile/automotive engineering" "power systems and automation"     
# [3] "chemical engineering"              "civil engineering"                
# [5] "information technology"            "metallurgical engineering" 
# This categorical field had 50 unique values which were grouped
# into 5 specializations   55% of the students in this dataset are from
# computer branches and 35% from electronics.
table(train$Specialization)

levels(train$X10board
)                                

train$X10board=ifelse(train$X10board!="cbse","others",train$X10board)
train$X10board=ifelse(train$X10board=="61","cbse",train$X10board)
train$X10board=ifelse(train$X10board!="cbse","others",train$X10board)
train$X12board=ifelse(train$X12board!="cbse","others",train$X12board)
train$X12board=ifelse(train$X12board!="78","cbse",train$X12board)

train$X12board=ifelse(train$X12board=="78","others",train$X12board)
# 10th board and 12th board had more than 200 unique categorical
# values. We grouped them into 2 categories, namely CBSE,
# and others board.

###model creation
linear=lm(train$Salary~ train$Quant+train$GraduationYear + train$X10percentage +train$collegeGPA + train$English + train$X12percentage +
            train$CollegeTier + train$CollegeCityTier +train$conscientiousness +
            train$extraversion +train$openess_to_experience)
##linear model

summary(linear)
par(mfrow=c(2,2))
plot(linear)
#check rmse value
library("Metrics")
rmse(train$Salary,exp(linear$fitted.values))


###building an decison tree model
library(rpart)
library(e1071)
library(rpart.plot)
library(caret)
##setting tree control parameters
fitcontrol=trainControl(method = "cv",number = 5)
cartGrid=expand.grid(.cp=(1:50)*0.01)
tree_model=train(train$Salary~ train$Quant+train$GraduationYear + train$X10percentage +train$collegeGPA + train$English + train$X12percentage +
                   train$CollegeTier + train$CollegeCityTier +train$conscientiousness +
                   train$extraversion +train$openess_to_experience, method = "rpart", trControl = fitcontrol, tuneGrid = cartGrid)
print(tree_model)
main_tree=rpart(Salary ~ ., data = new_train, control = rpart.control(cp=0.01))
prp(main_tree)
pre_score=predict(main_tree, type = "vector")
rmse(train$Salarys, pre_score)