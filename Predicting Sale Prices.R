
# Setting working directory

setwd("C:/Machine-Learning/House Prices (Kaggle)")

# Loading libraries

load.libraries <- c('data.table', 'testthat', 'gridExtra',
                    'corrplot', 'GGally', 'ggplot2', 'e1071', 'dplyr',
                    'rpart', 'Amelia', 'rattle', 'randomForest', 'VIM', 
                    'mice', 'caret')

sapply(load.libraries, require, character = TRUE)

# Importing datasets

train <- read.csv("train.csv")
test <- read.csv("test.csv")


# ------------------------------------------------------------------------------------ #
#                                   Data Exploration                                   #
# ------------------------------------------------------------------------------------ #

# Looking at missing data

  # Training set

    # Percent missing in overall dataset
    sum(is.na(train)) / prod(dim(train)) 
  
    # Missing by column
    sort(colSums(sapply(train, is.na)), decreasing = T)
  
    sort(colSums(sapply(train, is.na)), decreasing = T) / nrow(train)
    
    # Missing by row
    sum(complete.cases(train))
    table(rowSums(sapply(train, is.na)))
    
    # Missingness pattern
    mice::md.pattern(train)
    
    # Plot 1  
    aggr_plot <- VIM::aggr(train, col=c('navyblue','gray'), numbers=TRUE, 
                      sortVars=TRUE, labels=names(data)[1:15], cex.axis=.4,
                      gap=1, ylab=c("Histogram of missing data","Pattern"),
                                    combined=T)
    
    # Plot 2
    missmap(train)
    
  # Testing set
    
    # Percent missing in overall dataset
    sum(is.na(test)) / prod(dim(test))
    
    # Missing by column
    colSums(sapply(test, is.na))
    
    # Missing by row
    sum(complete.cases(test))
    table(rowSums(sapply(test, is.na)))
    
    # Missingness pattern
    mice::md.pattern(test)
    
    # Plot 1  
    aggr_plot <- VIM::aggr(test, col=c('navyblue','gray'), numbers=TRUE, 
                           sortVars=TRUE, labels=names(data)[1:15], cex.axis=.4,
                           gap=1, ylab=c("Histogram of missing data","Pattern"),
                           combined=T)
    
    # Plot
    missmap(test)
    
# Looking at structure of variables (numeric vs. character vs. factor)

fac_var <- names(train)[which(sapply(train, is.factor))]
numeric_var <- names(train)[which(sapply(train, is.numeric))]
char_var <- names(train)[which(sapply(train, is.character))]

# Splitting training dataset

train_cat <- train[,fac_var]
train_cont <- train[,numeric_var]

# Loading user defined functions

source("Prediction (functions).R")

# Plotting categorical variables

doPlots(train_cat, fun = plotHist, ii = 1:4, ncol = 2)
doPlots(train_cat, fun = plotHist, ii = 5:8, ncol = 2)
doPlots(train_cat, fun = plotHist, ii = 9:12, ncol = 2)
doPlots(train_cat, fun = plotHist, ii = 13:16, ncol = 2)
doPlots(train_cat, fun = plotHist, ii = 17:20, ncol = 2)

# Exploring correlations

correlations <- cor(na.omit(train_cont))

row_indic <- apply(correlations, 1, function(x) sum(x > 0.3 | x < -0.3) > 1)

correlations <- correlations[row_indic,row_indic]
corrplot(correlations, method="square")

# ------------------------------------------------------------------------------------ #
#                                Predictive Modeling                                   #
# ------------------------------------------------------------------------------------ #

# Dropping columns with a high number of missing values

drop.cols <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage")
train2 <- train %>% select(-one_of(drop.cols))

# Imputing values for predictors in training dataset

train.imputed <- rfImpute(SalePrice ~ ., data = train2, seed=42) 

# Imputing values for predictors in test dataset

test.imputed <- mice(xtest, m=6, method='cart', printFlag=FALSE, seed=42)
test.imputed2 <- complete(test.imputed)

# Check
sapply(train.imputed, function(x) sum(is.na(x)))
sapply(test.imputed, function(x) sum(is.na(x)))

# Using Rpart to create a single tree for visualization

tree1 <- rpart(SalePrice ~ ., data = train.imputed, method="anova")
fancyRpartPlot(tree1)

# Using random forest for actual predictive model

model_rf <- randomForest(SalePrice ~ . -Id, data = train.imputed)

# Results of random forest

  # Variable importance
    
    importance.scores <- randomForest::importance(model_rf)
    varImpPlot(model_rf)

  # Sorting importance scores
    
    top.scores <- data.frame(var = dimnames(importance.scores)[[1]], 
                             importance = as.numeric(importance.scores))
    
    i <- top.scores %>%
      arrange(desc(importance))
    
  # OOB error

    plot(model_rf) 

  # Results from 1st tree

    preds1 <- getTree(model_rf, 1)

# Predicting Sale Price
    
    predict(model_rf, test2)
    
# Note, we need to make sure that the levels for each factor are the same in both datasets
    
    xtest <- rbind(train.imputed[1,-1], test2)
    xtest <- xtest[-1,]    

# Predicting Sale Price
    
    Prediction <- predict(model_rf, test.imputed2)
    
# Preparing submission
    
    submit <- data.frame(Id = test$Id, SalePrice = Prediction)
    write.csv(submit, file = "secondforest.csv", row.names = FALSE)

