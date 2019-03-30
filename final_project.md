### Overview

The objective of this project is to use the data collected through
wearable devices, such as Jawbone Up, Fitbit to monitor personal
activities and predict how they perform the exercise.

The data we use in this analysis is from the source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
We use the training data to create our model and test data to evaluate
the model performance.

### Data Exploration

First, we read in the data and explore the dataset to see the
attributes, data type. We also take a look at the distributiont of our
response variable (classe).

*Please see appendix for outputs*

    library(mlbench)
    library(caret)
    library(parallel)
    library(iterators)
    library(foreach)
    library(doParallel)
    library(corrplot)

    training = read.csv('pml-training.csv',header=T)
    testing = read.csv('pml-testing.csv',header=T)
    head(training)
    dim(training) # 19622 160
    str(training)

    table(training$classe)

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

### Data Pre-Processing

After viewing the data, we decided to convert factor variables to
numeric variables and check missing data. We tranformed variables with
missing values into indicators.

    training$kurtosis_roll_belt = as.numeric(as.character(training$kurtosis_roll_belt))
    training$kurtosis_picth_belt = as.numeric(as.character(training$kurtosis_picth_belt))
    training$skewness_roll_belt = as.numeric(as.character(training$skewness_roll_belt))
    training$skewness_roll_belt.1 = as.numeric(as.character(training$skewness_roll_belt.1))
    training$max_yaw_belt = as.numeric(as.character(training$max_yaw_belt))
    training$min_yaw_belt = as.numeric(as.character(training$min_yaw_belt))
    training$kurtosis_roll_arm = as.numeric(as.character(training$kurtosis_roll_arm))
    training$kurtosis_picth_arm = as.numeric(as.character(training$kurtosis_picth_arm))
    training$skewness_yaw_arm = as.numeric(as.character(training$skewness_yaw_arm))
    training$kurtosis_roll_dumbbell = as.numeric(as.character(training$kurtosis_roll_dumbbell))
    training$kurtosis_picth_dumbbell = as.numeric(as.character(training$kurtosis_picth_dumbbell))
    training$max_yaw_dumbbell = as.numeric(as.character(training$max_yaw_dumbbell))
    training$min_yaw_dumbbell = as.numeric(as.character(training$min_yaw_dumbbell))
    training$kurtosis_roll_forearm = as.numeric(as.character(training$kurtosis_roll_forearm))
    training$kurtosis_picth_forearm = as.numeric(as.character(training$kurtosis_picth_forearm))
    training$max_yaw_forearm = as.numeric(as.character(training$max_yaw_forearm))
    training$min_yaw_forearm = as.numeric(as.character(training$min_yaw_forearm))

    missing_cnt = sapply(training,function(x) sum(is.na(x)))
    table(missing_cnt) 

    ## missing_cnt
    ##     0 19216 19218 19221 19225 19226 19227 19248 19294 19296 19300 19301 
    ##    76    67     1     3     1     3     1     2     1     1     3     1

    for(col in names(training)){
      if(sum(is.na(training[,col]))>0)
        training[,col] = ifelse(is.na(training[,col]),0,1)
    }

### Model Building

We split the data into training and testing set using 70/30 split.
Excluded 4 variables that are not associated with our response variable.
Then we calculated the correlation of all the variables and keep the top
30 variables that are relatively highly correlated with our response
variable.

    set.seed(123)
    idx = createDataPartition(training$classe,p=0.7,list=FALSE)
    train = training[idx,]
    test = training[-idx,]

    yvar = as.numeric(train$classe)
    xvar = train[,sapply(train,is.numeric)]
    xvar = xvar[,-c(1,2,3,4)]

    allvar = cbind(xvar,yvar)
    m = abs(cor(allvar))
    corr_df = data.frame(row=rownames(m)[row(m)[upper.tri(m)]], 
                         col=colnames(m)[col(m)[upper.tri(m)]], 
                         corr=m[upper.tri(m)])
    corr_df_y = corr_df[which(corr_df$col=='yvar'),]
    corr_df_y2 = corr_df_y[order(-corr_df_y$corr),]

    # take top 30 variables that are most correlated with response

    corr_df_y2[1:30,c(1)]

    ##  [1] pitch_forearm        magnet_arm_x         magnet_belt_y       
    ##  [4] magnet_arm_y         accel_arm_x          accel_forearm_x     
    ##  [7] magnet_forearm_x     magnet_belt_z        pitch_arm           
    ## [10] magnet_arm_z         total_accel_forearm  magnet_dumbbell_z   
    ## [13] total_accel_arm      accel_dumbbell_x     magnet_forearm_y    
    ## [16] accel_arm_y          accel_belt_z         roll_arm            
    ## [19] total_accel_belt     pitch_dumbbell       accel_dumbbell_z    
    ## [22] roll_belt            magnet_dumbbell_x    total_accel_dumbbell
    ## [25] yaw_forearm          roll_dumbbell        yaw_arm             
    ## [28] accel_arm_z          magnet_forearm_z     gyros_dumbbell_y    
    ## 136 Levels: accel_arm_x accel_arm_y accel_arm_z ... yaw_forearm

    corr_org = cor(allvar)
    corrplot(corr_org[corr_df_y2[1:30,c(1)],corr_df_y2[1:30,c(1)]])

![](final_project_files/figure-markdown_strict/unnamed-chunk-4-1.png)

#### Random Forest

Use the top 30 variables we selected and create a random foreset model
using 5-fold cross-validation. Then we apply the model to the testing
set. The accurary for the testing set is 0.9867 so the **out-of-sample
error is 0.0133**. We consider this as a decent model so we decided to
use the model in our validation (test) set to predict the 20 new cases.

    cluster = makeCluster(detectCores() - 1)
    registerDoParallel(cluster)
    fitControl = trainControl(method = "cv", number = 5, allowParallel = TRUE)
    rf_model = train(classe ~ pitch_forearm+magnet_arm_x+magnet_belt_y+magnet_arm_y+accel_arm_x+
                     accel_forearm_x+magnet_forearm_x+magnet_belt_z+pitch_arm+magnet_arm_z+
                     total_accel_forearm+magnet_dumbbell_z+total_accel_arm+accel_dumbbell_x+magnet_forearm_y+
                     accel_arm_y+accel_belt_z+roll_arm+total_accel_belt+pitch_dumbbell+
                     accel_dumbbell_z+roll_belt+magnet_dumbbell_x+total_accel_dumbbell+yaw_forearm+
                     roll_dumbbell+yaw_arm+accel_arm_z+magnet_forearm_z+gyros_dumbbell_y, 
                     data=train, method="rf", trControl = fitControl)
    print(rf_model)

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    30 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10991, 10989, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9818732  0.9770669
    ##   16    0.9849310  0.9809383
    ##   30    0.9793256  0.9738426
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 16.

    plot(rf_model,main="Accuracy of Random forest model by number of predictors")

![](final_project_files/figure-markdown_strict/unnamed-chunk-5-1.png)

    test_pred = predict(rf_model,newdata=test)
    confusionMatrix(test$classe,test_pred)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668    3    0    0    3
    ##          B   11 1122    6    0    0
    ##          C    0   19 1002    5    0
    ##          D    0    0   26  936    2
    ##          E    0    1    0    2 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9867          
    ##                  95% CI : (0.9835, 0.9895)
    ##     No Information Rate : 0.2853          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9832          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9934   0.9799   0.9691   0.9926   0.9954
    ## Specificity            0.9986   0.9964   0.9951   0.9943   0.9994
    ## Pos Pred Value         0.9964   0.9851   0.9766   0.9710   0.9972
    ## Neg Pred Value         0.9974   0.9952   0.9934   0.9986   0.9990
    ## Prevalence             0.2853   0.1946   0.1757   0.1602   0.1842
    ## Detection Rate         0.2834   0.1907   0.1703   0.1590   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9960   0.9882   0.9821   0.9935   0.9974

### Prediction for the Validation Data

Before we predict the test set, we applied the same transformation as
training set: Convert factor variables to numeric variables and create
missing indicators.

The prediction result on the validation set is 100% accurate.

    testing$kurtosis_roll_belt = as.numeric(as.character(testing$kurtosis_roll_belt))
    testing$kurtosis_picth_belt = as.numeric(as.character(testing$kurtosis_picth_belt))
    testing$skewness_roll_belt = as.numeric(as.character(testing$skewness_roll_belt))
    testing$skewness_roll_belt.1 = as.numeric(as.character(testing$skewness_roll_belt.1))
    testing$max_yaw_belt = as.numeric(as.character(testing$max_yaw_belt))
    testing$min_yaw_belt = as.numeric(as.character(testing$min_yaw_belt))
    testing$kurtosis_roll_arm = as.numeric(as.character(testing$kurtosis_roll_arm))
    testing$kurtosis_picth_arm = as.numeric(as.character(testing$kurtosis_picth_arm))
    testing$skewness_yaw_arm = as.numeric(as.character(testing$skewness_yaw_arm))
    testing$kurtosis_roll_dumbbell = as.numeric(as.character(testing$kurtosis_roll_dumbbell))
    testing$kurtosis_picth_dumbbell = as.numeric(as.character(testing$kurtosis_picth_dumbbell))
    testing$max_yaw_dumbbell = as.numeric(as.character(testing$max_yaw_dumbbell))
    testing$min_yaw_dumbbell = as.numeric(as.character(testing$min_yaw_dumbbell))
    testing$kurtosis_roll_forearm = as.numeric(as.character(testing$kurtosis_roll_forearm))
    testing$kurtosis_picth_forearm = as.numeric(as.character(testing$kurtosis_picth_forearm))
    testing$max_yaw_forearm = as.numeric(as.character(testing$max_yaw_forearm))
    testing$min_yaw_forearm = as.numeric(as.character(testing$min_yaw_forearm))
    for(col in names(testing)){
      if(sum(is.na(testing[,col]))>0)
        testing[,col] = ifelse(is.na(testing[,col]),0,1)
    }

    testing_pred = predict(rf_model,newdata=testing)
    testing_pred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

    stopCluster(cluster)
    registerDoSEQ()

### Appendix

    head(training)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
    ## 1         no         11      1.41       8.07    -94.4                3
    ## 2         no         11      1.41       8.07    -94.4                3
    ## 3         no         11      1.42       8.07    -94.4                3
    ## 4         no         12      1.48       8.05    -94.4                3
    ## 5         no         12      1.48       8.07    -94.4                3
    ## 6         no         12      1.45       8.06    -94.4                3
    ##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
    ## 1                  0                   0                  
    ## 2                  0                   0                  
    ## 3                  0                   0                  
    ## 4                  0                   0                  
    ## 5                  0                   0                  
    ## 6                  0                   0                  
    ##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
    ## 1                  0                    0                               0
    ## 2                  0                    0                               0
    ## 3                  0                    0                               0
    ## 4                  0                    0                               0
    ## 5                  0                    0                               0
    ## 6                  0                    0                               0
    ##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
    ## 1              0            0             0              0            0
    ## 2              0            0             0              0            0
    ## 3              0            0             0              0            0
    ## 4              0            0             0              0            0
    ## 5              0            0             0              0            0
    ## 6              0            0             0              0            0
    ##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
    ## 1                   0                    0                   
    ## 2                   0                    0                   
    ## 3                   0                    0                   
    ## 4                   0                    0                   
    ## 5                   0                    0                   
    ## 6                   0                    0                   
    ##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
    ## 1                    0             0                0             0
    ## 2                    0             0                0             0
    ## 3                    0             0                0             0
    ## 4                    0             0                0             0
    ## 5                    0             0                0             0
    ## 6                    0             0                0             0
    ##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
    ## 1              0                 0              0            0
    ## 2              0                 0              0            0
    ## 3              0                 0              0            0
    ## 4              0                 0              0            0
    ## 5              0                 0              0            0
    ## 6              0                 0              0            0
    ##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
    ## 1               0            0         0.00         0.00        -0.02
    ## 2               0            0         0.02         0.00        -0.02
    ## 3               0            0         0.00         0.00        -0.02
    ## 4               0            0         0.02         0.00        -0.03
    ## 5               0            0         0.02         0.02        -0.02
    ## 6               0            0         0.02         0.00        -0.02
    ##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
    ## 1          -21            4           22            -3           599
    ## 2          -22            4           22            -7           608
    ## 3          -20            5           23            -2           600
    ## 4          -22            3           21            -6           604
    ## 5          -21            2           24            -6           600
    ## 6          -21            4           21             0           603
    ##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
    ## 1          -313     -128      22.5    -161              34             0
    ## 2          -311     -128      22.5    -161              34             0
    ## 3          -305     -128      22.5    -161              34             0
    ## 4          -310     -128      22.1    -161              34             0
    ## 5          -302     -128      22.1    -161              34             0
    ## 6          -312     -128      22.0    -161              34             0
    ##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
    ## 1            0               0            0             0                0
    ## 2            0               0            0             0                0
    ## 3            0               0            0             0                0
    ## 4            0               0            0             0                0
    ## 5            0               0            0             0                0
    ## 6            0               0            0             0                0
    ##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
    ## 1             0           0              0           0        0.00
    ## 2             0           0              0           0        0.02
    ## 3             0           0              0           0        0.02
    ## 4             0           0              0           0        0.02
    ## 5             0           0              0           0        0.00
    ## 6             0           0              0           0        0.02
    ##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
    ## 1        0.00       -0.02        -288         109        -123         -368
    ## 2       -0.02       -0.02        -290         110        -125         -369
    ## 3       -0.02       -0.02        -289         110        -126         -368
    ## 4       -0.03        0.02        -289         111        -123         -372
    ## 5       -0.03        0.00        -289         111        -123         -374
    ## 6       -0.03        0.00        -289         111        -122         -369
    ##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
    ## 1          337          516                 0                  0
    ## 2          337          513                 0                  0
    ## 3          344          513                 0                  0
    ## 4          344          512                 0                  0
    ## 5          337          506                 0                  0
    ## 6          342          513                 0                  0
    ##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
    ## 1                                                                      0
    ## 2                                                                      0
    ## 3                                                                      0
    ## 4                                                                      0
    ## 5                                                                      0
    ## 6                                                                      0
    ##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
    ## 1            0             0           0            0             0
    ## 2            0             0           0            0             0
    ## 3            0             0           0            0             0
    ## 4            0             0           0            0             0
    ## 5            0             0           0            0             0
    ## 6            0             0           0            0             0
    ##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
    ## 1           0                  0                   0                 0
    ## 2           0                  0                   0                 0
    ## 3           0                  0                   0                 0
    ## 4           0                  0                   0                 0
    ## 5           0                  0                   0                 0
    ## 6           0                  0                   0                 0
    ##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
    ## 1      13.05217      -70.49400    -84.87394                      0
    ## 2      13.13074      -70.63751    -84.71065                      0
    ## 3      12.85075      -70.27812    -85.14078                      0
    ## 4      13.43120      -70.39379    -84.87363                      0
    ## 5      13.37872      -70.42856    -84.85306                      0
    ## 6      13.38246      -70.81759    -84.46500                      0
    ##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
    ## 1                       0                                             
    ## 2                       0                                             
    ## 3                       0                                             
    ## 4                       0                                             
    ## 5                       0                                             
    ## 6                       0                                             
    ##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
    ## 1                                                               0
    ## 2                                                               0
    ## 3                                                               0
    ## 4                                                               0
    ## 5                                                               0
    ## 6                                                               0
    ##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
    ## 1                  0                0                 0                  0
    ## 2                  0                0                 0                  0
    ## 3                  0                0                 0                  0
    ## 4                  0                0                 0                  0
    ## 5                  0                0                 0                  0
    ## 6                  0                0                 0                  0
    ##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
    ## 1                0                       0                        0
    ## 2                0                       0                        0
    ## 3                0                       0                        0
    ## 4                0                       0                        0
    ## 5                0                       0                        0
    ## 6                0                       0                        0
    ##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
    ## 1                                          37                  0
    ## 2                                          37                  0
    ## 3                                          37                  0
    ## 4                                          37                  0
    ## 5                                          37                  0
    ## 6                                          37                  0
    ##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
    ## 1                 0                    0                 0
    ## 2                 0                    0                 0
    ## 3                 0                    0                 0
    ## 4                 0                    0                 0
    ## 5                 0                    0                 0
    ## 6                 0                    0                 0
    ##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
    ## 1                  0                     0                  0
    ## 2                  0                     0                  0
    ## 3                  0                     0                  0
    ## 4                  0                     0                  0
    ## 5                  0                     0                  0
    ## 6                  0                     0                  0
    ##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
    ## 1                0                   0                0                0
    ## 2                0                   0                0                0
    ## 3                0                   0                0                0
    ## 4                0                   0                0                0
    ## 5                0                   0                0                0
    ## 6                0                   0                0                0
    ##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
    ## 1            -0.02             0.00             -234               47
    ## 2            -0.02             0.00             -233               47
    ## 3            -0.02             0.00             -232               46
    ## 4            -0.02            -0.02             -232               48
    ## 5            -0.02             0.00             -233               48
    ## 6            -0.02             0.00             -234               48
    ##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
    ## 1             -271              -559               293               -65
    ## 2             -269              -555               296               -64
    ## 3             -270              -561               298               -63
    ## 4             -269              -552               303               -60
    ## 5             -270              -554               292               -68
    ## 6             -269              -558               294               -66
    ##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
    ## 1         28.4         -63.9        -153                     0
    ## 2         28.3         -63.9        -153                     0
    ## 3         28.3         -63.9        -152                     0
    ## 4         28.1         -63.9        -152                     0
    ## 5         28.0         -63.9        -152                     0
    ## 6         27.9         -63.9        -152                     0
    ##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
    ## 1                      0                                           
    ## 2                      0                                           
    ## 3                      0                                           
    ## 4                      0                                           
    ## 5                      0                                           
    ## 6                      0                                           
    ##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
    ## 1                                                            0
    ## 2                                                            0
    ## 3                                                            0
    ## 4                                                            0
    ## 5                                                            0
    ## 6                                                            0
    ##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
    ## 1                 0               0                0                 0
    ## 2                 0               0                0                 0
    ## 3                 0               0                0                 0
    ## 4                 0               0                0                 0
    ## 5                 0               0                0                 0
    ## 6                 0               0                0                 0
    ##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
    ## 1               0                      0                       0
    ## 2               0                      0                       0
    ## 3               0                      0                       0
    ## 4               0                      0                       0
    ## 5               0                      0                       0
    ## 6               0                      0                       0
    ##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
    ## 1                                        36                 0
    ## 2                                        36                 0
    ## 3                                        36                 0
    ## 4                                        36                 0
    ## 5                                        36                 0
    ## 6                                        36                 0
    ##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
    ## 1                0                   0                0                 0
    ## 2                0                   0                0                 0
    ## 3                0                   0                0                 0
    ## 4                0                   0                0                 0
    ## 5                0                   0                0                 0
    ## 6                0                   0                0                 0
    ##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
    ## 1                    0                 0               0
    ## 2                    0                 0               0
    ## 3                    0                 0               0
    ## 4                    0                 0               0
    ## 5                    0                 0               0
    ## 6                    0                 0               0
    ##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
    ## 1                  0               0            0.03            0.00
    ## 2                  0               0            0.02            0.00
    ## 3                  0               0            0.03           -0.02
    ## 4                  0               0            0.02           -0.02
    ## 5                  0               0            0.02            0.00
    ## 6                  0               0            0.02           -0.02
    ##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
    ## 1           -0.02             192             203            -215
    ## 2           -0.02             192             203            -216
    ## 3            0.00             196             204            -213
    ## 4            0.00             189             206            -214
    ## 5           -0.02             189             206            -214
    ## 6           -0.03             193             203            -215
    ##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
    ## 1              -17              654              476      A
    ## 2              -18              661              473      A
    ## 3              -18              658              469      A
    ## 4              -16              658              469      A
    ## 5              -17              655              473      A
    ## 6               -9              660              478      A

    dim(training) 

    ## [1] 19622   160

    str(training)

    ## 'data.frame':    19622 obs. of  160 variables:
    ##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_picth_belt     : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_belt      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ skewness_roll_belt.1    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_belt           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_picth_belt          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_yaw_belt            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_roll_belt           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_pitch_belt          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_yaw_belt            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_roll_belt     : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_pitch_belt    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ var_total_accel_belt    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_roll_belt           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_roll_belt        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_roll_belt           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_pitch_belt          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_pitch_belt       : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_pitch_belt          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_yaw_belt            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_yaw_belt         : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_yaw_belt            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ var_accel_arm           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_roll_arm            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_roll_arm         : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_roll_arm            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_pitch_arm           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_pitch_arm        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_pitch_arm           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ avg_yaw_arm             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ stddev_yaw_arm          : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ var_yaw_arm             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ kurtosis_roll_arm       : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_picth_arm      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_arm        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_roll_arm            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_picth_arm           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_yaw_arm             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_roll_arm            : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_pitch_arm           : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_yaw_arm             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_roll_arm      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_pitch_arm     : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_yaw_arm       : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ kurtosis_roll_dumbbell  : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_picth_dumbbell : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_dumbbell       : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_picth_dumbbell      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ max_yaw_dumbbell        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_roll_dumbbell       : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_pitch_dumbbell      : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ min_yaw_dumbbell        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ amplitude_roll_dumbbell : num  0 0 0 0 0 0 0 0 0 0 ...
    ##   [list output truncated]
