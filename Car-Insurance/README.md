### Project Overview

 Company wants to know the reason why claim was not made. Doing so would allow insurance company to improve there policy for giving loan to the customer.
The dataset has details of 10302 Insurance claim with the 25 features.


### Learnings from the project

  In this project I dealwith various feature such as age, occupation etc. based on that I  get back to the final conculsion. After completing this Project I learn to deal with Imbalanced Data.


### Approach taken to solve the problem

 This is imbalanced dataset . Here 0 - Claim was not made, 1 - Claim made. After completing this project, you will have the better understanding of how to build deal with imbalanced dataset. In this project, I  applied following concepts.
Train-test split
Standard scaler
Logistic Regression
SMOTE
feature scaling


### Challenges faced

 As I got an accuracy of 74% without using any technique for Imbalaced data. One might think that it is a good score but even if the model always predicts 0, you will still get 74% accuracy(The target value distribution is 74% 0s and 26% 1s)
So if you applied model on this dataset it will give us a bad prediction. To overcome I need to use the technic Oversampling or Undersampling technic. Oversampling in data analysis is techniques used to adjust the class distribution of an imbalanced dataset. In this task, I apply the SMOTE to adjust the class distribution of the data set.


