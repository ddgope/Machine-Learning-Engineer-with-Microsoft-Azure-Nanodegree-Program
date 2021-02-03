# Optimizing an ML Pipeline in Azure

## Overview
Optimizing an ML Pipeline in Azure - in this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run , compare and find the best performance model.

## Summary
<p>I have used UCI Bank Marketing dataset, which is related with direct marketing campaigns of a Portuguese baking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).<a href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"> Read More </a></p>.
<p>Below is the pipeline Architecture!</p>
<img src='/images/Pipeline_Architecture.png'>
<p>In this project, I should be using scikit-learn Logistic Regression and tuned the hyperparameters(optimal) using HyperDrive. I have also used AutoML to build and optimize a model on the same dataset, so that I can compare the results of the two methods.
The best performing model was obtained through AutoML - <strong> VotingEnsemble </strong> with accuracy of <b>0.91482</b></p>

## Scikit-learn Pipeline
<ol>
  <li>Setup Training Script
    <ul>
      <li> Import data using <i>TabularDatasetFactory</i> </li>
      <li> Cleaning of data -  handling NULL values, one-hot encoding of categorical features and preprocessing of date </li>
      <li> Splitting of data into train and test data </li>
      <li> Using scikit-learn logistic regression model for classification </li>
    </ul>
  </li><br>
  <li>Create SKLearn Estimator for training the model selected (logistic regression) by passing the training script and later the estimator is passed to the hyperdrive                 configuration</li><br>
  <li> Configuration of Hyperdrive
    <ul>
      <li> Selection of parameter sampler </li>
      <li> Selection of primary metric </li>
      <li> Selection of early termination policy </li>
      <li> Selection of estimator (SKLearn) </li>
      <li> Allocation of resources </li>
      <li> Other configuration details </li>
    </ul>
  </li><br>  
  <li>Save the trained optimized model</li>
</ol>
<p>As specified above, I have used logistic regression model for our binary classification problem and hyperdrive tool to choose the best hyperparameter values from the parameter search space. Under the hood logistic regression uses logistic/sigmoidal function to estimate the probabilities between the dependent/target variable and one or more independent variables(features). In the below image, we can see that which hyperdrive run gave the best result.</p>
<img src = '/images/Project_2_Hyperdrive.JPG'>
</br>
<img src = '/images/Project_1_Hyperdrive.JPG'>

<strong>Parameter Sampler</strong>
<p>The parameter sampler I chose was <i>RandomParameterSampling</i> because it supports both discrete and continuous hyperparameters. It supports early termination of low-performance runs and supports early stopping policies. In random sampling , the hyperparameter (C : smaller values specify stronger regularization, max_iter : maximum number of iterations taken for the solvers to converge) values are randomly selected from the defined search space. </p>

<strong>Early Stopping Policy</strong>
<p> The early stopping policy I chose was <i>BanditPolicy</i> because it is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run.</p>

## AutoML
<ol>
  <li> Import data using <i>TabularDatasetFactory</i></li>
  <li> Cleaning of data -  handling NULL values, one-hot encoding of categorical features and preprocessing of date </li>
  <li> Splitting of data into train and test data </li>
  <li> Configuration of AutoML </li>
  <li> Save the best model generated </li>
</ol>
<img src= '/images/Project_4_Automated_ML.JPG'>
<p> The below snapshots gives the explanation of the best model prediction by highlighting feature importance values and discovering patterns in data at training time. It also shows differnt metrics and their value for model interpretability and explanation. </p>
<img src= '/images/accuracy_table.JPG'>
</br>
<img src= '/images/accuracy_table2.JPG'>
</br>
<img src='/images/accuracy_table3.JPG'>

## Pipeline comparison
<p>Both the approaches - Logistics + Hyperdrive and AutoML follow similar data processing steps and the difference lies in their configuration details. In the first approach ML model is fixed and I have used hyperdrive tool to find optimal hyperparametets while in second approach different models are automatic generated with their own optimal hyperparameter values and the best model is selected. In the below image, we see that the hyperdrive approach took overall <b>16m 54s</b> and the best model had an accuracy of <b>~0.91320</b> and the automl approach took overall <b>34m 37s</b> and the best model had an acccuracy of <b>~0.91482</b>.
</p>
<img src = '/images/Project_3_Hyperdrive_And_ML.JPG'>
<p> It is quite evident that AutoML results in better accurate model but takes time to find out one while the Logistic + Hyperdrive takes lesser time to find out an optimal hyperparameter values for a fixed model. Since I have used the same dataset and preprocessed the data in the same fashion we see that both the approaches generate model whose accuracy is very close.
</p>

## Future work
<ul>
 <li>To check or measure the fairness of the models</li>
 <li>Leverage additional interactive visualizations to assess which groups of users might be negatively impacted by a model and compare multiple models in terms of their              fairness and performance</li>
</ul>

## Proof of cluster clean up
<img src= '/images/Deleting.JPG'>
