# Optimizing an ML Pipeline in Azure

## Overview
Optimizing an ML Pipeline in Azure - in this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run , compare and find the best performance model.

## Summary
<p>I have used UCI Bank Marketing dataset, which is related with direct marketing campaigns of a Portuguese baking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).<a href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"> Read More </a></p>
<p>Below is the pipeline Architecture!</p>

![Pipeline Architecture](images/Pipeline_Architecture.png?raw=true "Pipeline Architecture")

<p>In this project, I should be using scikit-learn Logistic Regression and tuned the hyperparameters(optimal) using HyperDrive. I have also used AutoML to build and optimize a model on the same dataset, so that I can compare the results of the two methods.
The best performing model was obtained through AutoML - <strong> VotingEnsemble </strong> with accuracy of <b>0.91482</b></p>

Step 1: Set up the train script, create a Tabular Dataset from this set & evaluate it with the custom-code Scikit-learn logistic regression model.

Step 2: Creation of a Jupyter Notebook and use of HyperDrive to find the best hyperparameters for the logistic regression model.

Step 3: Next, load the same dataset in the Notebook with TabularDatasetFactory and use AutoML to find another optimized model.

Step 4: Finally, compare the results of the two methods and write a research report i.e. this Readme file.


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
**Parameter Sampler**

I specified the parameter sampler as such:

```
ps = RandomParameterSampling(
    {
       '--C' : choice(0.03,0.3,3,10,30),
       '--max_iter' : choice(25,50,75,100)
    }
)
```

I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_.

_C_ is the Regularization while _max_iter_ is the maximum number of iterations.

_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or _BayesianParameterSampling_ to explore the hyperparameter space. 

**Early Stopping Policy**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the _BanditPolicy_ which I specified as follows:
```
policy = BanditPolicy(evaluation_interval=3, slack_factor=0.1)
```
_evaluation_interval_: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it.

<p>As specified above, I have used logistic regression model for our binary classification problem and hyperdrive tool to choose the best hyperparameter values from the parameter search space. Under the hood logistic regression uses logistic/sigmoidal function to estimate the probabilities between the dependent/target variable and one or more independent variables(features). In the below image, we can see that which hyperdrive run gave the best result.</p>
<img src = '/images/Project_2_Hyperdrive.JPG'>
</br>
<img src = '/images/Project_1_Hyperdrive.JPG'>

## AutoML
<ol>
  <li> Import data using <i>TabularDatasetFactory</i></li>
  <li> Cleaning of data -  handling NULL values, one-hot encoding of categorical features and preprocessing of date </li>
  <li> Splitting of data into train and test data </li>
  <li> Configuration of AutoML </li>
  <li> Save the best model generated </li>
</ol>
I defined the following configuration for the AutoML run:

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=final_data,
    label_column_name='y',
    compute_target = comp_trget,
    max_concurrent_iterations = 3,
    enable_early_stopping = True,
    enable_onnx_compatible_models=True,
    n_cross_validations=5)
```
This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 15 minutes.

_task='classification'_

This defines the experiment type which in this case is classification.

_primary_metric='accuracy'_

I chose accuracy as the primary metric.

_enable_onnx_compatible_models=True_

I chose to enable enforcing the ONNX-compatible models. Open Neural Network Exchange (ONNX) is an open standard created from Microsoft and a community of partners for representing machine learning models. More info [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx).

_n_cross_validations=5_

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 5 folds for cross-validation; thus the metrics are calculated with the average of the 5 validation metrics.
</br>
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
* Our data is **highly imbalanced**:

![Highly imbalanced data](images/Imbalanced_data_plot.png?raw=true "Highly imbalanced data")

Class imbalance is a very common issue in classification problems in machine learning. Imbalanced data negatively impact the model's accuracy because it is easy for the model to be very accurate just by predicting the majority class, while the accuracy for the minority class can fail miserably. This means that taking into account a simple metric like **accuracy** in order to judge how good our model is can be misleading.

There are many ways to deal with imbalanced data. These include using: 
1. A different metric; for example, AUC_weighted which is more fit for imbalanced data
2. A different algorithm
3. Random Under-Sampling of majority class 
4. Random Over-Sampling of minority class
5. The [imbalanced-learn package](https://imbalanced-learn.readthedocs.io/en/stable/)

There are many other methods as well, but I will not get into much details here as it is out of scope. 

Concluding, the high data imbalance is something that can be handled in a future execution, leading to an obvious improvement of the model.

* Another factor that I would improve is **n_cross_validations**. As cross-validation is the process of taking many subsets of the full training data and training a model on each subset, the higher the number of cross validations is, the higher the accuracy achieved is. However, a high number also raises computation time (a.k.a. training time) thus costs so there must be a balance between the five factors.

    _Note_: In case I would be able to improve ```n_cross_validations```, I would also have to increase ```experiment_timeout_minutes``` as the current setting of 30 minutes would not be enough. 

***

## Proof of cluster clean up
<img src= '/images/Deleting.JPG'>
