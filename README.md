# find_fine_wine


Project Description:
    
    Wine is a staple in social events, there are 2 general categories of wine though their qualities and methods of production can vary immensely. The variation between the qualities identified by conoseiurs of the wine have an impact on the quality, thus the quality of the wine may be predictable.


## Project Goal

Predict wine quality by using clustering techniques to create features for a machine learning model

 ## Initial Thoughts
 
 ## The plan
 
     Acquire data from sql zillow database
         Pull relevant data columns only using a SQL Command
         Convert to DataFrame
         Look at the dataset
         Confirm understaning of all vairbles
         Create a data dictionary
     Prepare data
         Identify nulls/Nan and replace them with the mean or get rid of that column if there are too many nulls/Nans
         Identify columns with duplicate values and get rid of them
         Change column names to make them readable
         Split data between X and Y
         Identify outliers, and get rid of them
     Explore
         Split into Train, Validate, Test
         Start Visualizing Data
         Select most relevant and interesting visulaizations for final report
         Find out which features have the strongest relationship with the target variable
         Create hypothesis
         Create models
     Model
         Choose Evaluation Metric
         Baseline eval metric
         Fit and score model with Train
         Fit and score model with Validate
         Select best model
             Fit and score model with Test
             
## Explore Data



## Develop a Model to predict assessed value

      Use drivers identified in explore to build predictive models of different types
      Evaluate models on train and validate data
      Select the best model based on the highest accuracy
      Evaluate the best model on test data
      
      
      
  ## Data Dictionary

| **Column**          | **Description**                                            |
|---------------------|--------------------------------------------------------    |
| **fixed_acidity**   | Ranges between 4.8 mg/L and 14.2mg/L, and gauges the amount of a set of organic acids, ensures wine doesn't feel flat and soapy, can't be converted to gas so must go through kidneys 
   |
| **volatile_acidity**| Acid that primarily affects smell, and tase of vinegar, lower is better g/mL    
   |
| **citric_acid**     | found in small quantities, citric acid can add freshness’, sourness, and flavor to wines
   |
| **residual_sugar**  | the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet if over 35 g/L         
   |
| **chlorides**       |  the amount of salt in the wine    
   |
| **free_SO2**        | Related to pH. Determines how much SO2 is available. (Increases shelf-life, decreases palatability)          |
| **total_SO2**       | Summation of free and bound SO2. (Limited to 350ppm: 0-150, low-processed, 150+ highly processed)
   |  
| **density**         | the density of water is close to that of water depending on the percent alcohol and sugar content
   |
| **pH**              | describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
   |
| **sulphates**       | Added to stop fermentation, which acts as an antimicrobial and antioxidant
   |
| **alcohol**         | the percent alcohol content of the wine
   |
| **quality**         | rating from 1 to 10
   |
| **red**             | color and category of wine, 1
   |
| **white**           | color and category of wine, 0
















## Steps to Reproduce

    1. Clone this repo
    2. Acquire the data from data.world
    3. Put the data in the file containing the cloned repo
    4. Run notebook
      
     
## Takeaways and Conclusions

      
      
## Recommendations


      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      