# Find Fine Wine

How can you dine without some fine wine?

### Project Description

Wine is a staple in social events, there are 2 general categories of wine though their qualities and methods of production can vary immensely. The variation between the qualities identified by connoisseurs of the wine have an impact on the quality, thus the quality of the wine may be predictable.

### Project Goal

* Discover drivers of wine quality
* Use drivers to develop clusters or groupings
* Use drivers and clusters to develop a machine learning model to predict wine quality

### Initial Thoughts

Our initial hypothesis is that drivers of wine quality will be difficult to discover. Otherwise anyone can be a sommelier.

## The plan

- Acquire data from data.world
  - Combine red and white wine datasets
- Prepare data
  - Identify and handle Nulls, duplicate data, and outliers
  - Change column names to make them readable
- Explore data in search of drivers of wine quality
  - Split into Train, Validate, Test
  - Start Visualizing Data
  - Select most relevant and interesting visualizations for final report
  - Find out which features have the strongest relationship with the target variable
  - Answer the following initial questions
    - Is there a correlation between ??? and quality?
- Develop a model to predict wine quality
  - Use drivers identified in explore
  - Choose Evaluation Metric
  - Evaluate model with Train and Validate
  - Select best model
    - Evaluate best model with Test
- Draw conclusions

## Data Dictionary

| **Feature**          | Type    | **Description**                                                                                    |
| :------------------------- | :------ | :------------------------------------------------------------------------------------------------------- |
| **fixed_acidity**    | g/L     | Amount of tartaric acids, ensures wine doesn't feel flat and soapy                                      |
| **volatile_acidity** | g/L     | Amount of acetic aci, primarily affects smell and tastes of vinegar, lower is better                   |
| **citric_acid**      | g/L     | Amount of citric acid, can add freshness, sourness, and flavor to wines                                  |
| **residual_sugar**   | g/L     | Amount of sugar remaining after fermentation stops                                                       |
| **chlorides**        | g/L     | Amount of sodium in the wine (Affects color, clarity, flavor, aroma)                                    |
| **free_SO2**         | mg/L    | Amount of Sulfur Dioxide is available, related to pH (Increases shelf-life, decreases palatability)      |
| **total_SO2**        | mg/L    | Sum of free and bound Sulfur Dioxide. (Limited to 350ppm: 0-150, low-processed, 150+ highly processed)   |
| **density**          | g/L     | Density of water is close to that of water depending on the percent alcohol and sugar content            |
| **pH**               | Numeric | How acidic or basic from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale |
| **sulphates**        | g/L     | Added to stop fermentation, which acts as an antimicrobial and antioxidant                               |
| **alcohol**          | vol%    | Percent alcohol content of the wine                                                                      |
| **quality (target)** | Numeric | Median of at least 3 evaluations made by wine experts from 0 (very bad) to 10 (very excellent)           |
| **red**              | Boolean | Color and category of wine, 1 for red, 0 for white                                                       |

## Steps to Reproduce

1) Clone this repo
2) Run notebook

## Takeaways and Conclusions

## Recommendations and Next Steps

- 
- Given more time we would look at evaluation metrics for clustering to hopefully find better groupings of data using more than three features
- Just go to the store and get some $10 wine since it will probably taste just fine
