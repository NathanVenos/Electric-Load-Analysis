# Programming Assignment: Load Analysis - Nathan Venos' Results

## Index

- [Deliverables](#deliverables)
- [Extensions](#extensions)

## The Deliverables
Your code should be delivered as a zip file in an email to <engineering+jobs@voltus.co>, with source implementing the analysis and generating all other deliverables. Assume that the output of your code will be run in a production environment.  
**The emailed .zip file includes two files with functions: Voltus_Project_Nathan_Functions_Q1_2_Viz.py has functions used for questions 1 and 2 as well as my visualizations, Voltus_Project_Nathan_Functions_Q3.py has functions used for question 3.  
A Jupyter Notebook, Voltus_Project_Nathan.ipynb will generate all deliverables and includes some narrative on my approach as well as the visualizations and some initial EDA while a corresponding Python file, Voltus_Project_Nathan.py will just generate the deliverables for questions 1, 2 and 3**

Your code should:
1. Output some artifact (CSV, database, etc.) answering questions 1, 2, and 3 by `site_id`. If you're unsure of what's acceptable, ping Liz at <ltheurer@voltus.co>.  
**provided as 3 separate .csv files in the deliverables folder**  
2. Be readable on its own.
3. Include your preferred language's version of a `requirements.txt` or `package.json` so that someone else can easily duplicate your environment/dependencies.  
**A requirements.txt file is provided in the root directory**  
4. Your code should ***ideally*** include some visual output for visualizing your results against the dataset.
The requested data artifacts are provided as .csv files.  
**Visualizations can be found at the end of the Jupyter Notebook**
  
## Extensions.

1. Suppose we had access to real-times streaming data for these sites in the same format as this database. How would you implement startup/shutdown detection as an online algorithm?  
**For each new interval of data, I would use the same method used within Voltus_Project_Nathan_Functions_Q1_2_Viz.py to determine if a given time interval represented an operating interval or not. If there is a shift from operation to non-operation or vice versa then I would categorize that as either a stop or start time respectively**
2. How would you estimate the "downtime load" of a site--the amount of energy they use then they are not operating?  
**I would use a method similar to what was employed in my code where the data is aggregated by day-of-week and the median value of each interval is used to generate a typical week of data. From here, I would take the 5th percentile of each day's load to determine what I called near-base-load, and then I would define the minimum near-base-load among the days of the week as the "downtime load"**
