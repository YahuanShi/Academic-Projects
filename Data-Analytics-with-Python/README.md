# Engineering Data Analytics with Python

## Course Description

This course provides a practical introduction to data analytics for engineers using Python. It covers the entire data analysis pipeline from data cleaning and transformation to visualization, modeling, and interactive web applications. The curriculum also includes foundational Python programming, SQL, APIs, and basic machine learning concepts.

---

## Learning Outcomes

By the end of this course, I was able to:

- Analyze real-world engineering datasets using Python
- Perform data cleaning, transformation, and visualization using Pandas, Matplotlib, Seaborn
- Build statistical and linear models using scikit-learn and statsmodels
- Query structured data and work with database-like structures
- Build and deploy interactive web applications using Streamlit or Flask
- Interpret and document analysis results for technical and non-technical audiences

---

## Course Topics

- Python fundamentals & Pandas
- Data cleaning and merging
- Data visualization (matplotlib, seaborn, plotly)
- Statistical modeling and linear regression
- Machine learning introduction (decision trees, classification)
- SQL queries and API access
- Web frameworks for dashboarding (Flask, Streamlit)

---

## Project Structure

| Task | Description |
|------|-------------|
| General Tasks | [`general_tasks_group_20.ipynb`](./general_tasks_group_20.ipynb) — Contains solutions to six practical sub-tasks (data integration, visualization, modeling, querying). Interactive notebook with corresponding HTML export. |
| Case Study | [`case_study_group_20.ipynb`](./case_study_group_20.ipynb) — Full analysis pipeline for identifying affected customers in the automotive quality issue scenario. Web application implemented in [`case_study_app_group_20.py`](./case_study_app_group_20.py) for interactive results visualization. |
| Data | [`Data/final_data_group_20.csv`](./Data/final_data_group_20.csv) — Primary dataset used for both general tasks and case study analysis. |
| Additional Files | [`Additional_Files/`](./Additional_Files/) — Supplementary resources used during the project. |

---

## General Task Overview

The general task includes six sub-problems that simulate realistic data analysis challenges faced in engineering settings:

1. **Logistics Delay Analysis** (K7 component):  
   - Merge datasets and analyze delay distributions using statistical tests and visualizations  
   - Train a decision tree to predict part defectiveness

2. **Data Storage Concepts**:  
   - Benefits of relational structure, normalization principles

3. **Vehicle Part Tracing (T16)**:  
   - Query dataset to find T16 parts installed in vehicles registered in Adelshofen

4. **Schema Analysis**:  
   - Analyze the data types of the vehicle registration table

5. **Linear Model for Mileage**:  
   - Fit regression model and derive actionable OEM insights

6. **Hit and Run Investigation**:  
   - Trace vehicle body part to registered location using structured queries

📂 See [`general-task`](./general-task) folder for code, plots, and detailed write-up.

---

## Case Study Overview

In this task, I worked as a data analyst for a fictitious automotive company “217” to investigate a **systemic quality failure** caused by faulty steel used between **June 2012 and July 2014**. The task involved:

- Tracing components processed during the affected time window
- Matching them with production batches, customers, and vehicle types
- Identifying affected OEMs (OEM1, OEM2) and car types (Type11, Type12, Type21, Type22)
- Producing a report that informs which customers need to be contacted

💡 This project demonstrated my ability to:
- Work with multiple relational datasets
- Derive insights from temporal and relational data
- Deliver a comprehensive and data-driven business decision pipeline

📂 See [`case-study`](./case-study) folder for data processing, decision logic, and report outputs.

---

## Tools & Libraries Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn, Statsmodels
- SQL-like joins and queries
- Streamlit / Flask (optional for web demo)

---


> Explore each folder to find the full code, notebooks, visualizations, and markdown reports.
