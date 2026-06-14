 Emergency Triage Dataset Analysis

# Tutorial 1

## Project Description
This notebook performs initial data loading, exploration, and cleaning on an 'Emergency Triage Dataset'. 
The goal is to prepare the data for further analysis, focusing on standardizing key features like 'Gender'.

## Setup
This section covers the necessary environment setup, including Python version and required libraries.

## Data Loading
The dataset `EmergencyTriageDataset_Reduced_Dirty.csv` is loaded from Google Drive into a pandas DataFrame.

## Data Cleaning - Gender Column
This section specifically addresses the cleaning and standardization of the 'Gender' column,
mapping various string representations to numerical values (0 for female, 1 for male) and ensuring data integrity.

# Tutorial 2

Contributors - Kaylah Leigertwood-Ollivierre, Mya Symister, Sariana Ramoutar, Sekou Ruddock, Tyler Baksh, Zhanna McDonald

## Respiratory Rate Data Cleaning 

This document outlines the data cleaning steps and justifications for the 'RR' (Respiratory Rate) column in the `EmergencyTriageDataset_Reduced_Dirty.csv` dataset.

## Cleaning Decisions for 'RR' Column

### 1. Data Type Conversion

*   **Action:** The 'RR' column was converted to a numeric data type using `pd.to_numeric` with `errors='coerce'`. This handles any non-numeric entries by converting them to `NaN` (Not a Number).
*   **Justification:** While the column appeared to contain numerical values, this step ensures that all entries are indeed treated as numbers, allowing for proper mathematical operations and statistical analysis. We did not need to perform complex conversions (e.g., from categorical to numerical) as the data was already in a numerical-like format.

### 2. Handling Missing Values (Initial)

*   **Observation:** After the initial data load and type conversion, there were no pre-existing missing values (NaNs) in the 'RR' column that needed imputation.
*   **Decision:** No imputation was performed for initially missing values, as none were identified.

### 3. Range Filtering

*   **Action:** Values outside the clinically justified range of 5 to 60 breaths per minute (`VALID_MIN = 5`, `VALID_MAX = 60`) were identified and replaced with `NaN`.
*   **Justification:** Respiratory rates outside this range are considered physiologically implausible or indicative of severe medical conditions that may represent data entry errors or extreme outliers. Replacing them with `NaN` ensures that subsequent analysis focuses on plausible physiological values.

### 4. Imputation of Out-of-Range Values

*   **Action:** The `NaN` values resulting from the range filtering step were imputed (filled) using the `median` of the 'RR' column.
*   **Justification:** The distribution of the 'RR' column, as observed in the histogram and box plot, showed a **right-skewed** pattern. In skewed distributions, the median is a more robust measure of central tendency than the mean, as it is less affected by outliers. Therefore, using the median for imputation helps to maintain the overall distribution characteristics of the data while filling in the cleaned missing values.

These steps ensure that the 'RR' data is clean, clinically plausible, and suitable for further analysis.

# Tutorial 3
## Introduction
This notebook is part of an assignment focused on visualising data from the Mercer General Emergency dataset. The objective is to demonstrate data cleaning, exploration, and visualisation techniques to understand patient characteristics and clinical trends within an emergency department setting.

## Goal
The primary goal of this assignment is to leverage Python's data science libraries (Pandas, Matplotlib, Seaborn) to process, analyze, and visualise key health metrics from the provided dataset. This helps in identifying patterns, assessing clinical ranges, and presenting insights in an easily understandable format.

## Visualisations Explained

### 1. Age Vs Respiratory Rate (RR)
*   **Purpose**: This scatterplot investigates the relationship between patient age and respiratory rate, specifically addressing whether patients aged 18 and older fall within the adult normal respiratory rate range (12-20 breaths/min).
*   **Details**: The plot displays individual patient data points for age and respiratory rate. A regression line is included to show the overall trend of respiratory rate with age. A shaded green band highlights the normal adult respiratory rate range (12-20 breaths/min), and an orange band indicates the typical adult age range (18-64 years). Deviations from these normal ranges can indicate potential pathologies.

### 2. Temperature Categories
*   **Purpose**: This bar chart visualizes the proportion of patients presenting with hypothermia, normal temperature, or a fever.
*   **Details**: After cleaning and converting temperature data to Celsius, patients are classified into three categories: 'Hypothermic' (< 36.0 °C), 'Normal' (36.0–37.5 °C), and 'Febrile' (≥ 37.6 °C). The bar chart then shows the percentage of patients falling into each of these temperature categories, providing a clear overview of temperature distribution within the dataset.
