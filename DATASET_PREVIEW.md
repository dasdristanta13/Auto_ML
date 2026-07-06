# Data Preview Module – Functional & UI Implementation Specification (Platform-Native)

## Overview

The **Data Preview** module is the primary workspace where users inspect, understand, validate, and prepare their data before launching an AI-driven machine learning workflow. Unlike general-purpose data platforms, this solution is focused entirely on **AutoML orchestration**, so the interface should eliminate unnecessary data engineering features (such as downloading datasets, notebooks, SQL editing, or manual coding) and instead guide users toward building the best possible ML pipeline.

The design philosophy should combine the usability of **Kaggle's dataset explorer**, the profiling depth of **YData Profiling**, and the intelligence of an **AI Data Scientist**.

The module should support datasets from thousands to hundreds of millions of rows through server-side pagination, lazy loading, and asynchronous profiling.

The page should be divided into the following sections:

1. Dataset Header
2. Dataset Overview
3. Interactive Data Preview
4. Column Explorer
5. Dataset Profiling
6. Data Quality Center
7. AI Dataset Assistant
8. Dataset Versions
9. Related Experiments
10. Quick Actions

---

# 1. Dataset Header

The header should provide context about the uploaded dataset.

Display:

* Dataset Name
* Dataset Description
* Business Use Case
* Dataset Version
* Upload Timestamp
* Last Modified
* Owner
* Dataset Status (Processing, Ready, Profiling, Training, Archived)
* Processing Progress
* Dataset Tags
* Problem Type (Classification, Regression, Forecasting, Clustering, etc.)

Buttons:

* Launch AutoML
* Re-run Profiling
* Validate Dataset
* Generate AI Summary
* Upload New Version

The objective is to move users toward model building rather than dataset management.

---

# 2. Dataset Overview

Display KPI cards that summarize the dataset.

Cards should include:

* Total Rows
* Total Columns
* Missing Values
* Duplicate Rows
* Memory Usage
* Numerical Features
* Categorical Features
* Datetime Features
* Text Features
* Target Column
* Target Distribution
* Overall Data Quality Score
* ML Readiness Score

Each KPI should include:

* Current value
* Mini trend chart
* Change compared to previous dataset version (if applicable)
* Tooltip with explanation

Selecting a KPI should navigate to the relevant detailed analysis.

---

# 3. Interactive Data Preview

The main table should resemble Kaggle's dataset preview while remaining optimized for large datasets.

Capabilities:

* Lazy loading
* Infinite scrolling
* Server-side pagination
* Virtual scrolling
* Column resizing
* Column reordering
* Sticky header
* Sticky first column
* Multi-column sorting
* Advanced filtering
* Search across columns
* Column visibility toggle
* Highlight missing values
* Highlight duplicate values
* Highlight anomalous values
* Save preferred table layout

Each column header should display:

* Column Name
* Data Type Icon
* Nullable Indicator
* Unique Indicator
* Filter Button
* Sort Button
* Profile Button

Hovering over a column should display:

* Data Type
* Null Count
* Unique Count
* Missing Percentage
* Example Values
* Mean / Median / Min / Max (where applicable)

Different data types should have specialized rendering:

Numerical

* Right aligned
* Heatmap coloring
* Formatted numbers

Categorical

* Colored badges
* Frequency tooltip

Boolean

* Green/Red pills

Datetime

* Human-readable format

Text

* Truncated preview
* Expand on hover

JSON

* Expandable tree viewer

---

# 4. Column Explorer

Selecting a column should open a dedicated analysis panel.

Display:

General Information

* Column Name
* Business Meaning
* Data Type
* Semantic Type
* Nullable
* Unique Count
* Missing Count
* Cardinality

Distribution

* Histogram
* Density Plot
* Value Distribution
* Frequency Chart

Statistics

* Mean
* Median
* Standard Deviation
* Variance
* Min
* Max
* Quartiles
* Skewness
* Kurtosis

Relationships

* Correlation with Target
* Correlation with Other Features
* Mutual Information
* Feature Interaction Suggestions

ML Insights

* Feature Importance (if experiments exist)
* Leakage Risk
* Recommended Encoding
* Recommended Scaling
* Suggested Transformations
* Suggested Feature Engineering

Example Values

* Top Frequent Values
* Rare Values
* Random Samples

---

# 5. Dataset Profiling

Provide detailed analytical tabs.

### Column Summary

Display:

* Data Types
* Missing %
* Unique %
* Cardinality
* Entropy
* Distribution

### Correlations

Interactive heatmap using:

* Pearson
* Spearman
* Kendall
* Mutual Information

### Missing Values

Visualizations:

* Missing Matrix
* Missing Heatmap
* Missing Correlation

### Outliers

Detection methods:

* IQR
* Isolation Forest
* Z-score
* Local Outlier Factor

Display:

* Number of outliers
* Affected columns
* Example records

### Target Analysis

Display:

* Target Distribution
* Class Imbalance
* Feature vs Target Relationships
* Information Gain
* Chi-Square
* ANOVA

### Drift Analysis

When newer versions are uploaded:

Display:

* Distribution Shift
* Schema Changes
* New Columns
* Removed Columns
* Population Stability Index
* KS Test

---

# 6. Data Quality Center

Provide a comprehensive quality assessment with an overall score.

Evaluate:

* Completeness
* Consistency
* Validity
* Accuracy
* Uniqueness
* Freshness
* Schema Consistency
* ML Readiness

Automatically detect:

* Missing Values
* Duplicate Rows
* Constant Columns
* Near Constant Features
* High Cardinality
* Mixed Data Types
* Invalid Categories
* Impossible Values
* Date Issues
* Potential Target Leakage
* Highly Correlated Features
* Sparse Features
* Memory Optimization Opportunities

Each issue should include:

* Severity
* Affected Columns
* Explanation
* Recommended Fix
* One-click Auto Fix (where safe)

---

# 7. AI Dataset Assistant

A permanent AI assistant panel should remain available throughout the page.

The assistant should reason only from:

* Dataset metadata
* Profiling statistics
* Column summaries
* Previous experiments
* Feature importance
* Validation reports

Suggested prompts:

* Explain this dataset.
* Is this dataset suitable for machine learning?
* Detect target leakage.
* Which features should be removed?
* Suggest preprocessing.
* Suggest feature engineering.
* Recommend a target column.
* Explain missing values.
* Identify data quality issues.
* Estimate model difficulty.
* Which algorithms are likely to perform well?

The assistant should never access or expose the entire raw dataset.

---

# 8. Dataset Versions

Maintain a history of uploaded dataset versions.

Display:

* Version Number
* Upload Date
* Rows
* Columns
* Quality Score
* Profiling Status
* Schema Changes
* Linked Experiments

Allow users to compare versions and understand how the data has evolved.

---

# 9. Related Experiments

Show all experiments executed using this dataset.

Display:

* Experiment Name
* Status
* Model Type
* Best Metric
* Runtime
* Creation Time
* Current Stage

Provide actions to:

* View Results
* Compare Experiments
* Duplicate Configuration

---

# 10. Quick Actions

Provide prominent action cards for the most common platform workflows.

Actions:

* Launch AutoML
* Generate AI Insights
* Run Data Validation
* Re-profile Dataset
* Compare Dataset Versions
* View Experiment History
* Start New Experiment
* View Feature Importance
* Open Model Leaderboard

These actions should initiate guided workflows rather than exposing low-level tooling.

---

# Performance Requirements

* Support datasets with 100M+ rows using server-side pagination.
* Never load the complete dataset into the browser.
* Cache profiling results to avoid recomputation.
* Execute expensive analyses asynchronously with progress indicators.
* Stream results progressively so users can explore while profiling continues.
* Reuse profiling outputs across all platform modules (AutoML, AI Assistant, Feature Engineering, and Experiment Tracking).

---

# User Experience Principles

* Follow the existing white, lavender, and deep-purple design language.
* Use rounded cards with subtle shadows and generous spacing.
* Provide skeleton loaders while profiling or loading data.
* Animate transitions between tabs and panels smoothly.
* Keep advanced analytics collapsible to avoid overwhelming new users.
* Ensure every metric, chart, or issue links to a deeper explanation.
* Design the page as a decision-making workspace that naturally guides users from **understanding their dataset** to **launching an optimized AI-driven machine learning pipeline**, without requiring code or external tools.
