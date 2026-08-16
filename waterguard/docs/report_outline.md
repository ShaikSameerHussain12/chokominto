# Reconstructed Academic Report Outline

**Project Title:** Detecting Anomalies in Water Usage Patterns Using Data Mining Techniques  
**Subtitle:** Machine Learning Based Water Consumption Fraud Detection and Decision Support System  

---

## Front Matter

* **Title Page**
* **Certificate of Authenticity**
* **Declaration**
* **Acknowledgement**
* **Vision and Mission Statements**
* **Table of Contents**
* **List of Figures**
* **List of Tables**
* **List of Abbreviations**
  * *NTL:* Non-Technical Loss
  * *SVM:* Support Vector Machine
  * *KNN:* K-Nearest Neighbors
  * *CRISP-DM:* Cross-Industry Standard Process for Data Mining
  * *YWC:* Yarmouk Water Company

---

## Chapter 1: Introduction
* **1.1 Background:** Financial implications of water losses in distribution networks. Technical vs. Non-Technical Loss (NTL).
* **1.2 Problem Statement:** Revenue leakages at water supply firms (e.g., Yarmouk Water Company) due to meter tampering, billing bypasses, and unauthorized usage.
* **1.3 Reconstructed Project Objectives:** Operationalize an anomaly detection classification workflow (SVM and KNN) wrapped in an inspectable Django dashboard.
* **1.4 Decision Support Scope:** Explicitly clarifying that ML predictions act as resource-prioritization indicators rather than legally binding guilt declarations.

## Chapter 2: Literature Review
* **2.1 Water Fraud Detection Techniques:** Historical review of rule-based engines, statistical modeling, and modern data-mining applications.
* **2.2 Support Vector Machines in NTL:** Comparative audit of hyperplane margin separators in highly imbalanced domains.
* **2.3 Nearest Neighbor Classifiers:** Distance-based modeling, voting schemas, and feature-scale sensitivity.

## Chapter 3: Methodology
* **3.1 CRISP-DM Framework:** Mapping project phases: Business Understanding $\rightarrow$ Data Understanding $\rightarrow$ Data Preparation $\rightarrow$ Modeling $\rightarrow$ Evaluation $\rightarrow$ Deployment.
* **3.2 Dataset Context:** Yarmouk Water Company (YWC) Qasabat Irbid statistics (historical baseline context of 1.5M records / 90k customers).
* **3.3 Supervised Modeling Strategy:** Class balancing configurations, train/test split rules, standard scaling to prevent data leakages.

## Chapter 4: System Analysis
* **4.1 Requirement Specifications:** Hardware boundaries (Pentium baseline contexts vs. modern 8GB server targets) and software specifications (Python, Django, MySQL, scikit-learn, ChartJS).
* **4.2 Ingestion Validation Schema:** Schema formats, validation parameters, and type enforcement on CSV/Excel file uploads.
* **4.3 Use Case Assessments:** Actor boundaries for administrators (upload, train, score, inspect, block) and standard customer users.

## Chapter 5: System Design
* **5.1 Architecture Diagram:** Decoupled flow showing raw data processing $\rightarrow$ features extraction $\rightarrow$ machine learning training & prediction $\rightarrow$ database ORM persistence $\rightarrow$ templates rendering.
* **5.2 Relational Entity-Relationship (ER) Schema:** Structured representations of UserProfile, ConsumptionRecord, DatasetUpload, Prediction, Investigation, Feedback, and BlockedCustomer models.
* **5.3 ML Pipeline Dataflow:** Preprocessing filters $\rightarrow$ rolling window averages $\rightarrow$ stratified data separation $\rightarrow$ classifier evaluation curves.

## Chapter 6: Implementation
* **6.1 Data Preprocessing Module:** Multi-step pipeline implementation details (deduplication, missing reading interpolation, negative resets flagging, extreme outliers trimming).
* **6.2 Feature Engineering Metrics:** Formulas and implementation details of statistical indicators (CV, skewness, kurtosis) and behavioral attributes (deviations from neighborhood/group averages).
* **6.3 Model Implementations:** Configurations of SVM (RBF kernel, balanced weights) and KNN ($K=5$, distance weights) via scikit-learn.

## Chapter 7: Testing
* **7.1 Verification Strategy:** Reviewing unit, integration, and functional test coverage.
* **7.2 Black-Box and White-Box Testing:** Validation of logical thresholds (risk scoring boundaries) and UI view constraints (unauthorized route blocking).
* **7.3 User Acceptance Testing (UAT):** Detailed 22-part checklist tracking operational success of login, data ingestion, pipeline training, scoring, notes updating, and logs exporting.

## Chapter 8: Results and Discussion
* **8.1 Statistical Summary:** Preprocessing counts, duplicate drop rates, and outlier adjustments on the seed dataset.
* **8.2 Model Comparisons:** Dynamically generated evaluation metrics on the test partition (Accuracy, Precision, Recall, F1, training times) for SVM vs. KNN.
* **8.3 Interpretation of Risk Levels:** Performance of the risk scoring thresholds (High, Medium, Low) and feature-based risk indicators.

## Chapter 9: Conclusion and Future Work
* **9.1 Concluding Summary:** Operational effectiveness of prior-inspections prioritization using data mining.
* **9.2 Limitations:** Requirement of historical labeled baselines for initial classifier calibration.
* **9.3 Future Directions:** Integration of smart IoT meters, deep-learning sequential models (LSTMs), and GIS location mapping.

---

## References
* List of referenced research papers on NTL, water consumption anomalies, SVM classifiers, and machine-learning frameworks.

---

## Annexures
* **Annexure A:** System screenshots and dashboards.
* **Annexure B:** Preprocessing and data cleaning scripts.
* **Annexure C:** SVM and KNN model training code.
* **Annexure D:** Consumption trend lines and graph exports.
