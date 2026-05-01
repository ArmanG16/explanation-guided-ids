# Explanation-Guided Intrusion Detection System (IDS)

An interpretable, rule-based Intrusion Detection System built on top of PyIDS, designed to provide transparent, human-readable security insights while maintaining strong detection performance.

This project explores a lightweight alternative to black-box machine learning IDS models by leveraging Class Association Rule (CAR) mining and Explainable AI (XAI) to generate meaningful, auditable decisions for cybersecurity analysts.
---
## Paper
Full MQP Report:  

“Crack the Code: Where AI Meets Cybersecurity”
---
## Contributors
Cole Gilbert (CS)  
Matthew Cloutier (ECE)  
Arman Gevorgyan (CS) 
Nathan Ewell (ECE)  


### Advisors
Professor Fatemeh Ganji  
Seyedmohammad Nouraniboosjin  
---
## Project Overview

Traditional IDS systems often rely on:
- Signature-based detection, which cannot detect new attacks  
- Machine learning models, which lack interpretability  

This project addresses these limitations by building a system that is:
- Interpretable through rule-based decisions  
- Lightweight and suitable for edge environments  
- Auditable with human-readable explanations  
- Adaptive through data-driven learning  

The system uses PyIDS to generate a compact set of rules that classify network traffic as benign or malicious, while an explanation layer translates these rules into insights analysts can understand.
---
## Key Features
- Rule-based learning using PyIDS  
- Class Association Rule (CAR) mining  
- Lambda optimization (coordinate ascent, grid search)  
- Data preprocessing and feature engineering pipeline  
- Stratified sampling and validation  
- Explainable AI layer with:
  - Support  
  - Confidence  
  - Rule firing frequency  
- Lightweight and efficient design  
---
## System Pipeline
Raw Data → Preprocessing → Train/Validation Split → CAR Mining → Lambda Optimization → IDS Training → Rule Generation → Explanation Layer
The final output is a compact, interpretable ruleset that explains:
- Why traffic was flagged  
- What features triggered the alert  
- How confident the system is  
---
## Datasets
BETH Dataset  
Link: https://www.kaggle.com/datasets/katehighnam/beth-dataset

UNR-IDD Dataset  
Link: https://www.kaggle.com/code/phuonghoainguyen/unr-idd

NSL-KDD Dataset  
Link: https://www.kaggle.com/datasets/hassan06/nslkdd  

UNSW-NB15 Dataset  
Link: https://www.kaggle.com/code/shahtiham/unsw-nb15

These datasets were selected for their realism, diversity of attack types, and compatibility with rule-based learning.
---
## Tech Stack
- Python  
- PyIDS / pyARC  
- scikit-learn  
- pandas / numpy  
- Turing Cluster (WPI HPC)  
---
## Results Summary
- Effective binary classification (benign vs malicious)  
- Compact and interpretable rule sets  
- Low computational overhead  
- Clear explanations for every prediction  

This demonstrates that strong intrusion detection does not require sacrificing interpretability.
---
## Why This Matters
Most IDS systems today operate as black boxes, making it difficult for analysts to:
- Trust alerts  
- Investigate threats  
- Improve detection strategies  

This project shows that rule-based, explainable systems can improve analyst trust, support decision-making, and remain viable in resource-constrained environments.
---
## Future Work
- Multi-class classification  
- Real-time deployment on edge devices  
- Integration with analyst feedback  
- Hybrid approaches combining rule-based and neural methods  
---
## Acknowledgements
We thank:
James Kingsley (Director of HPC, WPI)  
Jackson Henry (Data Scientist I)  
Joshua Solomon (CS) (OX Tresurer)

for their support with the Turing Cluster and scientific computing resources.
“Results in this paper were obtained in part using a high-performance computing system acquired through NSF MRI grant DMS-1337943 to WPI.”
---
## Contact
Arman Gevorgyan  
GitHub: https://github.com/ArmanG16  
LinkedIn: https://linkedin.com/in/arman-gevorgyan16/
---
