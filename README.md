# Self Research 2

This repository is dedicated to exploring **Vision Language Models (VLMs)** for **improving the performance of Medical AI**. The project involves reproducing existing research and implementing performance enhancements based on experimental results.

## 🔍 Research Overview
This study focuses on leveraging **CheXzero** and **Xplainer** to analyze Medical AI performance. The research consists of two main phases:

1. **Reproducing Research**  
   - **CheXzero**: A zero-shot learning model for medical image interpretation using self-supervised learning.  
   - **Xplainer**: A model designed to enhance explainability in medical AI through vision-language alignment.  

2. **Performance Improvement**  
   - Implementing modifications and enhancements to improve model accuracy and robustness.  
   - Evaluating performance improvements using benchmark medical datasets.

## 📁 Repository Structure
```yaml
self_research_2
│── CheXzero-base/       # Codebase for reproducing CheXzero
│── Xplainer-base/       # Codebase for reproducing XPLAINER
│── README.md  
```
## 🔬 Experiment Results

| Model     | Dataset       | AUC    | Notes                      |
|-----------|---------------|--------|----------------------------|
| CheXzero  | CheXpert | 0.8432 | Baseline reproduction      |
|  CheXzero | CheXpert | 0.8564 | Xplainer's Prompt Template |
|  CheXzero | CheXpert | 0.8663 | Ours Method                |
| Xplainer  | CheXpert | 0.8246 | Baseline reproduction      |


## How To Use
### 1. Clone the Repository
clone the Repository and navigate to the project directory : 
```shell
git clone https://github.com/meansash/self_research_2.git
cd self_research_2
```
### 2. Navigate to each model
Navigate to the base code of each model you want to experiment with : 
```shell
cd CheXzero-base
# or
cd Xplainer-base
```
### 3. Check README.md

Please check the README.md in each base directory😁

## 📜 References
- CheXzero : [Original Paper](https://www.nature.com/articles/s41551-022-00936-9)
- CheXzero : [Original Code](https://github.com/rajpurkarlab/CheXzero)
- Xplainer : [Original Paper](https://arxiv.org/abs/2303.13391)
- Xplainer : [Original Code](https://github.com/ChantalMP/Xplainer)
