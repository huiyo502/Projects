# Molecular Optimization by Capturing Chemist's Intuition using Deep Neural Networks
https://pubmed.ncbi.nlm.nih.gov/33743817/
---

## ✨ My work (Due to the confidential issue, I cannot share the code)
We have implemented a Transformer-based architecture to realize the intuition-capturing and molecular generation phases. Our work includes:

* **Data Pipeline:** Extracting, cleaning, and filtering molecular data from public datasets to create a training set of expert-favored molecular transformations.
* **Core Model Development:** Building the fundamental Transformer-based model using **PyTorch**.
* **Architecture Exploration:** Investigating different approaches to integrate additional requirements (constraints/objectives) into the generation process to further guide model prediction.
---

## 🧠 Model Architecture

### 1. The Basic Transformer Structure

The foundation of our implementation is a sequence-to-sequence Transformer architecture, designed to create new molecular based on chemistist requirements.

![Molecular Optimization Workflow](transformer.png)

### 2. Guided Generation Mechanisms

We explore different mechanisms to inject **external guidance** (e.g., molecular weight, logP, etc) into the generation process for generating favorible molecule and also predict the property for this molecule. The three different approached are shown in below:

#### 2.1 The Generator

![Generator](Generator.png)

#### 2.2 Decoder 

![Decoder](Decoder.png)

#### 2.3 Attention 
![Multihead attention](attention.png)

---
