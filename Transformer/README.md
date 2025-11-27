# Molecular Optimization by Capturing Chemist's Intuition using Deep Neural Networks
https://pubmed.ncbi.nlm.nih.gov/33743817/
---

## ✨ My work
We have implemented a Transformer-based architecture to realize the intuition-capturing and molecular generation phases. Our work includes:

* **Data Pipeline:** Extracting, cleaning, and filtering molecular data from public datasets to create a training set of expert-favored molecular transformations.
* **Core Model Development:** Building the fundamental Transformer-based model using **PyTorch**.
* **Architecture Exploration:** Investigating different approaches to integrate additional requirements (constraints/objectives) into the generation process to further guide model prediction.
---

## 🧠 Model Architecture

### 1. The Basic Transformer Structure

The foundation of our implementation is a sequence-to-sequence Transformer architecture, designed to learn the probability of making 'good moves' based on a starting molecular scaffold.

![Molecular Optimization Workflow](transformer.png)

### 2. Guided Generation Mechanisms

We explore different mechanisms to inject **external guidance** (e.g., specific property constraints, target features) into the generation process, moving beyond simple next-token prediction.

#### 2.1 The Generator Module

This module is responsible for outputting the optimized molecular structure, informed by both the learned chemical context and the specified requirements.

![Generator](Generator.png)

#### 2.2 Decoder and Multi-Head Attention

The **Decoder** processes the latent representation and the generated tokens, while the **Multi-Head Attention** mechanism is crucial for allowing the model to focus on different parts of the input/intermediate sequences simultaneously, capturing complex structure-activity relationships.

* **Decoder Structure:**
    ![Decoder](Decoder.png)
* **Multi-Head Attention Mechanism:**
    ![Multihead attention](attention.png)

---
