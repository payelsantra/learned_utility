# Learned Utility for Query Term Weighting

This repository implements **supervised term utility learning for information retrieval**.  
The framework learns **term importance (utility scores)** using different learning strategies and then integrates these utilities into a retrieval system.

Supported learning methods:

- **InfoNCE (contrastive learning)**
- **Ridge Regression**
- **Linear Regression**

The learned utilities are then used during retrieval to improve ranking performance.

---


# Dependencies

## Java

- Java 11+
- Maven
- Apache Lucene

Build the Java project.


---

# Dataset

Experiments are performed on **HotpotQA**.

Place dataset files inside: 

# Pipeline Overview

The system consists of four stages:

1. Index construction
2. Feature generation
3. Utility learning (InfoNCE / Ridge / Linear)
4. Retrieval and evaluation


# Step 1: Build the Index

# Step 2: Generate Sparse Term Vectors

This step generates term-level features required for training utility models.

---

# Step 3: Train Utility Models

Utility scores can be learned using three different approaches.

---

## InfoNCE Training

InfoNCE learns term utilities through **contrastive learning**.

Run:



