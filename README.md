# Conflicting Evidence in Retrieval-Augmented Generation: 
# A Controlled Study of Faithfulness–Safety Trade-offs in Medical Question Answering

## MSc Advanced Computer Science Dissertation

This repository contains the implementation developed for an MSc Advanced Computer Science dissertation investigating how conflicting evidence affects the behaviour of medical Retrieval-Augmented Generation (RAG) systems.

The study uses a fixed set of **100 MedQA questions** and an approximately **50,000-document PubMed corpus**. Questions and documents are represented using **BAAI/bge-small-en-v1.5** embeddings and retrieved through **ChromaDB** using dense vector search. **Qwen2.5-1.5B-Instruct** is used for the language-processing stages of the pipeline.

The main experiment introduces controlled factual conflicts into the retrieved evidence, evaluates the generated answers at the atomic claim level, and uses the resulting evaluation to determine whether any corrective action is needed.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Problem](#research-problem)
- [Research Questions](#research-questions)
- [Key Contributions](#key-contributions)
- [Experimental Pipeline](#experimental-pipeline)
- [System Architecture](#system-architecture)
- [Experimental Conditions](#experimental-conditions)
- [Dataset and Evidence Corpus](#dataset-and-evidence-corpus)
- [Models and Components](#models-and-components)
- [Retrieval](#retrieval)
- [Conflict Generation and Evidence Cache](#conflict-generation-and-evidence-cache)
- [Claim-Level Evaluation](#claim-level-evaluation)
- [Decision Engine](#decision-engine)
- [Refinement Strategies](#refinement-strategies)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Conflict Experiment](#running-the-conflict-experiment)
- [Results and Output](#results-and-output)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Project Status](#project-status)
- [Citation](#citation)

---

## Project Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with language-model generation. Instead of relying only on the model's internal knowledge, the system first retrieves relevant documents and then uses them as evidence when generating an answer.

This can improve grounding, but it also introduces another source of failure: the retrieved evidence may be incomplete, irrelevant, or factually inconsistent.

This project focuses specifically on the following question:

> **What happens when a medical RAG system is given conflicting factual evidence?**

To study this, the experiment keeps the main RAG pipeline fixed and introduces a controlled factual conflict into the retrieved context. The resulting responses are then analysed in terms of:

- answer-level faithfulness;
- claim-level evidence grounding;
- safety;
- abstention behaviour; and
- the corrective action selected by the Decision Engine.

The same question set, retrieval corpus, embedding model, retrieval depth and generator configuration are used across the main Baseline and Conflicting conditions. This keeps the comparison focused on the effect of the evidence conflict.

---

## Research Problem

A medical RAG system depends on the retrieved evidence to support its generated answers. However, the evidence supplied to the model may sometimes contain conflicting information.

This creates a situation in which the model has to generate an answer from evidence that does not fully agree.

The problem investigated in this project can therefore be described as follows:

1. a medical question is submitted to the RAG system;
2. relevant evidence is retrieved from the PubMed corpus;
3. one additional passage containing a controlled factual conflict is introduced;
4. the model generates an answer from this mixed evidence; and
5. the resulting answer is evaluated to determine whether it remains grounded and safe.

The project also examines whether the evaluation information can be used to decide what the system should do when a problem is detected.

---

## Research Questions

### RQ1

**How does controlled conflicting evidence affect the faithfulness and safety of a medical RAG system?**

This question is addressed by comparing the Baseline and Conflicting conditions using the same set of 100 MedQA questions.

### RQ2

**Can claim-level evaluation of generated answers guide a decision-based corrective refinement response?**

To investigate this, generated answers are decomposed into atomic claims. Each claim is evaluated against the retrieved evidence and, where required, assessed for safety. The resulting information is then used by the Decision Engine to choose between:

- `NO_ACTION`
- `ANSWER_REFINEMENT`
- `EVIDENCE_REFINEMENT`

---

## Key Contributions

### 1. Controlled Conflict Injection

The project implements a controlled way of introducing factual conflicts into retrieved medical evidence.

The conflicting passages are generated and verified **before the main experiment** and stored in an offline evidence cache. The original PubMed corpus is left unchanged.

This means that the same prepared conflict can be reused during the experiment rather than generating a different conflict each time the pipeline is run.

### 2. Claim-Level Evidence Evaluation

Rather than evaluating only the final answer as a whole, the system first breaks each generated answer into atomic factual claims.

Each claim is then compared with the retrieved evidence using three attribution labels:

- `SUPPORTS`
- `CONTRADICTS`
- `NEUTRAL`

These labels are subsequently used to determine whether each claim is:

- `SUPPORTED`
- `CONTRADICTED`
- `UNSUPPORTED`

This provides a more detailed view of where an answer is and is not grounded in the retrieved evidence.

### 3. Joint Faithfulness and Safety Assessment

Faithfulness and safety are evaluated as separate properties.

Claims that require safety assessment are passed through a dedicated Qwen-based safety prompt, which returns:

- `SAFE`
- `UNSAFE`

together with a severity level.

This makes it possible to examine cases where evidence grounding and safety do not necessarily change in the same way.

### 4. Decision-Based Refinement

The evaluation results are passed to a Decision Engine that determines whether the generated response should be left unchanged or further processed.

Depending on the detected failure, the system can select:

- no action;
- answer refinement; or
- evidence refinement.

The purpose is to avoid regenerating every answer unnecessarily and instead apply a corrective action that matches the type of problem identified.

---

## Experimental Pipeline

The overall pipeline used in the experiment is shown below.

```text
                         MEDQA QUESTION
                               │
                               ▼
                    BGE-small Embedding
                               │
                               ▼
                           ChromaDB
                               │
                               ▼
                           Top-5 Docs
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
                BASELINE              CONFLICT
            Original Evidence    Cached Conflict Added
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Qwen2.5-1.5B
                               │
                               ▼
                       Generated Answer
                               │
                               ▼
                        Atomic Claims
                               │
                               ▼
                     Evidence Attribution
                    SUPPORTS / CONTRADICTS /
                           NEUTRAL
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Faithfulness             Safety
                    │                     │
                    └──────────┬──────────┘
                               ▼
                        Decision Engine
                         /      |       \
                        /       |        \
                       ▼        ▼         ▼
                 NO_ACTION   ANSWER_   EVIDENCE_
                             REFINEMENT REFINEMENT
