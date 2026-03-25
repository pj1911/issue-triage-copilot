# Project Spec: Issue Triage Copilot

## Problem
Open-source maintainers receive many issues. Triage is slow and inconsistent.
This project builds a system that can assist with issue triage and draft a response.

## Input
A GitHub issue:
- title
- body
- optionally metadata

## Outputs
The system should produce:
1. predicted labels
2. predicted priority
3. predicted severity
4. similar historical issues
5. relevant documentation snippets
6. a draft maintainer response with cited evidence
7. abstain / human-review recommendation when confidence is low

## Users
- open-source maintainers
- engineering teams handling issue intake
- ML recruiters evaluating applied AI/ML project depth

## Scope for V1
- offline batch inference on historical issues
- retrieval over docs + historical issues
- one demo UI
- one local deployment path

## Explicit Non-Goals
- full GitHub bot automation on day 1
- multi-repo production service
- training a language model from scratch

## Evaluation Plan
### Classification
- micro/macro F1 for labels
- weighted F1 or accuracy for priority/severity

### Retrieval
- Recall@5
- Recall@10
- MRR or nDCG

### Generation
- helpfulness
- correctness
- hallucination rate
- citation support
- abstention quality

## Risks
- label imbalance
- noisy issue text
- inconsistent priority/severity annotations
- retrieval quality bottleneck
- LLM generating unsupported claims

## Success Criteria
A recruiter should be able to see:
- real data pipeline
- baselines
- model improvement
- retrieval evaluation
- fine-tuned LLM usage
- end-to-end demo

This spec is important because it prevents the project from drifting into "random LLM app."
