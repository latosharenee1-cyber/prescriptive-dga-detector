# Mapping Guide: Your lecture materials → Prescriptive DGA Detector

This guide shows how the uploaded lecture hands on code lines up with the assignment deliverables.

## What you uploaded

- `hands-on/1_prepare_data.py`, `2_train_ner_model.py`, `3_run_inference.py`
  - Fine tunes a token classification model for threat entities. This is not required for DGA detection but useful for future work to extract IoCs from resolver logs or tickets.
- `hands-on/4_generate_threat_summary.py`
  - Demonstrates a GenAI prompt that converts structured IoCs to a short intelligence summary and action list.
  - We reuse the **prompt structure** pattern to generate a **prescriptive SOC playbook** in `2_analyze_domain.py`.
- `hands-on/archive/adaptive_playbook.py` and `run_simulation.py`
  - Implements an epsilon greedy adaptive playbook. This relates to the **prescriptive** vision. You can treat the final playbook sections from `2_analyze_domain.py` as the **actions** and later add learning to choose among variants in your environment.

## How it maps to the assignment

**Part 1 Code**
- AutoML training and export → `1_train_and_export.py` (H2O AutoML, MOJO output)
- Main app pipeline → `2_analyze_domain.py` (feature compute, predict, SHAP, summary, GenAI playbook)

**XAI → GenAI bridge**
- The structured summary in `2_analyze_domain.py` mirrors the idea in `4_generate_threat_summary.py` and feeds the LLM with precise context tied to the prediction.

**Optional extras for future improvement**
- Use `hands-on/` NER pipeline to parse resolver notes or passive DNS text and feed the extracted entities to the playbook generator.
- Use `archive/adaptive_playbook.py` to turn the static playbook into a learning agent that selects among multiple response variants by context.
