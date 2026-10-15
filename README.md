# ss5-source-code
Source code accompanying ME32031 Technical Report Appendix A — SS5 analytical pipeline.

This repository contains the source code accompanying Appendix A of the ME32031 Technical Report:

**Fab Materials Ltd — Meso Fibre Particle Size Characterisation**  
**Sub-system 5: Analytical Pipeline**

SS5 converts per-fibre image-analysis measurements into batch-level engineering outputs, including:

- synthetic data generation for software validation;
- data ingestion and schema validation;
- descriptor computation and distribution modelling;
- uncertainty quantification using GUM / Monte Carlo methods;
- qualitative risk-index design logic;
- ML-ready predictive modelling framework;
- message handling, plus output-persistence design.

## Repository structure

```text
ss5-source-code/
├── common.py
├── main.py
├── ss3.py
├── ss4.py
├── ss5.py
├── ss5_module1_synthetic_generator/
├── ss5_module2_7_pipeline/
├── ss5_module3_descriptors/
├── ss5_module4_uncertainty/
├── ss5_module6_predictive/
└── validate_*.py
```

## How to run

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python validate_orchestrator.py
```

Each `validate_*.py` script is standalone and should be run from the repository root.

## Scope

Modules 1, 2, 3, 4, 6 and 7 are implemented and validated. Modules 5 (qualitative risk index) and 8 (output persistence) are specified in the report but not implemented in code. The version matching the submitted report is tagged `technical-report-submission-v1`.
