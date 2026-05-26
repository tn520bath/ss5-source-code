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
- message-handler and output-persistence design support.

## Repository structure

```text
ss5_source_code/
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
