# Notebook workflow

This notebook set is intentionally structured around the actual project logic, with reusable code kept in the `src` package and the notebooks used as a presentation layer.

1. `01_data_loading_and_preprocessing.ipynb` — loading, feature creation, and initial filtering
2. `02_exploration_and_feature_analysis.ipynb` — exploratory analysis and feature insight
3. `03_modeling_and_validation.ipynb` — time-aware model evaluation and comparison
4. `04_jst_analysis_and_summary.ipynb` — institutional analysis and final narrative

The notebooks are designed to read like a coherent story, while the heavy logic remains in the Python modules under `src`.
