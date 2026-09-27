# ShelfSight package contents

This archive contains the complete portable ShelfSight project source and runtime artifacts.

## Included

- Streamlit app and Python package source
- Requirements, project configuration, example planogram, and Streamlit settings
- Training and evaluation scripts
- Unit tests and project documentation
- Synthetic seven-category training crops and contact sheet
- Trained MobileNetV2 SKU classifier, class labels, model metadata, and evaluation metrics
- Three sample shelf images
- Cached MobileNetV2 ImageNet weights used by the offline synthetic-demo trainer
- `reports/ShelfSight_Project_Report.docx`, prepared for Mohammed Aariff L

## Not included

- `.venv`: machine-specific installed Python environment; recreate it with the commands in `README.md`
- Python bytecode, `__pycache__`, `.pytest_cache`, and temporary report-render previews

The trained model and its data are included. The synthetic model's small 7-image evaluation is a pipeline demonstration and is not a real-shelf accuracy benchmark.
