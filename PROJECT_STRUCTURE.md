# MedNeutral Project Structure

```
medneutral/
├── 📁 src/                          # Source code
│   ├── 📁 core/                     # Core processing modules
│   │   ├── stigma_classifier.py     # Traditional rule-based classifier
│   │   ├── bert_enhanced_classifier.py  # BERT-enhanced classifier
│   │   └── hybrid_processor.py      # Hybrid processor combining both approaches
│   ├── 📁 web/                      # Web application
│   │   ├── app.py                   # Flask web application
│   │   └── templates/               # HTML templates
│   │       └── index.html           # Main web interface
│   ├── 📁 analysis/                 # Analysis and notebooks
│   │   └── notebooks/               # Jupyter notebooks
│   │       └── 01_exploration.ipynb # Initial data exploration
│   └── 📁 utils/                    # Utility functions
├── 📁 scripts/                      # Executable scripts
│   ├── demo.py                      # Traditional demo script
│   ├── bert_demo.py                 # BERT-enhanced demo script
│   ├── hybrid_demo.py               # Hybrid processor demo
│   ├── mimic_analysis.py            # MIMIC dataset analysis
│   ├── run_app.py                   # Web app launcher (legacy)
│   ├── run_web_app.py               # Web app launcher (new)
│   └── test_app.py                  # Web app testing
├── 📁 tests/                        # Test files
│   ├── test_system.py               # System functionality tests
│   └── test_app.py                  # Web app tests
├── 📁 data/                         # Data files
│   ├── keywords.json                # Stigmatizing keywords database
│   └── mimic_sample.csv             # MIMIC clinical notes sample
├── 📁 docs/                         # Documentation (future)
├── 📁 examples/                     # Example usage (future)
├── 📄 README.md                     # Main project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 LICENSE                       # Project license
└── 📄 .gitignore                    # Git ignore rules
```

## Usage with New Structure

### **Running Scripts**
```bash
# From medneutral/ directory
python scripts/demo.py                    # Traditional demo
python scripts/bert_demo.py               # BERT demo
python scripts/hybrid_demo.py             # Hybrid demo
python scripts/mimic_analysis.py          # MIMIC analysis
python scripts/run_web_app.py             # Web app
```

### **Importing Modules**
```python
# In any script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.hybrid_processor import HybridProcessor
from core.stigma_classifier import ClinicalNoteProcessor
```

### **Data Access**
```python
# Automatic path resolution
processor = ClinicalNoteProcessor()  # Finds data/keywords.json automatically
```
