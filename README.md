# MedNeutral

**AI-Powered Clinical Note Rewriter - Detecting and Replacing Stigmatizing Language**

MedNeutral is an intelligent system that automatically detects and rewrites stigmatizing language in clinical notes while preserving clinical accuracy and important medical information.

## 🎯 Project Goal

The overall goal is to use AI to rewrite clinical notes by:
- **Detecting** stigmatizing language patterns
- **Removing** biased or judgmental terminology  
- **Replacing** with neutral, professional alternatives
- **Preserving** clinical accuracy and important information

## ✨ Features

### 🔍 Stigma Detection
- **Keyword-based detection** using categorized stigmatizing terms
- **BERT-enhanced detection** with semantic similarity analysis
- **Context-aware analysis** with surrounding text consideration
- **Multi-category classification**: Adamant, Compliance, and Other behavioral terms
- **Confidence scoring** for detected instances

### ✏️ Intelligent Rewriting
- **Neutral replacement mapping** for each stigmatizing term
- **BERT-based alternatives** using semantic similarity for better word selection
- **Context preservation** during text replacement
- **Clinical accuracy maintenance** - no loss of important medical information
- **Batch processing** capabilities for multiple notes

### 🤖 AI-Powered Enhancements
- **BERT embeddings** for semantic word similarity analysis
- **Hybrid processing** combining traditional and BERT approaches
- **Automatic method selection** based on text complexity
- **Multiple alternative suggestions** for each stigmatizing word

### 🌐 Web Interface
- **Modern, responsive UI** for easy interaction
- **Real-time processing** with visual feedback
- **Side-by-side comparison** of original vs rewritten text
- **Detailed change tracking** with context highlighting

## 📊 Stigma Categories

The system detects and categorizes stigmatizing language into three main groups:

### 1. Adamant Language
Terms that suggest patients are overly insistent or demanding:
- `adamant` → `firm`
- `claimed` → `reported` 
- `insisted` → `stated`

### 2. Compliance Issues
Language about patient adherence to treatment:
- `refused` → `declined`
- `noncompliant` → `non-adherent`
- `adherence` → `compliance`

### 3. Other Behavioral Terms
General behavioral and appearance descriptors:
- `aggressive` → `agitated`
- `drug seeking` → `seeking medication`
- `poorly groomed` → `disheveled`

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MedNeutral/medneutral
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**
   ```bash
   python demo.py
   ```

### Web Application

1. **Start the Flask server** (Option 1 - Using launcher script)
   ```bash
   python run_app.py
   ```

2. **Start the Flask server** (Option 2 - Direct method)
   ```bash
   cd app
   python app.py
   ```

3. **Open your browser**
   Navigate to `http://localhost:5001`

4. **Try the interface**
   - Enter a clinical note in the text area
   - Click "Process Note" to see the results
   - Use "Load Sample" to try example text

### BERT-Enhanced Demos

1. **Run BERT demo** to see semantic similarity analysis:
   ```bash
   python bert_demo.py
   ```

2. **Run hybrid demo** to compare different approaches:
   ```bash
   python hybrid_demo.py
   ```

## 📖 Usage Examples

### Python API - Traditional Approach

```python
from models.stigma_classifier import ClinicalNoteProcessor

# Initialize the traditional processor
processor = ClinicalNoteProcessor()

# Process a single note
note = "Patient was adamant about their symptoms and refused to comply with treatment."
result = processor.process_note(note)

print(f"Original: {result['original_text']}")
print(f"Rewritten: {result['rewritten_text']}")
print(f"Changes: {result['num_changes']}")
```

### Python API - BERT-Enhanced Approach

```python
from models.bert_enhanced_classifier import BERTEnhancedProcessor

# Initialize the BERT-enhanced processor
processor = BERTEnhancedProcessor()

# Process a single note with BERT
note = "Patient was adamant about their symptoms and refused to comply with treatment."
result = processor.process_note(note)

print(f"Original: {result['original_text']}")
print(f"BERT Rewritten: {result['rewritten_text']}")
print(f"Changes: {result['num_changes']}")
print(f"BERT Enhanced: {result['bert_enhanced']}")

# Get BERT-based alternatives for a word
alternatives = processor.get_word_alternatives("adamant", top_k=5)
for word, similarity in alternatives:
    print(f"  {word}: {similarity:.3f}")
```

### Python API - Hybrid Approach

```python
from models.hybrid_processor import create_processor

# Initialize the hybrid processor (combines both approaches)
processor = create_processor(use_bert=True)

# Process with different methods
note = "Patient was adamant about their symptoms and refused to comply with treatment."

# Traditional method
result_trad = processor.process_note(note, method="traditional")

# BERT method
result_bert = processor.process_note(note, method="bert")

# Hybrid method (auto-selects best approach)
result_hybrid = processor.process_note(note, method="hybrid")

# Compare all methods
comparison = processor.compare_methods(note)
```

### Batch Processing

```python
# Process multiple notes at once
notes = [
    "Patient was adamant about pain levels.",
    "Patient refused medication and was uncooperative.",
    "Patient was compliant with treatment plan."
]

results = processor.batch_process(notes)
```

## 🏗️ Architecture

### Core Components

1. **StigmaClassifier** (`models/stigma_classifier.py`)
   - Detects stigmatizing language using regex patterns
   - Categorizes findings by type
   - Provides confidence scores

2. **StigmaRewriter** (`models/stigma_classifier.py`)
   - Maps stigmatizing terms to neutral alternatives
   - Preserves text structure and context
   - Tracks all changes made

3. **ClinicalNoteProcessor** (`models/stigma_classifier.py`)
   - Main interface combining detection and rewriting
   - Provides comprehensive analysis results
   - Supports batch processing

4. **Web Interface** (`app/app.py`, `app/templates/index.html`)
   - Flask-based web application
   - Real-time processing and visualization
   - User-friendly interface

### Data Structure

```
medneutral/
├── app/
│   ├── app.py                 # Flask web application
│   └── templates/
│       └── index.html         # Web interface
├── data/
│   ├── keywords.json          # Stigmatizing terms database
│   ├── discharge.csv.gz       # MIMIC-IV discharge notes
│   └── mimic_sample.csv       # Sample data for testing
├── models/
│   └── stigma_classifier.py   # Core detection and rewriting logic
├── notebooks/
│   └── 01_exploration.ipynb   # Data exploration and analysis
├── demo.py                    # Command-line demo script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Adding New Keywords

Edit `data/keywords.json` to add new stigmatizing terms:

```json
{
  "category_name": [
    "new_stigmatizing_term",
    "another_term"
  ]
}
```

### Customizing Replacements

Modify the `replacements` dictionary in `StigmaRewriter.__init__()` to change how terms are rewritten.

## 📈 Performance

- **Detection Accuracy**: High precision with regex-based pattern matching
- **Processing Speed**: Real-time for individual notes, batch processing for large datasets
- **Memory Usage**: Minimal, suitable for production deployment
- **Scalability**: Can handle thousands of notes efficiently

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MIMIC-IV Database** for clinical note examples
- **Medical NLP Community** for research on clinical language bias
- **Healthcare Professionals** for domain expertise and feedback

## 🔮 Future Enhancements

- **Fine-tuned BERT models** for clinical domain-specific language
- **Custom Training** on domain-specific datasets
- **API Integration** with EHR systems
- **Multi-language Support** for international healthcare
- **Audit Trail** for compliance and quality assurance
- **Real-time BERT processing** for live clinical notes
- **Advanced context analysis** using transformer attention mechanisms

---

**MedNeutral** - Making clinical documentation more compassionate and professional, one note at a time. 
