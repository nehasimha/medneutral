# MedNeutral User Guide

## 🚀 How to Use MedNeutral

This guide shows you how to use the MedNeutral system to detect and rewrite stigmatizing language in clinical notes, including analysis of real MIMIC data.

## 📋 Prerequisites

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start Options

### **Option 1: Web Interface (Recommended for Interactive Use)**

1. **Start the web application:**
   ```bash
   python scripts/run_web_app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5001
   ```

3. **Use the web interface to:**
   - Paste clinical notes for analysis
   - See real-time stigma detection
   - View rewritten text with changes highlighted
   - Get detailed analysis and statistics

### **Option 2: Command Line Analysis**

#### **A. Analyze MIMIC Clinical Notes**
```bash
# Analyze 10 notes from MIMIC dataset
python scripts/mimic_analysis.py --sample-size 10 --output results.json

# Analyze 100 notes for more comprehensive results
python scripts/mimic_analysis.py --sample-size 100 --output comprehensive_results.json

# Analyze all 10,000 notes (takes longer)
python scripts/mimic_analysis.py --sample-size 10000 --output full_analysis.json
```

#### **B. Run Demo Scripts**
```bash
# Traditional rule-based demo
python scripts/demo.py

# BERT-enhanced demo
python scripts/bert_demo.py

# Hybrid processor demo
python scripts/hybrid_demo.py
```

## 📊 Understanding MIMIC Analysis Results

### **What the Analysis Shows**

When you run the MIMIC analysis, you'll see output like this:

```
================================================================================
MIMIC Clinical Notes Stigma Analysis
================================================================================
✅ Hybrid processor initialized successfully!
Loading MIMIC data (sample size: 10)...
Loaded 10000 total notes
Selected 10 notes for analysis

================================================================================
ANALYSIS RESULTS
================================================================================
📊 Total Notes Analyzed: 10
🎯 Notes with Stigma: 0 (0.0%)
✏️  Total Changes Made: 0
⚠️  Processing Errors: 0

📋 Stigma Categories Found:
   • compliance: 2 instances
   • other: 2 instances
```

### **Understanding the Results**

- **📊 Total Notes Analyzed**: Number of clinical notes processed
- **🎯 Notes with Stigma**: Notes that contained stigmatizing language
- **✏️ Total Changes Made**: Number of word replacements made
- **📋 Stigma Categories Found**: Types of stigmatizing language detected

### **Categories of Stigmatizing Language**

1. **compliance**: Terms like "refused", "noncompliant", "adherence"
2. **adamant**: Terms like "adamant", "insisted", "claimed"
3. **other**: Terms like "aggressive", "combative", "drug seeking"

## 📄 Viewing Detailed Results

### **JSON Results File**

The analysis creates a JSON file with detailed results. Here's how to view it:

```bash
# View the results file
cat mimic_demo_results.json

# Or use Python to pretty-print it
python -c "import json; data=json.load(open('mimic_demo_results.json')); print(json.dumps(data, indent=2))"
```

### **Sample Results Structure**

```json
{
  "summary": {
    "total_notes": 10,
    "notes_with_stigma": 0,
    "stigma_percentage": 0.0,
    "total_changes": 0,
    "error_count": 0
  },
  "category_breakdown": {
    "compliance": 2,
    "other": 2
  },
  "detailed_results": [
    {
      "note_id": 12345678,
      "contains_stigma": false,
      "num_changes": 0,
      "categories": ["compliance"],
      "changes": []
    }
  ]
}
```

## 🔍 Finding Examples with Stigma

To see actual examples of stigma detection and rewriting, let's run a larger analysis:

```bash
# Run analysis on more notes to find examples
python scripts/mimic_analysis.py --sample-size 100 --output detailed_results.json
```

### **Viewing Specific Examples**

```python
import json

# Load results
with open('detailed_results.json', 'r') as f:
    results = json.load(f)

# Find notes with stigma
notes_with_stigma = [r for r in results['detailed_results'] if r['contains_stigma']]

print(f"Found {len(notes_with_stigma)} notes with stigma")

# Show first example
if notes_with_stigma:
    example = notes_with_stigma[0]
    print(f"Note ID: {example['note_id']}")
    print(f"Categories: {example['categories']}")
    print(f"Changes: {example['changes']}")
```

## 🎯 Interactive Examples

### **Example 1: Single Note Analysis**

```python
from src.core.hybrid_processor import HybridProcessor

# Initialize processor
processor = HybridProcessor(use_bert=True)

# Example clinical note
note = """
Patient was adamant about their symptoms and refused to comply with treatment. 
The patient appeared agitated and uncooperative during the examination.
"""

# Process the note
result = processor.process_note(note, method="hybrid")

print("Original:", note)
print("Rewritten:", result['rewritten_text'])
print("Changes:", result['changes'])
```

### **Example 2: Batch Processing**

```python
from src.core.hybrid_processor import HybridProcessor

# Initialize processor
processor = HybridProcessor(use_bert=True)

# Multiple notes
notes = [
    "Patient refused medication.",
    "Patient was cooperative during exam.",
    "Patient appeared drug seeking."
]

# Process all notes
for i, note in enumerate(notes):
    result = processor.process_note(note, method="hybrid")
    print(f"Note {i+1}:")
    print(f"  Original: {note}")
    print(f"  Rewritten: {result['rewritten_text']}")
    print(f"  Has stigma: {result['contains_stigma']}")
    print()
```

## 🔧 Advanced Usage

### **Different Processing Methods**

```python
from src.core.hybrid_processor import HybridProcessor

processor = HybridProcessor(use_bert=True)

# Traditional rule-based processing
result_traditional = processor.process_note(text, method="traditional")

# BERT-enhanced processing
result_bert = processor.process_note(text, method="bert")

# Hybrid processing (recommended)
result_hybrid = processor.process_note(text, method="hybrid")

# Automatic method selection
result_auto = processor.process_note(text, method="auto")
```

### **Quality Control Settings**

The system includes quality control to ensure clinical accuracy:

- **Similarity threshold**: Only replaces words with high semantic similarity
- **Grammatical checks**: Ensures replacements maintain sentence structure
- **Clinical meaning preservation**: Prevents inappropriate changes
- **Context awareness**: Considers surrounding text

## 📈 Performance Tips

### **For Large Datasets**

1. **Start small**: Begin with 10-50 notes to test
2. **Monitor memory**: Large datasets may require more RAM
3. **Save results**: Always save results to avoid reprocessing
4. **Use sampling**: For very large datasets, use sampling

### **Example: Large Dataset Processing**

```bash
# Process in batches
for i in {1..10}; do
    python scripts/mimic_analysis.py --sample-size 1000 --output batch_${i}.json
done
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Import errors**: Make sure you're in the correct directory
2. **BERT loading**: First run may take time to download models
3. **Memory issues**: Reduce sample size for large datasets
4. **Port conflicts**: Change port in run_web_app.py if needed

### **Getting Help**

- Check `TROUBLESHOOTING.md` for common solutions
- Review `PROJECT_STRUCTURE.md` for file organization
- See `QUALITY_IMPROVEMENTS.md` for technical details

## 🎉 Summary

You now know how to:

✅ **Use the web interface** for interactive analysis
✅ **Run command-line analysis** on MIMIC data
✅ **Understand results** and interpret findings
✅ **View detailed examples** of stigma detection
✅ **Process large datasets** efficiently
✅ **Troubleshoot common issues**

The MedNeutral system is ready to help you detect and rewrite stigmatizing language in clinical notes while maintaining accuracy and professional standards!
