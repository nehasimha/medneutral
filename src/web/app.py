from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.stigma_classifier import ClinicalNoteProcessor
import os

app = Flask(__name__)

# Initialize the processor (it will find the keywords.json automatically)
processor = ClinicalNoteProcessor()

@app.route('/')
def index():
    """Main page with text input form"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_note():
    """Process clinical note and return results"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Please provide some text to process'}), 400
        
        # Process the note
        result = processor.process_note(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_process', methods=['POST'])
def batch_process():
    """Process multiple clinical notes"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Please provide texts to process'}), 400
        
        # Process all notes
        results = processor.batch_process(texts)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get statistics about the stigma detection system"""
    try:
        # Count keywords by category
        keyword_counts = {}
        for category, words in processor.classifier.keyword_dict.items():
            keyword_counts[category] = len(words)
        
        return jsonify({
            'total_keywords': sum(keyword_counts.values()),
            'categories': keyword_counts,
            'categories_count': len(keyword_counts)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
