#!/usr/bin/env python3
"""
Hybrid MedNeutral Demo
Demonstrates the hybrid processor that combines traditional and BERT approaches
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.hybrid_processor import create_processor

def print_separator(title=""):
    """Print a formatted separator"""
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
    else:
        print(f"\n{'-'*70}")

def demo_processor_initialization():
    """Demo processor initialization with fallback handling"""
    print_separator("Hybrid Processor Initialization")
    
    print("Attempting to create BERT-enhanced processor...")
    processor = create_processor(use_bert=True)
    
    print(f"Processor created successfully!")
    print(f"BERT available: {processor.bert_processor is not None}")
    print(f"Methods available: {['traditional'] + (['bert', 'hybrid'] if processor.bert_processor else [])}")

def demo_method_comparison():
    """Demo comparison of different processing methods"""
    print_separator("Method Comparison Demo")
    
    processor = create_processor(use_bert=True)
    
    # Test cases with varying complexity
    test_cases = [
        {
            "title": "Simple Case",
            "text": "Patient was adamant about symptoms."
        },
        {
            "title": "Complex Case", 
            "text": "Patient was adamant about their symptoms and refused to comply with the treatment plan. The patient appeared agitated and uncooperative during the examination. They claimed to be in severe pain but seemed to be exaggerating their symptoms."
        }
    ]
    
    for test_case in test_cases:
        print_separator(test_case["title"])
        print(f"Text: '{test_case['text']}'")
        
        # Analyze complexity
        complexity = processor.analyze_text_complexity(test_case["text"])
        print(f"\nText Complexity Analysis:")
        print(f"- Word count: {complexity['word_count']}")
        print(f"- Sentence count: {complexity['sentence_count']}")
        print(f"- Average sentence length: {complexity['avg_sentence_length']:.1f}")
        print(f"- Recommended method: {complexity['recommended_method']}")
        
        # Compare all methods
        comparison = processor.compare_methods(test_case["text"])
        
        print(f"\nMethod Comparison:")
        for method in comparison["methods_available"]:
            result = comparison[method]
            print(f"\n{method.upper()} METHOD:")
            print(f"  Rewritten: '{result['rewritten_text']}'")
            print(f"  Changes: {result['num_changes']}")
            print(f"  Has stigma: {result['has_stigma']}")
            if method == "bert" and "confidence_scores" in result:
                avg_confidence = sum(result["confidence_scores"]) / len(result["confidence_scores"]) if result["confidence_scores"] else 0
                print(f"  Avg confidence: {avg_confidence:.3f}")

def demo_auto_method_selection():
    """Demo automatic method selection based on text characteristics"""
    print_separator("Automatic Method Selection")
    
    processor = create_processor(use_bert=True)
    
    # Test texts of varying complexity
    test_texts = [
        "Patient was adamant.",
        "Patient was adamant about symptoms and refused treatment.",
        "Patient was adamant about their symptoms and refused to comply with the treatment plan. The patient appeared agitated and uncooperative during the examination. They claimed to be in severe pain but seemed to be exaggerating their symptoms. The patient was poorly groomed and appeared to be drug seeking."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: '{text}'")
        
        # Analyze complexity
        complexity = processor.analyze_text_complexity(text)
        print(f"Complexity: {complexity['word_count']} words, {complexity['sentence_count']} sentences")
        print(f"Recommended method: {complexity['recommended_method']}")
        
        # Process with auto method
        result = processor.process_note(text, method="auto")
        print(f"Auto-selected method: {result['method']}")
        print(f"Rewritten: '{result['rewritten_text']}'")
        print(f"Changes: {result['num_changes']}")

def demo_word_alternatives():
    """Demo word alternative suggestions using different methods"""
    print_separator("Word Alternative Suggestions")
    
    processor = create_processor(use_bert=True)
    
    test_words = ["adamant", "refused", "aggressive", "drug seeking"]
    
    for word in test_words:
        print(f"\nAlternatives for '{word}':")
        
        # Traditional alternatives
        if processor.bert_processor:
            print("  BERT-based alternatives:")
            bert_alternatives = processor.get_word_alternatives(word, method="bert", top_k=3)
            for alt, score in bert_alternatives:
                print(f"    - '{alt}' (similarity: {score:.3f})")
        
        # Traditional alternatives
        print("  Traditional alternatives:")
        trad_alternatives = processor.get_word_alternatives(word, method="traditional")
        for alt, score in trad_alternatives:
            print(f"    - '{alt}' (score: {score:.3f})")

def demo_hybrid_processing():
    """Demo hybrid processing approach"""
    print_separator("Hybrid Processing Approach")
    
    processor = create_processor(use_bert=True)
    
    complex_text = """
    Patient was adamant about their symptoms and refused to comply with the treatment plan. 
    During the examination, the patient appeared agitated and uncooperative. 
    They claimed to be in severe pain but seemed to be exaggerating their symptoms. 
    The patient was poorly groomed and appeared to be drug seeking. 
    Despite being charming initially, they became belligerent when questioned about their medication use.
    """
    
    print("Complex Clinical Note:")
    print(f"'{complex_text.strip()}'")
    
    # Process with hybrid method
    result = processor.process_note(complex_text, method="hybrid")
    
    print(f"\nHybrid Processing Results:")
    print(f"Primary method used: {result['primary_method']}")
    print(f"BERT enhanced: {result['bert_enhanced']}")
    print(f"Rewritten text: '{result['rewritten_text']}'")
    print(f"Number of changes: {result['num_changes']}")
    
    if result['bert_enhanced']:
        print(f"\nDetailed BERT Analysis:")
        for change in result['changes_made']:
            print(f"  '{change['original']}' → '{change['replacement']}'")
            print(f"    Similarity score: {change['similarity_score']:.3f}")
            print(f"    Category: {change['category']}")

def demo_error_handling():
    """Demo error handling and fallback behavior"""
    print_separator("Error Handling and Fallback")
    
    print("Creating processor with BERT (may fail if dependencies not installed)...")
    processor = create_processor(use_bert=True)
    
    print(f"BERT available: {processor.bert_processor is not None}")
    
    test_text = "Patient was adamant about symptoms."
    
    # Try different methods
    methods_to_try = ["traditional", "auto"]
    if processor.bert_processor:
        methods_to_try.extend(["bert", "hybrid"])
    
    for method in methods_to_try:
        try:
            print(f"\nTrying {method} method...")
            result = processor.process_note(test_text, method=method)
            print(f"  Success! Changes: {result['num_changes']}")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Run all hybrid demos"""
    print("Hybrid MedNeutral Demo")
    print("=" * 70)
    print("This demo showcases the hybrid processor that combines")
    print("traditional rule-based and BERT-enhanced approaches.")
    
    try:
        # Run all demo functions
        demo_processor_initialization()
        demo_method_comparison()
        demo_auto_method_selection()
        demo_word_alternatives()
        demo_hybrid_processing()
        demo_error_handling()
        
        print_separator("Demo Complete")
        print("The hybrid processor provides maximum flexibility and")
        print("robustness by combining multiple approaches.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install torch transformers scikit-learn")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
