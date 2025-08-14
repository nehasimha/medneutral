#!/usr/bin/env python3
"""
BERT-Enhanced MedNeutral Demo
Demonstrates BERT-based stigma detection and intelligent word replacement
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.bert_enhanced_classifier import BERTEnhancedProcessor

def print_separator(title=""):
    """Print a formatted separator"""
    if title:
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
    else:
        print(f"\n{'-'*70}")

def demo_bert_word_similarity():
    """Demo BERT-based word similarity analysis"""
    print_separator("BERT Word Similarity Analysis")
    
    processor = BERTEnhancedProcessor()
    
    # Test word similarities
    test_pairs = [
        ("adamant", "firm"),
        ("adamant", "stubborn"),
        ("refused", "declined"),
        ("refused", "accepted"),
        ("aggressive", "agitated"),
        ("aggressive", "calm"),
        ("drug seeking", "seeking medication"),
        ("drug seeking", "avoiding treatment")
    ]
    
    print("Word Similarity Analysis using BERT:")
    print("-" * 50)
    
    for word1, word2 in test_pairs:
        similarity = processor.analyze_word_similarity(word1, word2)
        print(f"'{word1}' vs '{word2}': {similarity:.4f}")
    
    print("\nHigher scores indicate more similar meanings in BERT's understanding.")

def demo_bert_alternatives():
    """Demo BERT-based alternative word suggestions"""
    print_separator("BERT Alternative Word Suggestions")
    
    processor = BERTEnhancedProcessor()
    
    # Test stigmatizing words
    test_words = [
        "adamant",
        "refused", 
        "aggressive",
        "drug seeking",
        "noncompliant",
        "belligerent"
    ]
    
    print("BERT-Based Alternative Suggestions:")
    print("-" * 50)
    
    for word in test_words:
        print(f"\nAlternatives for '{word}':")
        alternatives = processor.get_word_alternatives(word, top_k=5)
        
        for i, (alt_word, similarity) in enumerate(alternatives, 1):
            print(f"  {i}. '{alt_word}' (similarity: {similarity:.4f})")

def demo_bert_enhanced_processing():
    """Demo BERT-enhanced clinical note processing"""
    print_separator("BERT-Enhanced Clinical Note Processing")
    
    processor = BERTEnhancedProcessor()
    
    # Example clinical notes
    examples = [
        {
            "title": "Example 1: Complex Behavioral Description",
            "text": "Patient was adamant about their symptoms and refused to comply with the treatment plan. The patient appeared agitated and uncooperative during the examination."
        },
        {
            "title": "Example 2: Substance Use Concerns",
            "text": "Patient appears to be drug seeking and was combative when questioned about their medication use. They claimed to be in severe pain but seemed to be exaggerating their symptoms."
        },
        {
            "title": "Example 3: Mixed Clinical Scenario",
            "text": "Patient is a poor historian and was nonadherent to their medication regimen. They were charming but seemed to be malingering. The patient refused multiple treatment options."
        }
    ]
    
    for example in examples:
        print_separator(example["title"])
        print(f"Original Text:\n{example['text']}")
        
        # Process with BERT enhancement
        result = processor.process_note(example["text"])
        
        print(f"\nBERT-Enhanced Rewritten Text:\n{result['rewritten_text']}")
        
        print(f"\nBERT Analysis:")
        print(f"- Contains Stigma: {result['has_stigma']}")
        print(f"- Stigma Categories: {', '.join(result['stigma_categories'])}")
        print(f"- Number of Changes: {result['num_changes']}")
        print(f"- BERT Enhanced: {result['bert_enhanced']}")
        
        if result['changes_made']:
            print(f"\nBERT-Based Changes Made:")
            for i, change in enumerate(result['changes_made'], 1):
                print(f"  {i}. '{change['original']}' → '{change['replacement']}'")
                print(f"     Category: {change['category']}")
                print(f"     Similarity Score: {change['similarity_score']:.4f}")
                print(f"     All Alternatives: {[alt[0] for alt in change['all_alternatives'][:3]]}")

def demo_bert_confidence_scoring():
    """Demo BERT-based confidence scoring"""
    print_separator("BERT Confidence Scoring")
    
    processor = BERTEnhancedProcessor()
    
    # Test different types of stigmatizing language
    test_cases = [
        "Patient was adamant about symptoms.",
        "Patient refused medication.",
        "Patient was aggressive during exam.",
        "Patient appeared drug seeking.",
        "Patient was noncompliant with treatment."
    ]
    
    print("BERT Confidence Scores for Stigma Detection:")
    print("-" * 50)
    
    for test_case in test_cases:
        result = processor.process_note(test_case)
        
        print(f"\nText: '{test_case}'")
        if result['stigma_matches']:
            for match in result['stigma_matches']:
                print(f"  Detected: '{match['text']}' (confidence: {match['confidence']:.4f})")
        else:
            print("  No stigmatizing language detected")

def demo_bert_vs_traditional():
    """Compare BERT-enhanced vs traditional approach"""
    print_separator("BERT-Enhanced vs Traditional Approach")
    
    from models.stigma_classifier import ClinicalNoteProcessor
    
    bert_processor = BERTEnhancedProcessor()
    traditional_processor = ClinicalNoteProcessor()
    
    test_text = "Patient was adamant about their symptoms and refused to comply with treatment."
    
    print("Test Text:")
    print(f"'{test_text}'")
    
    # Traditional approach
    print("\n" + "="*40)
    print("TRADITIONAL APPROACH:")
    traditional_result = traditional_processor.process_note(test_text)
    print(f"Rewritten: '{traditional_result['rewritten_text']}'")
    print(f"Changes: {traditional_result['num_changes']}")
    
    # BERT-enhanced approach
    print("\n" + "="*40)
    print("BERT-ENHANCED APPROACH:")
    bert_result = bert_processor.process_note(test_text)
    print(f"Rewritten: '{bert_result['rewritten_text']}'")
    print(f"Changes: {bert_result['num_changes']}")
    
    print("\nBERT Advantages:")
    print("- More intelligent word selection based on semantic similarity")
    print("- Confidence scoring for each detection")
    print("- Multiple alternative suggestions for each word")
    print("- Better understanding of word context and meaning")

def main():
    """Run all BERT-enhanced demos"""
    print("BERT-Enhanced MedNeutral Demo")
    print("=" * 70)
    print("This demo showcases BERT-based stigma detection and intelligent")
    print("word replacement using semantic similarity analysis.")
    
    try:
        # Run all demo functions
        demo_bert_word_similarity()
        demo_bert_alternatives()
        demo_bert_enhanced_processing()
        demo_bert_confidence_scoring()
        demo_bert_vs_traditional()
        
        print_separator("Demo Complete")
        print("The BERT-enhanced system provides more intelligent and")
        print("context-aware stigma detection and replacement.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install torch transformers scikit-learn")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
