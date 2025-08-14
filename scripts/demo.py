#!/usr/bin/env python3
"""
MedNeutral Demo Script
Demonstrates the stigma detection and rewriting capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.stigma_classifier import ClinicalNoteProcessor

def print_separator(title=""):
    """Print a formatted separator"""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*60}")

def demo_single_notes():
    """Demo with individual clinical notes"""
    processor = ClinicalNoteProcessor()
    
    # Example clinical notes with stigmatizing language
    examples = [
        {
            "title": "Example 1: Patient Compliance Issues",
            "text": "Patient was adamant about their symptoms and refused to comply with the treatment plan. The patient appeared agitated and uncooperative during the examination."
        },
        {
            "title": "Example 2: Substance Use Concerns",
            "text": "Patient appears to be drug seeking and was combative when questioned about their medication use. They claimed to be in severe pain but seemed to be exaggerating their symptoms."
        },
        {
            "title": "Example 3: Behavioral Observations",
            "text": "Patient was belligerent and argumentative with staff. They were poorly groomed and appeared unmotivated to participate in their care. The patient insisted on leaving against medical advice."
        },
        {
            "title": "Example 4: Mixed Clinical Scenario",
            "text": "Patient is a poor historian and was nonadherent to their medication regimen. They were charming but seemed to be malingering. The patient refused multiple treatment options."
        }
    ]
    
    for example in examples:
        print_separator(example["title"])
        print(f"Original Text:\n{example['text']}")
        
        # Process the note
        result = processor.process_note(example["text"])
        
        print(f"\nRewritten Text:\n{result['rewritten_text']}")
        
        print(f"\nAnalysis:")
        print(f"- Contains Stigma: {result['has_stigma']}")
        print(f"- Stigma Categories: {', '.join(result['stigma_categories'])}")
        print(f"- Number of Changes: {result['num_changes']}")
        
        if result['changes_made']:
            print(f"\nChanges Made:")
            for i, change in enumerate(result['changes_made'], 1):
                print(f"  {i}. '{change['original']}' → '{change['replacement']}' ({change['category']})")

def demo_batch_processing():
    """Demo batch processing of multiple notes"""
    print_separator("Batch Processing Demo")
    
    processor = ClinicalNoteProcessor()
    
    # Sample notes for batch processing
    notes = [
        "Patient was adamant about pain levels.",
        "Patient refused medication and was uncooperative.",
        "Patient appeared agitated and combative.",
        "Patient was compliant with treatment plan.",
        "Patient was drug seeking during visit.",
        "Patient was pleasant and cooperative."
    ]
    
    print("Processing batch of clinical notes...")
    results = processor.batch_process(notes)
    
    print(f"\nBatch Results Summary:")
    total_with_stigma = sum(1 for r in results if r['has_stigma'])
    total_changes = sum(r['num_changes'] for r in results)
    
    print(f"- Total Notes: {len(notes)}")
    print(f"- Notes with Stigma: {total_with_stigma}")
    print(f"- Total Changes Made: {total_changes}")
    
    print(f"\nDetailed Results:")
    for i, (note, result) in enumerate(zip(notes, results), 1):
        status = "✓" if result['has_stigma'] else "○"
        print(f"  {i}. {status} {note[:50]}{'...' if len(note) > 50 else ''}")

def demo_keyword_analysis():
    """Demo keyword analysis and statistics"""
    print_separator("Keyword Analysis")
    
    processor = ClinicalNoteProcessor()
    
    # Analyze keyword distribution
    keyword_counts = {}
    total_keywords = 0
    
    for category, words in processor.classifier.keyword_dict.items():
        keyword_counts[category] = len(words)
        total_keywords += len(words)
    
    print(f"Total Keywords: {total_keywords}")
    print(f"Categories: {len(keyword_counts)}")
    
    print(f"\nKeywords by Category:")
    for category, count in keyword_counts.items():
        print(f"  {category.capitalize()}: {count} keywords")
    
    # Show some examples from each category
    print(f"\nSample Keywords by Category:")
    for category, words in processor.classifier.keyword_dict.items():
        sample_words = words[:5]  # Show first 5 words
        print(f"  {category.capitalize()}: {', '.join(sample_words)}")

def demo_advanced_features():
    """Demo advanced features like context detection"""
    print_separator("Advanced Features Demo")
    
    processor = ClinicalNoteProcessor()
    
    # Test with a complex note
    complex_note = """
    Patient was adamant about their symptoms and refused to comply with the treatment plan. 
    During the examination, the patient appeared agitated and uncooperative. 
    They claimed to be in severe pain but seemed to be exaggerating their symptoms. 
    The patient was poorly groomed and appeared to be drug seeking. 
    Despite being charming initially, they became belligerent when questioned about their medication use.
    """
    
    print("Complex Clinical Note Analysis:")
    print(f"Original Text:\n{complex_note.strip()}")
    
    result = processor.process_note(complex_note)
    
    print(f"\nRewritten Text:\n{result['rewritten_text']}")
    
    print(f"\nDetailed Stigma Analysis:")
    for match in result['stigma_matches']:
        print(f"- '{match['text']}' (Category: {match['category']})")
        print(f"  Context: ...{match['context']}...")
        print(f"  Position: {match['position']}")

def main():
    """Run all demos"""
    print("MedNeutral - Clinical Note Rewriter Demo")
    print("=" * 60)
    
    try:
        # Run all demo functions
        demo_keyword_analysis()
        demo_single_notes()
        demo_batch_processing()
        demo_advanced_features()
        
        print_separator("Demo Complete")
        print("The system successfully detected and rewrote stigmatizing language")
        print("while preserving clinical accuracy and important information.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
