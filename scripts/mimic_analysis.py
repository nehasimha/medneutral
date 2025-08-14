#!/usr/bin/env python3
"""
MIMIC Clinical Notes Analysis Script

This script analyzes real clinical notes from the MIMIC dataset using the
MedNeutral stigma detection and rewriting system.
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.hybrid_processor import HybridProcessor
from core.stigma_classifier import ClinicalNoteProcessor

def load_mimic_data(sample_size=100):
    """Load a sample of MIMIC clinical notes"""
    print(f"Loading MIMIC data (sample size: {sample_size})...")
    
    # Read the CSV file
    df = pd.read_csv('data/mimic_sample.csv')
    print(f"Loaded {len(df)} total notes")
    
    # Sample the data
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"Selected {len(df_sample)} notes for analysis")
    return df_sample

def analyze_note(processor, text, note_id):
    """Analyze a single clinical note"""
    try:
        # Process the note
        result = processor.process_note(text, method="hybrid")
        
        # Extract key information
        contains_stigma = result.get('contains_stigma', False)
        num_changes = len(result.get('changes', []))
        categories = result.get('stigma_categories', [])
        
        return {
            'note_id': note_id,
            'contains_stigma': contains_stigma,
            'num_changes': num_changes,
            'categories': categories,
            'original_length': len(text),
            'rewritten_length': len(result.get('rewritten_text', text)),
            'changes': result.get('changes', [])
        }
    except Exception as e:
        print(f"Error processing note {note_id}: {e}")
        return {
            'note_id': note_id,
            'error': str(e),
            'contains_stigma': False,
            'num_changes': 0,
            'categories': []
        }

def analyze_mimic_dataset(sample_size=100, output_file=None):
    """Analyze MIMIC dataset for stigma detection and rewriting"""
    
    print("=" * 80)
    print("MIMIC Clinical Notes Stigma Analysis")
    print("=" * 80)
    
    # Initialize processor
    print("Initializing MedNeutral processor...")
    try:
        processor = HybridProcessor(use_bert=True)
        print("✅ Hybrid processor initialized successfully!")
    except Exception as e:
        print(f"⚠️  BERT initialization failed: {e}")
        print("Falling back to traditional processor...")
        processor = ClinicalNoteProcessor()
    
    # Load MIMIC data
    df = load_mimic_data(sample_size)
    
    # Analyze notes
    print(f"\nAnalyzing {len(df)} clinical notes...")
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing note {idx + 1}/{len(df)}...")
        
        result = analyze_note(processor, row['text'], row['subject_id'])
        results.append(result)
    
    # Compile statistics
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Basic statistics
    total_notes = len(results)
    notes_with_stigma = sum(1 for r in results if r.get('contains_stigma', False))
    total_changes = sum(r.get('num_changes', 0) for r in results)
    error_count = sum(1 for r in results if 'error' in r)
    
    print(f"📊 Total Notes Analyzed: {total_notes}")
    print(f"🎯 Notes with Stigma: {notes_with_stigma} ({notes_with_stigma/total_notes*100:.1f}%)")
    print(f"✏️  Total Changes Made: {total_changes}")
    print(f"⚠️  Processing Errors: {error_count}")
    
    # Category breakdown
    all_categories = []
    for r in results:
        all_categories.extend(r.get('categories', []))
    
    if all_categories:
        category_counts = pd.Series(all_categories).value_counts()
        print(f"\n📋 Stigma Categories Found:")
        for category, count in category_counts.items():
            print(f"   • {category}: {count} instances")
    
    # Most common changes
    all_changes = []
    for r in results:
        for change in r.get('changes', []):
            original = change.get('original', '')
            replacement = change.get('replacement', '')
            if original and replacement:
                all_changes.append((original, replacement))
    
    if all_changes:
        change_counts = pd.Series(all_changes).value_counts()
        print(f"\n🔄 Most Common Changes:")
        for (original, replacement), count in change_counts.head(10).items():
            print(f"   • '{original}' → '{replacement}': {count} times")
    
    # Detailed examples
    print(f"\n📝 Detailed Examples:")
    examples_shown = 0
    for r in results:
        if r.get('contains_stigma', False) and examples_shown < 3:
            print(f"\n--- Example {examples_shown + 1} ---")
            print(f"Note ID: {r['note_id']}")
            print(f"Categories: {', '.join(r['categories'])}")
            print(f"Changes: {r['num_changes']}")
            
            # Show first few changes
            for i, change in enumerate(r.get('changes', [])[:3]):
                print(f"  {i+1}. '{change.get('original', '')}' → '{change.get('replacement', '')}'")
                if 'context' in change:
                    context = change['context'][:100] + "..." if len(change['context']) > 100 else change['context']
                    print(f"     Context: {context}")
            
            examples_shown += 1
    
    # Save results
    if output_file:
        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_notes': total_notes,
                    'notes_with_stigma': notes_with_stigma,
                    'stigma_percentage': notes_with_stigma/total_notes*100,
                    'total_changes': total_changes,
                    'error_count': error_count
                },
                'category_breakdown': category_counts.to_dict() if all_categories else {},
                'common_changes': change_counts.head(20).to_dict() if all_changes else {},
                'detailed_results': results
            }, f, indent=2)
        print("✅ Results saved successfully!")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MIMIC clinical notes for stigma detection")
    parser.add_argument("--sample-size", type=int, default=100, 
                       help="Number of notes to analyze (default: 100)")
    parser.add_argument("--output", type=str, default="mimic_analysis_results.json",
                       help="Output file for results (default: mimic_analysis_results.json)")
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_mimic_dataset(
        sample_size=args.sample_size,
        output_file=args.output
    )
    
    print(f"\n🎉 Analysis complete! Processed {len(results)} clinical notes.")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
