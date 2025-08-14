#!/usr/bin/env python3
"""
MIMIC Results Viewer

This script helps you view and analyze results from MIMIC clinical notes analysis,
showing specific examples of stigma detection and rewriting.
"""

import json
import sys
import pandas as pd
from pathlib import Path

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def show_summary(results):
    """Display summary statistics"""
    summary = results['summary']
    print("=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📋 Total Notes Analyzed: {summary['total_notes']}")
    print(f"🎯 Notes with Stigma: {summary['notes_with_stigma']} ({summary['stigma_percentage']:.1f}%)")
    print(f"✏️  Total Changes Made: {summary['total_changes']}")
    print(f"⚠️  Processing Errors: {summary['error_count']}")
    
    if 'category_breakdown' in results and results['category_breakdown']:
        print(f"\n📋 Stigma Categories Found:")
        for category, count in results['category_breakdown'].items():
            print(f"   • {category}: {count} instances")

def show_examples_with_stigma(results, max_examples=5):
    """Show examples of notes with stigma detection"""
    detailed_results = results['detailed_results']
    
    # Find notes with stigma
    notes_with_stigma = [r for r in detailed_results if r.get('contains_stigma', False)]
    
    if not notes_with_stigma:
        print("\n🔍 No notes with stigma found in this sample.")
        print("   Try running analysis with a larger sample size:")
        print("   python scripts/mimic_analysis.py --sample-size 500")
        return
    
    print(f"\n📝 EXAMPLES WITH STIGMA DETECTION ({len(notes_with_stigma)} found)")
    print("=" * 60)
    
    for i, note in enumerate(notes_with_stigma[:max_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Note ID: {note['note_id']}")
        print(f"Categories: {', '.join(note['categories'])}")
        print(f"Changes Made: {note['num_changes']}")
        
        if note['changes']:
            print("Changes:")
            for j, change in enumerate(note['changes'][:3]):  # Show first 3 changes
                print(f"  {j+1}. '{change.get('original', '')}' → '{change.get('replacement', '')}'")
                if 'context' in change:
                    context = change['context'][:80] + "..." if len(change['context']) > 80 else change['context']
                    print(f"     Context: {context}")
        print()

def show_notes_with_categories(results, category=None):
    """Show notes that contain specific stigma categories"""
    detailed_results = results['detailed_results']
    
    if category:
        # Filter by specific category
        notes_with_category = [r for r in detailed_results if category in r.get('categories', [])]
        print(f"\n📋 NOTES WITH '{category.upper()}' CATEGORY ({len(notes_with_category)} found)")
    else:
        # Show all notes with any stigma
        notes_with_category = [r for r in detailed_results if r.get('categories', [])]
        print(f"\n📋 NOTES WITH ANY STIGMA CATEGORY ({len(notes_with_category)} found)")
    
    print("=" * 60)
    
    for i, note in enumerate(notes_with_category[:10]):  # Show first 10
        print(f"{i+1}. Note ID: {note['note_id']}")
        print(f"   Categories: {', '.join(note['categories'])}")
        print(f"   Changes: {note['num_changes']}")
        print()

def show_common_changes(results):
    """Show most common word changes made"""
    if 'common_changes' not in results or not results['common_changes']:
        print("\n🔄 No changes were made in this analysis.")
        return
    
    print("\n🔄 MOST COMMON CHANGES")
    print("=" * 60)
    
    changes = results['common_changes']
    for i, ((original, replacement), count) in enumerate(list(changes.items())[:10]):
        print(f"{i+1}. '{original}' → '{replacement}': {count} times")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="View MIMIC analysis results")
    parser.add_argument("--file", type=str, default="detailed_mimic_results.json",
                       help="Results file to analyze (default: detailed_mimic_results.json)")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary only")
    parser.add_argument("--examples", action="store_true",
                       help="Show examples with stigma")
    parser.add_argument("--category", type=str, choices=['compliance', 'adamant', 'other'],
                       help="Show notes with specific category")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        print("Available files:")
        for f in Path(".").glob("*.json"):
            print(f"  - {f}")
        return
    
    # Load results
    print(f"📂 Loading results from {args.file}...")
    results = load_results(args.file)
    
    # Run based on arguments
    if args.summary:
        show_summary(results)
    elif args.examples:
        show_examples_with_stigma(results)
    elif args.category:
        show_notes_with_categories(results, args.category)
    else:
        # Default: show summary and examples
        show_summary(results)
        show_examples_with_stigma(results)

if __name__ == "__main__":
    main()
