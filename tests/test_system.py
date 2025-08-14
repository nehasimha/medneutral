#!/usr/bin/env python3
"""
Test script for MedNeutral system
Verifies stigma detection and rewriting functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.stigma_classifier import ClinicalNoteProcessor

def test_basic_functionality():
    """Test basic stigma detection and rewriting"""
    print("Testing basic functionality...")
    
    processor = ClinicalNoteProcessor()
    
    # Test case 1: Simple stigma detection
    test_text = "Patient was adamant about symptoms."
    result = processor.process_note(test_text)
    
    assert result['has_stigma'] == True, "Should detect stigma"
    assert 'adamant' in result['stigma_categories'], "Should detect adamant category"
    assert result['num_changes'] == 1, "Should make one change"
    assert result['rewritten_text'] == "Patient was firm about symptoms.", "Should replace adamant with firm"
    
    print("✓ Basic functionality test passed")

def test_multiple_stigma_terms():
    """Test detection of multiple stigmatizing terms"""
    print("Testing multiple stigma terms...")
    
    processor = ClinicalNoteProcessor()
    
    test_text = "Patient refused treatment and was combative."
    result = processor.process_note(test_text)
    
    assert result['has_stigma'] == True, "Should detect stigma"
    assert result['num_changes'] == 2, "Should make two changes"
    assert 'refused' not in result['rewritten_text'], "Should replace refused"
    assert 'combative' not in result['rewritten_text'], "Should replace combative"
    
    print("✓ Multiple stigma terms test passed")

def test_no_stigma():
    """Test text with no stigmatizing language"""
    print("Testing no stigma detection...")
    
    processor = ClinicalNoteProcessor()
    
    test_text = "Patient was cooperative and followed treatment plan."
    result = processor.process_note(test_text)
    
    assert result['has_stigma'] == False, "Should not detect stigma"
    assert result['num_changes'] == 0, "Should make no changes"
    assert result['rewritten_text'] == test_text, "Should not change text"
    
    print("✓ No stigma test passed")

def test_case_insensitive():
    """Test case insensitive detection"""
    print("Testing case insensitive detection...")
    
    processor = ClinicalNoteProcessor()
    
    test_text = "Patient was ADAMANT and REFUSED treatment."
    result = processor.process_note(test_text)
    
    assert result['has_stigma'] == True, "Should detect stigma regardless of case"
    assert result['num_changes'] == 2, "Should make two changes"
    
    print("✓ Case insensitive test passed")

def test_context_preservation():
    """Test that context is preserved during rewriting"""
    print("Testing context preservation...")
    
    processor = ClinicalNoteProcessor()
    
    test_text = "Patient was adamant about pain levels and refused medication."
    result = processor.process_note(test_text)
    
    # Check that the overall meaning is preserved
    assert "pain levels" in result['rewritten_text'], "Should preserve clinical information"
    assert "medication" in result['rewritten_text'], "Should preserve clinical information"
    assert "adamant" not in result['rewritten_text'], "Should replace stigmatizing term"
    assert "refused" not in result['rewritten_text'], "Should replace stigmatizing term"
    
    print("✓ Context preservation test passed")

def test_batch_processing():
    """Test batch processing functionality"""
    print("Testing batch processing...")
    
    processor = ClinicalNoteProcessor()
    
    test_texts = [
        "Patient was adamant.",
        "Patient refused treatment.",
        "Patient was cooperative."
    ]
    
    results = processor.batch_process(test_texts)
    
    assert len(results) == 3, "Should process all texts"
    assert results[0]['has_stigma'] == True, "First text should have stigma"
    assert results[1]['has_stigma'] == True, "Second text should have stigma"
    assert results[2]['has_stigma'] == False, "Third text should not have stigma"
    
    print("✓ Batch processing test passed")

def test_keyword_categories():
    """Test that keywords are properly categorized"""
    print("Testing keyword categories...")
    
    processor = ClinicalNoteProcessor()
    
    # Test adamant category
    adamant_text = "Patient claimed symptoms."
    adamant_result = processor.process_note(adamant_text)
    assert 'adamant' in adamant_result['stigma_categories'], "Should detect adamant category"
    
    # Test compliance category
    compliance_text = "Patient refused medication."
    compliance_result = processor.process_note(compliance_text)
    assert 'compliance' in compliance_result['stigma_categories'], "Should detect compliance category"
    
    # Test other category
    other_text = "Patient was aggressive."
    other_result = processor.process_note(other_text)
    assert 'other' in other_result['stigma_categories'], "Should detect other category"
    
    print("✓ Keyword categories test passed")

def run_all_tests():
    """Run all tests"""
    print("MedNeutral System Tests")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_multiple_stigma_terms()
        test_no_stigma()
        test_case_insensitive()
        test_context_preservation()
        test_batch_processing()
        test_keyword_categories()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! The system is working correctly.")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
