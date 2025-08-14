import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
from core.stigma_classifier import ClinicalNoteProcessor
from core.bert_enhanced_classifier import BERTEnhancedProcessor

class HybridProcessor:
    """
    Hybrid processor that combines traditional rule-based and BERT-enhanced approaches
    for maximum flexibility and accuracy in stigma detection and replacement
    """
    
    def __init__(self, keywords_path: str = "data/keywords.json", 
                 use_bert: bool = True, bert_model: str = "bert-base-uncased"):
        """
        Initialize hybrid processor
        
        Args:
            keywords_path: Path to keywords JSON file
            use_bert: Whether to use BERT enhancement (default: True)
            bert_model: BERT model to use (default: "bert-base-uncased")
        """
        self.use_bert = use_bert
        
        # Initialize traditional processor
        self.traditional_processor = ClinicalNoteProcessor(keywords_path)
        
        # Initialize BERT processor if requested
        self.bert_processor = None
        if use_bert:
            try:
                print("Initializing BERT-enhanced processor...")
                self.bert_processor = BERTEnhancedProcessor(keywords_path)
                print("BERT processor initialized successfully!")
            except Exception as e:
                print(f"Warning: Could not initialize BERT processor: {e}")
                print("Falling back to traditional processor only.")
                self.use_bert = False
    
    def process_note(self, text: str, method: str = "auto") -> Dict:
        """
        Process a clinical note using the specified method
        
        Args:
            text: Clinical note text to process
            method: Processing method ("traditional", "bert", "hybrid", "auto")
        
        Returns:
            Dictionary with processing results
        """
        if method == "traditional":
            return self._process_traditional(text)
        elif method == "bert":
            return self._process_bert(text)
        elif method == "hybrid":
            return self._process_hybrid(text)
        elif method == "auto":
            return self._process_auto(text)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'traditional', 'bert', 'hybrid', or 'auto'")
    
    def _process_traditional(self, text: str) -> Dict:
        """Process using traditional rule-based approach"""
        result = self.traditional_processor.process_note(text)
        result["method"] = "traditional"
        result["bert_enhanced"] = False
        return result
    
    def _process_bert(self, text: str) -> Dict:
        """Process using BERT-enhanced approach"""
        if not self.bert_processor:
            raise RuntimeError("BERT processor not available. Use traditional method instead.")
        
        result = self.bert_processor.process_note(text)
        result["method"] = "bert"
        return result
    
    def _process_hybrid(self, text: str) -> Dict:
        """Process using hybrid approach (combine both methods)"""
        # Get results from both methods
        traditional_result = self._process_traditional(text)
        bert_result = self._process_bert(text) if self.bert_processor else None
        
        # Combine results
        hybrid_result = {
            "original_text": text,
            "method": "hybrid",
            "bert_enhanced": self.bert_processor is not None,
            "traditional_result": traditional_result,
            "bert_result": bert_result
        }
        
        # Choose the best approach based on number of changes and confidence
        if bert_result and bert_result["num_changes"] > traditional_result["num_changes"]:
            # Use BERT result as primary
            hybrid_result.update({
                "rewritten_text": bert_result["rewritten_text"],
                "has_stigma": bert_result["has_stigma"],
                "stigma_categories": bert_result["stigma_categories"],
                "changes_made": bert_result["changes_made"],
                "num_changes": bert_result["num_changes"],
                "primary_method": "bert"
            })
        else:
            # Use traditional result as primary
            hybrid_result.update({
                "rewritten_text": traditional_result["rewritten_text"],
                "has_stigma": traditional_result["has_stigma"],
                "stigma_categories": traditional_result["stigma_categories"],
                "changes_made": traditional_result["changes_made"],
                "num_changes": traditional_result["num_changes"],
                "primary_method": "traditional"
            })
        
        return hybrid_result
    
    def _process_auto(self, text: str) -> Dict:
        """Automatically choose the best method based on text characteristics"""
        # Simple heuristic: use BERT for longer, more complex texts
        if self.bert_processor and len(text.split()) > 20:
            return self._process_bert(text)
        else:
            return self._process_traditional(text)
    
    def compare_methods(self, text: str) -> Dict:
        """
        Compare results from all available methods
        
        Args:
            text: Clinical note text to process
        
        Returns:
            Dictionary comparing results from different methods
        """
        comparison = {
            "original_text": text,
            "methods_available": []
        }
        
        # Traditional method
        traditional_result = self._process_traditional(text)
        comparison["traditional"] = {
            "rewritten_text": traditional_result["rewritten_text"],
            "num_changes": traditional_result["num_changes"],
            "has_stigma": traditional_result["has_stigma"],
            "stigma_categories": traditional_result["stigma_categories"]
        }
        comparison["methods_available"].append("traditional")
        
        # BERT method
        if self.bert_processor:
            bert_result = self._process_bert(text)
            comparison["bert"] = {
                "rewritten_text": bert_result["rewritten_text"],
                "num_changes": bert_result["num_changes"],
                "has_stigma": bert_result["has_stigma"],
                "stigma_categories": bert_result["stigma_categories"],
                "confidence_scores": [match["confidence"] for match in bert_result["stigma_matches"]]
            }
            comparison["methods_available"].append("bert")
        
        # Hybrid method
        if self.bert_processor:
            hybrid_result = self._process_hybrid(text)
            comparison["hybrid"] = {
                "rewritten_text": hybrid_result["rewritten_text"],
                "num_changes": hybrid_result["num_changes"],
                "has_stigma": hybrid_result["has_stigma"],
                "stigma_categories": hybrid_result["stigma_categories"],
                "primary_method": hybrid_result["primary_method"]
            }
            comparison["methods_available"].append("hybrid")
        
        return comparison
    
    def get_word_alternatives(self, word: str, method: str = "bert", top_k: int = 10) -> List:
        """
        Get alternative words for a given word using specified method
        
        Args:
            word: Word to find alternatives for
            method: Method to use ("traditional" or "bert")
            top_k: Number of alternatives to return
        
        Returns:
            List of alternative words with scores
        """
        if method == "bert" and self.bert_processor:
            return self.bert_processor.get_word_alternatives(word, top_k)
        elif method == "traditional":
            # Get traditional replacements
            for category, replacements in self.traditional_processor.rewriter.replacements.items():
                if word.lower() in replacements:
                    return [(replacements[word.lower()], 1.0)]
            return []
        else:
            raise ValueError(f"Method '{method}' not available")
    
    def analyze_text_complexity(self, text: str) -> Dict:
        """
        Analyze text complexity to help choose the best processing method
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with complexity metrics
        """
        words = text.split()
        sentences = text.split('.')
        
        complexity = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "recommended_method": "traditional"
        }
        
        # Recommend BERT for complex texts
        if len(words) > 20 or len(sentences) > 3:
            complexity["recommended_method"] = "bert" if self.bert_processor else "traditional"
        
        return complexity

def create_processor(use_bert: bool = True, **kwargs) -> HybridProcessor:
    """
    Factory function to create a processor with error handling
    
    Args:
        use_bert: Whether to attempt BERT initialization
        **kwargs: Additional arguments for processor initialization
    
    Returns:
        HybridProcessor instance
    """
    try:
        return HybridProcessor(use_bert=use_bert, **kwargs)
    except Exception as e:
        print(f"Error creating processor: {e}")
        print("Creating traditional-only processor...")
        return HybridProcessor(use_bert=False, **kwargs)
