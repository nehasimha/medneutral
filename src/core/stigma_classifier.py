import json
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class StigmaMatch:
    """Represents a detected stigmatizing phrase and its context"""
    original_text: str
    start_pos: int
    end_pos: int
    category: str
    confidence: float
    context: str

class StigmaClassifier:
    """Detects and classifies stigmatizing language in clinical text"""
    
    def __init__(self, keywords_path: str = None):
        """Initialize with keyword dictionary"""
        if keywords_path is None:
            # Default to data directory relative to project root
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            keywords_path = str(project_root / "data" / "keywords.json")
        
        with open(keywords_path, 'r') as f:
            self.keyword_dict = json.load(f)
        
        # Create regex patterns for each category
        self.patterns = {}
        for category, words in self.keyword_dict.items():
            # Create case-insensitive patterns
            pattern = r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def detect_stigma(self, text: str) -> List[StigmaMatch]:
        """Detect all stigmatizing phrases in text"""
        matches = []
        text_lower = text.lower()
        
        for category, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                # Get context (50 chars before and after)
                start_ctx = max(0, match.start() - 50)
                end_ctx = min(len(text), match.end() + 50)
                context = text[start_ctx:end_ctx]
                
                stigma_match = StigmaMatch(
                    original_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    category=category,
                    confidence=1.0,  # Could be enhanced with ML model
                    context=context
                )
                matches.append(stigma_match)
        
        return matches
    
    def has_stigma(self, text: str) -> bool:
        """Quick check if text contains any stigmatizing language"""
        return len(self.detect_stigma(text)) > 0
    
    def get_stigma_categories(self, text: str) -> List[str]:
        """Get list of stigma categories present in text"""
        matches = self.detect_stigma(text)
        return list(set(match.category for match in matches))

class StigmaRewriter:
    """Rewrites stigmatizing language to neutral alternatives"""
    
    def __init__(self):
        """Initialize with replacement dictionaries"""
        self.replacements = {
            "adamant": {
                "adamant": "firm",
                "adamantly": "firmly", 
                "adament": "firm",
                "adamently": "firmly",
                "claim": "report",
                "claimed": "reported",
                "claiming": "reporting",
                "claims": "reports",
                "insist": "state",
                "insisted": "stated",
                "insistence": "statement",
                "insisting": "stating",
                "insists": "states"
            },
            "compliance": {
                "adherance": "adherence",
                "adhere": "follow",
                "adhered": "followed",
                "adherence": "compliance",
                "adherent": "compliant",
                "adheres": "follows",
                "adhering": "following",
                "compliance": "adherence",
                "compliant": "adherent",
                "complied": "followed",
                "complies": "follows",
                "comply": "follow",
                "complying": "following",
                "declined": "did not accept",
                "declines": "does not accept",
                "declining": "not accepting",
                "nonadherance": "non-adherence",
                "nonadherence": "non-adherence",
                "nonadherent": "non-adherent",
                "noncompliance": "non-adherence",
                "noncompliant": "non-adherent",
                "refusal": "declined",
                "refuse": "decline",
                "refused": "declined",
                "refuses": "declines",
                "refusing": "declining"
            },
            "other": {
                "aggression": "agitation",
                "aggressive": "agitated",
                "aggressively": "agitated",
                "agitated": "agitated",
                "agitation": "agitation",
                "anger": "frustration",
                "angered": "frustrated",
                "angers": "frustrates",
                "angrier": "more frustrated",
                "angrily": "frustrated",
                "angry": "frustrated",
                "argumentative": "disagreeing",
                "argumentatively": "disagreeing",
                "belligerence": "confrontation",
                "belligerent": "confrontational",
                "belligerently": "confrontationally",
                "combative": "confrontational",
                "combatively": "confrontationally",
                "confrontational": "confrontational",
                "defensive": "defensive",
                "disheveled": "disheveled",
                "drug seeking": "seeking medication",
                "drug-seeking": "seeking medication",
                "exaggerate": "overstate",
                "exaggerates": "overstates",
                "exaggerating": "overstating",
                "historian": "patient",
                "malinger": "feign",
                "malingered": "feigned",
                "malingerer": "person feigning",
                "malingering": "feigning",
                "malingers": "feigns",
                "narcotic seeking": "seeking pain medication",
                "narcotic-seeking": "seeking pain medication",
                "poorly groomed": "disheveled",
                "poorly-groomed": "disheveled",
                "secondary gain": "secondary benefit",
                "uncooperative": "uncooperative",
                "unkempt": "disheveled",
                "unmotivated": "unmotivated",
                "unwilling": "unwilling",
                "unwillingly": "unwillingly"
            }
        }
    
    def get_replacement(self, word: str, category: str) -> str:
        """Get neutral replacement for a stigmatizing word"""
        word_lower = word.lower()
        if category in self.replacements:
            return self.replacements[category].get(word_lower, word)
        return word
    
    def rewrite_text(self, text: str, classifier: StigmaClassifier) -> Tuple[str, List[Dict]]:
        """
        Rewrite text to remove stigmatizing language
        
        Returns:
            Tuple of (rewritten_text, list_of_changes)
        """
        matches = classifier.detect_stigma(text)
        changes = []
        
        # Sort matches by position (reverse order to avoid index shifting)
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        rewritten_text = text
        for match in matches:
            replacement = self.get_replacement(match.original_text, match.category)
            
            # Only replace if it's actually different
            if replacement.lower() != match.original_text.lower():
                # Replace in text
                rewritten_text = (
                    rewritten_text[:match.start_pos] + 
                    replacement + 
                    rewritten_text[match.end_pos:]
                )
                
                # Record the change
                changes.append({
                    "original": match.original_text,
                    "replacement": replacement,
                    "category": match.category,
                    "position": match.start_pos,
                    "context": match.context
                })
        
        return rewritten_text, changes

class ClinicalNoteProcessor:
    """Main class for processing clinical notes"""
    
    def __init__(self, keywords_path: str = None):
        self.classifier = StigmaClassifier(keywords_path)
        self.rewriter = StigmaRewriter()
    
    def process_note(self, text: str) -> Dict:
        """
        Process a clinical note to detect and rewrite stigmatizing language
        
        Returns:
            Dictionary with original text, rewritten text, and analysis
        """
        # Detect stigma
        stigma_matches = self.classifier.detect_stigma(text)
        stigma_categories = self.classifier.get_stigma_categories(text)
        
        # Rewrite text
        rewritten_text, changes = self.rewriter.rewrite_text(text, self.classifier)
        
        return {
            "original_text": text,
            "rewritten_text": rewritten_text,
            "has_stigma": len(stigma_matches) > 0,
            "stigma_matches": [
                {
                    "text": match.original_text,
                    "category": match.category,
                    "position": match.start_pos,
                    "context": match.context
                }
                for match in stigma_matches
            ],
            "stigma_categories": stigma_categories,
            "changes_made": changes,
            "num_changes": len(changes)
        }
    
    def batch_process(self, texts: List[str]) -> List[Dict]:
        """Process multiple clinical notes"""
        return [self.process_note(text) for text in texts]
