import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

@dataclass
class BERTStigmaMatch:
    """Enhanced stigma match with BERT-based analysis"""
    original_text: str
    start_pos: int
    end_pos: int
    category: str
    confidence: float
    context: str
    bert_alternatives: List[Tuple[str, float]]  # (word, similarity_score)

class BERTEnhancedClassifier:
    """BERT-enhanced stigma detection and replacement system"""
    
    def __init__(self, keywords_path: str = None,
                 model_name: str = "bert-base-uncased"):
        """Initialize BERT model and load keywords"""
        if keywords_path is None:
            # Default to data directory relative to project root
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            keywords_path = str(project_root / "data" / "keywords.json")
        
        self.model_name = model_name
        
        # Load BERT model and tokenizer
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Load stigmatizing keywords
        with open(keywords_path, 'r') as f:
            self.keyword_dict = json.load(f)
        
        # Create regex patterns for detection
        self.patterns = {}
        for category, words in self.keyword_dict.items():
            pattern = r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        # Pre-compute embeddings for all stigmatizing words
        self.stigma_embeddings = {}
        self.compute_stigma_embeddings()
        
        # Load neutral alternatives database
        self.neutral_alternatives = self.load_neutral_alternatives()
        
    def compute_stigma_embeddings(self):
        """Pre-compute BERT embeddings for all stigmatizing words"""
        print("Computing embeddings for stigmatizing words...")
        
        all_stigma_words = []
        for category, words in self.keyword_dict.items():
            for word in words:
                all_stigma_words.append((word, category))
        
        # Get embeddings for all words
        embeddings = self.get_word_embeddings([word for word, _ in all_stigma_words])
        
        # Store embeddings with category information
        for i, (word, category) in enumerate(all_stigma_words):
            self.stigma_embeddings[word.lower()] = {
                'embedding': embeddings[i],
                'category': category
            }
        
        print(f"Computed embeddings for {len(all_stigma_words)} stigmatizing words")
    
    def get_word_embeddings(self, words: List[str]) -> np.ndarray:
        """Get BERT embeddings for a list of words"""
        embeddings = []
        
        for word in words:
            # Tokenize the word
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            
            # Get BERT output
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token embedding or average of all tokens
                word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(word_embedding)
        
        return np.array(embeddings)
    
    def load_neutral_alternatives(self) -> Dict[str, List[str]]:
        """Load database of neutral alternatives for each category"""
        return {
            "adamant": [
                "firm", "assertive", "determined", "resolute", "steadfast", 
                "persistent", "consistent", "clear", "direct", "straightforward"
            ],
            "compliance": [
                "declined", "did not accept", "chose not to", "elected not to",
                "was unable to", "had difficulty with", "struggled with",
                "follow", "adhere", "participate", "engage"  # For positive compliance terms
            ],
            "behavioral": [
                "agitated", "distressed", "concerned", "frustrated", "worried",
                "anxious", "upset", "troubled", "bothered", "uncomfortable",
                "confrontational", "defensive", "resistant"
            ],
            "appearance": [
                "disheveled", "unkept", "tired", "weary", "exhausted",
                "fatigued", "drained", "worn", "strained", "stressed"
            ],
            "substance_use": [
                "seeking medication", "requesting pain relief", "asking for treatment",
                "seeking help", "requesting assistance", "asking for support"
            ],
            "general": [
                "patient", "individual", "person", "client", "resident",
                "participant", "subject", "case", "client", "recipient"
            ]
        }
    
    def find_similar_neutral_alternatives(self, stigmatizing_word: str, 
                                        top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar neutral alternatives using BERT embeddings with quality filtering"""
        
        # Get embedding for the stigmatizing word
        stigma_embedding = self.get_word_embeddings([stigmatizing_word])[0]
        
        # Get category for the word
        category = self.stigma_embeddings.get(stigmatizing_word.lower(), {}).get('category', 'general')
        
        # Get neutral alternatives for this category
        neutral_words = self.neutral_alternatives.get(category, self.neutral_alternatives['general'])
        
        # Get embeddings for neutral alternatives
        neutral_embeddings = self.get_word_embeddings(neutral_words)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([stigma_embedding], neutral_embeddings)[0]
        
        # Create word-similarity pairs
        word_similarity_pairs = list(zip(neutral_words, similarities))
        
        # Apply quality filtering and prioritization
        filtered_pairs = self._filter_and_prioritize_alternatives(stigmatizing_word, word_similarity_pairs, category)
        
        return filtered_pairs[:top_k]
    
    def _filter_and_prioritize_alternatives(self, original_word: str, 
                                          word_similarity_pairs: List[Tuple[str, float]], 
                                          category: str) -> List[Tuple[str, float]]:
        """Filter and prioritize alternatives based on quality and clinical appropriateness"""
        
        # Define high-quality, proven replacements
        proven_replacements = {
            "adamant": ["firm", "determined", "resolute"],
            "refused": ["declined", "did not accept", "chose not to"],
            "historian": ["patient"],
            "noncompliant": ["non-adherent", "had difficulty with"],
            "aggressive": ["agitated", "distressed", "frustrated", "confrontational"],
            "agitated": ["agitated"],  # Keep as is - it's already neutral
            "uncooperative": ["uncooperative"],  # Keep as is - it's already neutral
            "combative": ["confrontational", "defensive"],
            "drug seeking": ["seeking medication", "requesting pain relief"],
            "poorly groomed": ["disheveled", "unkept"],
            "malingering": ["feigning", "exaggerating"],
            "uncooperative": ["uncooperative"],  # Keep as is
            "belligerent": ["confrontational", "defensive"],
            "exaggerating": ["overstating", "emphasizing"],
            "claimed": ["reported", "stated", "described"],
            "insisted": ["stated", "reported", "described"],
            "adherence": ["compliance"],
            "compliant": ["adherent"],
            "compliance": ["adherence"],
            "adherent": ["compliant"],
            "comply": ["follow", "adhere", "participate"]  # Prevent "declined"
        }
        
        # Boost scores for proven good replacements
        boosted_pairs = []
        for word, similarity in word_similarity_pairs:
            boosted_similarity = similarity
            
            # Boost proven replacements
            if original_word.lower() in proven_replacements:
                if word.lower() in proven_replacements[original_word.lower()]:
                    boosted_similarity += 0.2  # Significant boost for proven replacements
            
            # Penalize poor replacements
            poor_replacements = {
                "refused": ["adhere", "comply", "accept"],  # These change meaning completely
                "historian": ["person", "individual", "client"],  # Too generic
                "noncompliant": ["cooperate", "participate"],  # Changes meaning
                "comply": ["declined", "refused", "did not accept"],  # Changes meaning completely
                "claimed": ["persistent", "consistent"],  # Changes meaning
                "insisted": ["persistent", "consistent"],  # Changes meaning
            }
            
            if original_word.lower() in poor_replacements:
                if word.lower() in poor_replacements[original_word.lower()]:
                    boosted_similarity -= 0.3  # Penalty for poor replacements
            
            boosted_pairs.append((word, boosted_similarity))
        
        # Sort by boosted similarity
        boosted_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_pairs
    
    def detect_stigma_with_bert(self, text: str) -> List[BERTStigmaMatch]:
        """Detect stigmatizing language with BERT-enhanced analysis"""
        matches = []
        
        for category, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                # Get context
                start_ctx = max(0, match.start() - 50)
                end_ctx = min(len(text), match.end() + 50)
                context = text[start_ctx:end_ctx]
                
                # Find BERT-based alternatives
                original_word = match.group()
                alternatives = self.find_similar_neutral_alternatives(original_word)
                
                # Calculate confidence based on similarity to alternatives
                confidence = alternatives[0][1] if alternatives else 0.5
                
                bert_match = BERTStigmaMatch(
                    original_text=original_word,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    category=category,
                    confidence=confidence,
                    context=context,
                    bert_alternatives=alternatives
                )
                matches.append(bert_match)
        
        return matches
    
    def rewrite_text_with_bert(self, text: str) -> Tuple[str, List[Dict]]:
        """Rewrite text using BERT-based alternatives with quality control"""
        matches = self.detect_stigma_with_bert(text)
        changes = []
        
        # Sort matches by position (reverse order to avoid index shifting)
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        rewritten_text = text
        for match in matches:
            # Get the best alternative
            if match.bert_alternatives:
                best_alternative, similarity_score = match.bert_alternatives[0]
                
                # Quality control checks
                if self._is_appropriate_replacement(match.original_text, best_alternative, 
                                                   match.category, similarity_score, text, match.start_pos):
                    
                    # Replace in text
                    rewritten_text = (
                        rewritten_text[:match.start_pos] + 
                        best_alternative + 
                        rewritten_text[match.end_pos:]
                    )
                    
                    # Record the change
                    changes.append({
                        "original": match.original_text,
                        "replacement": best_alternative,
                        "category": match.category,
                        "position": match.start_pos,
                        "context": match.context,
                        "similarity_score": similarity_score,
                        "all_alternatives": match.bert_alternatives,
                        "quality_score": self._calculate_quality_score(match.original_text, best_alternative, similarity_score)
                    })
        
        return rewritten_text, changes
    
    def _is_appropriate_replacement(self, original: str, replacement: str, category: str, 
                                   similarity_score: float, full_text: str, position: int) -> bool:
        """Check if a replacement is appropriate and maintains clinical accuracy"""
        
        # 1. Similarity threshold (must be reasonably similar)
        if similarity_score < 0.6:
            return False
        
        # 2. Don't replace already neutral terms
        neutral_terms = ["agitated", "uncooperative", "patient", "individual", "person"]
        if original.lower() in neutral_terms:
            return False
        
        # 3. Category-specific quality checks
        if category == "compliance":
            # For compliance terms, ensure we're not changing meaning too drastically
            if original.lower() in ["refused", "declined"] and replacement.lower() in ["adhere", "comply"]:
                return False  # This would change meaning completely
            
            # Prevent "comply" from becoming "declined" (grammatically incorrect)
            if original.lower() == "comply" and replacement.lower() == "declined":
                return False
        
        # 4. Grammatical correctness check
        if not self._is_grammatically_correct(original, replacement, full_text, position):
            return False
        
        # 5. Clinical meaning preservation check
        if not self._preserves_clinical_meaning(original, replacement, category):
            return False
        
        # 6. Context appropriateness check
        if not self._is_contextually_appropriate(original, replacement, full_text, position):
            return False
        
        return True
    
    def _is_grammatically_correct(self, original: str, replacement: str, full_text: str, position: int) -> bool:
        """Check if replacement maintains grammatical correctness"""
        
        # Get surrounding context
        start_ctx = max(0, position - 20)
        end_ctx = min(len(full_text), position + len(original) + 20)
        context = full_text[start_ctx:end_ctx]
        
        # Check for common grammatical issues
        problematic_patterns = [
            # Verb tense mismatches
            (r'\b(was|were)\s+' + re.escape(original) + r'\b', r'\1 ' + replacement),
            (r'\b(is|are)\s+' + re.escape(original) + r'\b', r'\1 ' + replacement),
            
            # Article mismatches
            (r'\b(a|an)\s+' + re.escape(original) + r'\b', r'\1 ' + replacement),
            
            # Preposition mismatches
            (r'\b(to|with|for|of)\s+' + re.escape(original) + r'\b', r'\1 ' + replacement),
        ]
        
        # Simple heuristic: check if replacement makes sense in context
        test_context = context.replace(original, replacement)
        
        # Check for obvious grammatical errors
        if any(pattern in test_context.lower() for pattern in [
            "was adhere", "were adhere", "is adhere", "are adhere",
            "was comply", "were comply", "is comply", "are comply",
            "a patient", "an patient",  # Article issues
            "to declined", "with declined", "for declined",  # Preposition issues
            "declined to declined", "refused to declined",  # Double negative issues
        ]):
            return False
        
        return True
    
    def _preserves_clinical_meaning(self, original: str, replacement: str, category: str) -> bool:
        """Check if replacement preserves clinical meaning"""
        
        # Critical clinical terms that shouldn't be changed
        critical_terms = {
            "historian": "patient",  # Only allow this specific change
            "refused": "declined",   # Only allow this specific change
            "noncompliant": "non-adherent",  # Only allow this specific change
            "nonadherent": "non-adherent",  # Keep as is (just fix spelling)
        }
        
        # If it's a critical term, only allow specific replacements
        if original.lower() in critical_terms:
            return replacement.lower() == critical_terms[original.lower()]
        
        # For other terms, check if replacement maintains similar clinical meaning
        meaning_preserving = {
            "adamant": ["firm", "determined", "resolute"],
            "aggressive": ["agitated", "distressed", "frustrated"],
            "combative": ["confrontational", "defensive"],
            "drug seeking": ["seeking medication", "requesting pain relief"],
            "poorly groomed": ["disheveled", "unkept"],
            "malingering": ["feigning", "exaggerating"],
            "comply": ["follow", "adhere", "participate"],  # Don't allow "declined"
            "claimed": ["reported", "stated", "described"],
            "insisted": ["stated", "reported", "described"],
        }
        
        if original.lower() in meaning_preserving:
            return replacement.lower() in meaning_preserving[original.lower()]
        
        return True
    
    def _is_contextually_appropriate(self, original: str, replacement: str, full_text: str, position: int) -> bool:
        """Check if replacement is contextually appropriate"""
        
        # Get sentence context
        sentence_start = max(0, full_text.rfind('.', 0, position) + 1)
        sentence_end = full_text.find('.', position)
        if sentence_end == -1:
            sentence_end = len(full_text)
        
        sentence = full_text[sentence_start:sentence_end].strip()
        
        # Check for context-specific issues
        problematic_contexts = [
            # "historian" in medical context should become "patient"
            (r'\bhistorian\b', r'patient', r'poor\s+historian'),
            
            # "refused" should become "declined" in most contexts
            (r'\brefused\b', r'declined', r'refused\s+to\s+comply'),
            
            # "noncompliant" should become "non-adherent"
            (r'\bnoncompliant\b', r'non-adherent', r'noncompliant\s+with'),
        ]
        
        for pattern, replacement_pattern, context_pattern in problematic_contexts:
            if re.search(pattern, original, re.IGNORECASE):
                if re.search(context_pattern, sentence, re.IGNORECASE):
                    return replacement.lower() == replacement_pattern
        
        return True
    
    def _calculate_quality_score(self, original: str, replacement: str, similarity_score: float) -> float:
        """Calculate overall quality score for a replacement"""
        
        # Base score from similarity
        quality_score = similarity_score
        
        # Bonus for high-quality replacements
        high_quality_replacements = {
            "adamant": "firm",
            "refused": "declined", 
            "historian": "patient",
            "noncompliant": "non-adherent",
            "aggressive": "agitated",
            "combative": "confrontational",
            "drug seeking": "seeking medication",
            "poorly groomed": "disheveled",
            "malingering": "feigning"
        }
        
        if original.lower() in high_quality_replacements:
            if replacement.lower() == high_quality_replacements[original.lower()]:
                quality_score += 0.1  # Bonus for proven good replacements
        
        return min(quality_score, 1.0)  # Cap at 1.0

class BERTEnhancedProcessor:
    """Main interface for BERT-enhanced clinical note processing"""
    
    def __init__(self, keywords_path: str = "data/keywords.json"):
        self.classifier = BERTEnhancedClassifier(keywords_path)
    
    def process_note(self, text: str) -> Dict:
        """Process a clinical note with BERT-enhanced detection and rewriting"""
        
        # Detect stigma with BERT
        stigma_matches = self.classifier.detect_stigma_with_bert(text)
        stigma_categories = list(set(match.category for match in stigma_matches))
        
        # Rewrite text with BERT alternatives
        rewritten_text, changes = self.classifier.rewrite_text_with_bert(text)
        
        return {
            "original_text": text,
            "rewritten_text": rewritten_text,
            "has_stigma": len(stigma_matches) > 0,
            "stigma_matches": [
                {
                    "text": match.original_text,
                    "category": match.category,
                    "position": match.start_pos,
                    "context": match.context,
                    "confidence": match.confidence,
                    "alternatives": match.bert_alternatives
                }
                for match in stigma_matches
            ],
            "stigma_categories": stigma_categories,
            "changes_made": changes,
            "num_changes": len(changes),
            "bert_enhanced": True
        }
    
    def get_word_alternatives(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get BERT-based alternatives for a specific word"""
        return self.classifier.find_similar_neutral_alternatives(word, top_k)
    
    def analyze_word_similarity(self, word1: str, word2: str) -> float:
        """Analyze similarity between two words using BERT"""
        embeddings = self.classifier.get_word_embeddings([word1, word2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
