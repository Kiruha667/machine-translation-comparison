"""
Data preprocessing module for machine translation
"""

import re
from collections import Counter
from typing import List, Tuple, Dict
import torch


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, max_length: int = 15, min_freq: int = 2):
        self.max_length = max_length
        self.min_freq = min_freq
        
        # Initialize vocabularies
        self.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.src_idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.tgt_idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text: lowercase and remove punctuation
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple whitespace tokenization
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def build_vocab(self, sentences: List[List[str]], 
                   vocab: Dict, idx2word: Dict) -> Tuple[Dict, Dict]:
        """
        Build vocabulary from sentences
        
        Args:
            sentences: List of tokenized sentences
            vocab: Vocabulary dictionary (word -> idx)
            idx2word: Reverse vocabulary (idx -> word)
            
        Returns:
            Updated vocab and idx2word dictionaries
        """
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        idx = len(vocab)
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in vocab:
                vocab[word] = idx
                idx2word[idx] = word
                idx += 1
        
        return vocab, idx2word
    
    def sentence_to_indices(self, sentence: List[str], 
                           vocab: Dict, add_sos_eos: bool = True) -> List[int]:
        """
        Convert sentence tokens to indices
        
        Args:
            sentence: List of tokens
            vocab: Vocabulary dictionary
            add_sos_eos: Whether to add start/end tokens
            
        Returns:
            List of indices
        """
        indices = [vocab.get(word, vocab['<unk>']) for word in sentence]
        
        if add_sos_eos:
            indices = [vocab['<sos>']] + indices + [vocab['<eos>']]
        
        return indices
    
    def indices_to_sentence(self, indices: List[int], 
                           idx2word: Dict, remove_special: bool = True) -> str:
        """
        Convert indices back to sentence
        
        Args:
            indices: List of indices
            idx2word: Reverse vocabulary
            remove_special: Remove special tokens
            
        Returns:
            Sentence string
        """
        words = [idx2word.get(idx, '<unk>') for idx in indices]
        
        if remove_special:
            words = [w for w in words if w not in ['<sos>', '<eos>', '<pad>']]
        
        return ' '.join(words)
    
    def load_and_preprocess(self, filepath: str) -> List[Tuple[List[str], List[str]]]:
        """
        Load and preprocess dataset from file
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            List of (source_tokens, target_tokens) pairs
        """
        print(f"Loading dataset from {filepath}...")
        pairs = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eng, fra = parts[0], parts[1]
                    pairs.append((eng, fra))
        
        print(f"Loaded {len(pairs):,} sentence pairs")
        
        # Clean and tokenize
        print("Cleaning and tokenizing...")
        processed_pairs = []
        
        for eng, fra in pairs:
            eng_clean = self.clean_text(eng)
            fra_clean = self.clean_text(fra)
            
            eng_tokens = self.tokenize(eng_clean)
            fra_tokens = self.tokenize(fra_clean)
            
            # Filter by length
            if (1 <= len(eng_tokens) <= self.max_length and 
                1 <= len(fra_tokens) <= self.max_length):
                processed_pairs.append((eng_tokens, fra_tokens))
        
        print(f"After filtering: {len(processed_pairs):,} pairs")
        
        # Build vocabularies
        print("Building vocabularies...")
        src_sentences = [pair[0] for pair in processed_pairs]
        tgt_sentences = [pair[1] for pair in processed_pairs]
        
        self.src_vocab, self.src_idx2word = self.build_vocab(
            src_sentences, self.src_vocab, self.src_idx2word
        )
        self.tgt_vocab, self.tgt_idx2word = self.build_vocab(
            tgt_sentences, self.tgt_vocab, self.tgt_idx2word
        )
        
        print(f"Source vocabulary size: {len(self.src_vocab):,}")
        print(f"Target vocabulary size: {len(self.tgt_vocab):,}")
        
        return processed_pairs
    
    def split_data(self, pairs: List, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple:
        """
        Split data into train/val/test sets
        
        Args:
            pairs: List of data pairs
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train, val, test) splits
        """
        train_size = int(train_ratio * len(pairs))
        val_size = int(val_ratio * len(pairs))
        
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_pairs):,} pairs")
        print(f"  Validation: {len(val_pairs):,} pairs")
        print(f"  Test: {len(test_pairs):,} pairs")
        
        return train_pairs, val_pairs, test_pairs
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """Get source and target vocabulary sizes"""
        return len(self.src_vocab), len(self.tgt_vocab)
    
    def save_vocabs(self, path: str):
        """Save vocabularies to file"""
        torch.save({
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'src_idx2word': self.src_idx2word,
            'tgt_idx2word': self.tgt_idx2word
        }, path)
        print(f"Vocabularies saved to {path}")
    
    def load_vocabs(self, path: str):
        """Load vocabularies from file"""
        checkpoint = torch.load(path)
        self.src_vocab = checkpoint['src_vocab']
        self.tgt_vocab = checkpoint['tgt_vocab']
        self.src_idx2word = checkpoint['src_idx2word']
        self.tgt_idx2word = checkpoint['tgt_idx2word']
        print(f"Vocabularies loaded from {path}")


def print_data_statistics(pairs: List[Tuple[List[str], List[str]]]):
    """Print statistics about the dataset"""
    src_lengths = [len(pair[0]) for pair in pairs]
    tgt_lengths = [len(pair[1]) for pair in pairs]
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total pairs: {len(pairs):,}")
    print(f"\nSource sentences:")
    print(f"  Avg length: {sum(src_lengths)/len(src_lengths):.2f} words")
    print(f"  Min length: {min(src_lengths)} words")
    print(f"  Max length: {max(src_lengths)} words")
    print(f"\nTarget sentences:")
    print(f"  Avg length: {sum(tgt_lengths)/len(tgt_lengths):.2f} words")
    print(f"  Min length: {min(tgt_lengths)} words")
    print(f"  Max length: {max(tgt_lengths)} words")
    print("="*80)


if __name__ == '__main__':
    # Test preprocessing
    preprocessor = DataPreprocessor(max_length=15)
    
    # Test cleaning
    test_text = "Hello, World! How are you?"
    cleaned = preprocessor.clean_text(test_text)
    tokens = preprocessor.tokenize(cleaned)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
