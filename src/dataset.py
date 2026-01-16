"""
PyTorch Dataset classes for machine translation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict


class TranslationDataset(Dataset):
    """Dataset class for translation pairs"""
    
    def __init__(self, pairs: List[Tuple[List[str], List[str]]], 
                 src_vocab: Dict, tgt_vocab: Dict, preprocessor):
        """
        Args:
            pairs: List of (source_tokens, target_tokens)
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            preprocessor: DataPreprocessor instance
        """
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.preprocessor = preprocessor
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        
        # Convert to indices
        src_indices = self.preprocessor.sentence_to_indices(src, self.src_vocab)
        tgt_indices = self.preprocessor.sentence_to_indices(tgt, self.tgt_vocab)
        
        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences in a batch
    
    Args:
        batch: List of (src_tensor, tgt_tensor) pairs
        
    Returns:
        Tuple of padded (src_batch, tgt_batch)
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch


def create_dataloaders(train_pairs: List, val_pairs: List, test_pairs: List,
                       src_vocab: Dict, tgt_vocab: Dict, preprocessor,
                       batch_size: int = 64, num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_pairs: Training data pairs
        val_pairs: Validation data pairs
        test_pairs: Test data pairs
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        preprocessor: DataPreprocessor instance
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, preprocessor)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab, preprocessor)
    test_dataset = TranslationDataset(test_pairs, src_vocab, tgt_vocab, preprocessor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset creation
    from preprocessing import DataPreprocessor
    
    # Sample data
    pairs = [
        (['hello', 'world'], ['bonjour', 'monde']),
        (['good', 'morning'], ['bon', 'matin']),
        (['thank', 'you'], ['merci'])
    ]
    
    preprocessor = DataPreprocessor()
    preprocessor.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'hello': 3, 'world': 4, 'good': 5, 'morning': 6, 'thank': 7, 'you': 8}
    preprocessor.tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'bonjour': 3, 'monde': 4, 'bon': 5, 'matin': 6, 'merci': 7}
    
    dataset = TranslationDataset(pairs, preprocessor.src_vocab, preprocessor.tgt_vocab, preprocessor)
    
    print(f"Dataset size: {len(dataset)}")
    src, tgt = dataset[0]
    print(f"Sample: src={src}, tgt={tgt}")
