#!/usr/bin/env python3
"""
Dataset Handling Utilities for VEP-eMMCSE

This module provides tools for downloading, preprocessing, and managing
datasets used in VEP-eMMCSE experiments, including the Enron email corpus
and synthetic dataset generation.

"""

import os
import re
import email
import tarfile
import zipfile
import requests
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Represents a document with keywords for SSE experiments"""
    doc_id: str
    keywords: List[str]
    source: str
    metadata: Dict[str, str] = None
    

@dataclass
class DatasetStatistics:
    """Statistics about a processed dataset"""
    num_documents: int
    num_unique_keywords: int
    avg_keywords_per_doc: float
    min_keywords_per_doc: int
    max_keywords_per_doc: int
    keyword_frequency_distribution: Dict[str, int]
    zipf_parameter: Optional[float] = None


class EnronDatasetProcessor:
    """
    Processor for the Enron email corpus
    
    Downloads and preprocesses the Enron dataset for use in SSE experiments.
    Extracts keywords from email subjects and bodies using configurable filters.
    """
    
    def __init__(self, data_dir: str = "data/enron"):
        """
        Initialize Enron dataset processor
        
        Args:
            data_dir: Directory to store dataset files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and checksums
        self.dataset_url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
        self.expected_checksum = "0ca8f2bb8e7ea2ea51ab7a73a20b48fcf5473fc7d1d46ea6959b5d89f71b1abf"
        
        # Preprocessing parameters
        self.min_keyword_length = 3
        self.max_keyword_length = 50
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> set:
        """Load common English stop words"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'which',
            'who', 'whom', 'whose', 'all', 'any', 'some', 'no', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'now',
            'also', 'here', 'there', 'then', 'get', 'go', 'come', 'see', 'know',
            'take', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel',
            'try', 'leave', 'call'
        }
        return stop_words
        
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download the Enron dataset if not already present
        
        Args:
            force_download: Force re-download even if file exists
            
        Returns:
            True if download successful, False otherwise
        """
        archive_path = self.data_dir / "enron_mail_20150507.tar.gz"
        
        if archive_path.exists() and not force_download:
            logger.info("Enron dataset already exists")
            return True
            
        logger.info(f"Downloading Enron dataset from {self.dataset_url}")
        
        try:
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='')
                            
            print()  # New line after progress
            
            # Verify checksum
            if self._verify_checksum(archive_path):
                logger.info("Dataset downloaded and verified successfully")
                return True
            else:
                logger.error("Checksum verification failed")
                archive_path.unlink()  # Remove corrupted file
                return False
                
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
            
    def _verify_checksum(self, file_path: Path) -> bool:
        """Verify SHA-256 checksum of downloaded file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
                    
            computed_checksum = sha256_hash.hexdigest()
            return computed_checksum == self.expected_checksum
        except Exception:
            return False
            
    def extract_dataset(self) -> bool:
        """Extract the Enron dataset archive"""
        archive_path = self.data_dir / "enron_mail_20150507.tar.gz"
        extract_dir = self.data_dir / "maildir"
        
        if extract_dir.exists():
            logger.info("Dataset already extracted")
            return True
            
        if not archive_path.exists():
            logger.error("Dataset archive not found. Run download_dataset() first.")
            return False
            
        try:
            logger.info("Extracting Enron dataset...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
                
            logger.info("Dataset extracted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            return False
            
    def preprocess_emails(self, max_documents: Optional[int] = None) -> List[DocumentRecord]:
        """
        Preprocess Enron emails and extract keywords
        
        Args:
            max_documents: Maximum number of documents to process (None for all)
            
        Returns:
            List of DocumentRecord objects
        """
        maildir_path = self.data_dir / "maildir"
        
        if not maildir_path.exists():
            raise FileNotFoundError("Maildir not found. Run extract_dataset() first.")
            
        documents = []
        processed_count = 0
        
        logger.info("Processing Enron emails...")
        
        for user_dir in maildir_path.iterdir():
            if not user_dir.is_dir():
                continue
                
            user_name = user_dir.name
            
            # Process different mail folders
            for folder_name in ['sent', 'sent_items', 'inbox', '_sent_mail']:
                folder_path = user_dir / folder_name
                
                if not folder_path.exists():
                    continue
                    
                for email_file in folder_path.iterdir():
                    if max_documents and processed_count >= max_documents:
                        break
                        
                    if email_file.is_file():
                        try:
                            keywords = self._extract_keywords_from_email(email_file)
                            
                            if len(keywords) >= 3:  # Minimum keywords for meaningful document
                                doc_id = f"{user_name}_{folder_name}_{email_file.name}"
                                
                                doc_record = DocumentRecord(
                                    doc_id=doc_id,
                                    keywords=keywords,
                                    source="enron",
                                    metadata={
                                        'user': user_name,
                                        'folder': folder_name,
                                        'original_file': str(email_file)
                                    }
                                )
                                
                                documents.append(doc_record)
                                processed_count += 1
                                
                                if processed_count % 1000 == 0:
                                    logger.info(f"Processed {processed_count} documents...")
                                    
                        except Exception as e:
                            logger.debug(f"Failed to process {email_file}: {e}")
                            continue
                            
                if max_documents and processed_count >= max_documents:
                    break
                    
            if max_documents and processed_count >= max_documents:
                break
                
        logger.info(f"Preprocessing completed. {len(documents)} documents processed.")
        return documents
        
    def _extract_keywords_from_email(self, email_file: Path) -> List[str]:
        """Extract keywords from a single email file"""
        try:
            with open(email_file, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
                
            # Extract subject and body
            subject = msg.get('Subject', '').lower()
            
            # Get email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore').lower()
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore').lower()
                    
            # Combine subject and body
            text = subject + " " + body
            
            # Extract keywords using regex
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            # Filter keywords
            keywords = []
            for word in words:
                if (self.min_keyword_length <= len(word) <= self.max_keyword_length and
                    word not in self.stop_words and
                    not word.isdigit()):
                    keywords.append(word)
                    
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword not in seen:
                    seen.add(keyword)
                    unique_keywords.append(keyword)
                    
            return unique_keywords[:50]  # Limit to 50 keywords per document
            
        except Exception:
            return []


class SyntheticDatasetGenerator:
    """
    Generator for synthetic datasets with configurable properties
    
    Creates synthetic document-keyword datasets following Zipfian distributions
    to match natural language keyword frequency patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize synthetic dataset generator
        
        Args:
            seed: Random seed for reproducible generation
        """
        self.rng = np.random.RandomState(seed)
        
    def generate_dataset(self,
                        num_documents: int,
                        vocab_size: int,
                        zipf_param: float = 1.2,
                        min_keywords_per_doc: int = 5,
                        max_keywords_per_doc: int = 50) -> List[DocumentRecord]:
        """
        Generate synthetic dataset with Zipfian keyword distribution
        
        Args:
            num_documents: Number of documents to generate
            vocab_size: Size of keyword vocabulary
            zipf_param: Zipfian distribution parameter (higher = more skewed)
            min_keywords_per_doc: Minimum keywords per document
            max_keywords_per_doc: Maximum keywords per document
            
        Returns:
            List of DocumentRecord objects
        """
        logger.info(f"Generating synthetic dataset: {num_documents} docs, "
                   f"vocab size {vocab_size}, Zipf param {zipf_param}")
        
        # Generate vocabulary
        vocabulary = [f"keyword_{i:06d}" for i in range(vocab_size)]
        
        documents = []
        
        for doc_id in range(num_documents):
            # Random number of keywords per document
            num_keywords = self.rng.randint(min_keywords_per_doc, max_keywords_per_doc + 1)
            
            # Select keywords following Zipfian distribution
            # Using rejection sampling to approximate Zipfian distribution
            keyword_indices = []
            for _ in range(num_keywords * 2):  # Generate extra to account for duplicates
                # Simple Zipfian approximation using power law
                rand_val = self.rng.random()
                index = int(vocab_size * (1 - rand_val) ** (1.0 / zipf_param))
                index = min(index, vocab_size - 1)
                keyword_indices.append(index)
                
            # Remove duplicates and limit to desired count
            unique_indices = list(set(keyword_indices))
            selected_indices = unique_indices[:num_keywords]
            
            # If we don't have enough unique keywords, pad with random ones
            while len(selected_indices) < min_keywords_per_doc:
                rand_index = self.rng.randint(0, vocab_size)
                if rand_index not in selected_indices:
                    selected_indices.append(rand_index)
                    
            keywords = [vocabulary[i] for i in selected_indices]
            
            doc_record = DocumentRecord(
                doc_id=f"synthetic_doc_{doc_id:08d}",
                keywords=keywords,
                source="synthetic",
                metadata={
                    'generation_params': {
                        'zipf_param': zipf_param,
                        'vocab_size': vocab_size
                    }
                }
            )
            
            documents.append(doc_record)
            
            if (doc_id + 1) % 10000 == 0:
                logger.info(f"Generated {doc_id + 1}/{num_documents} documents")
                
        logger.info("Synthetic dataset generation completed")
        return documents
        
    def generate_scalability_datasets(self) -> Dict[str, List[DocumentRecord]]:
        """Generate multiple datasets for scalability testing"""
        configurations = {
            'small': {'docs': 1000, 'vocab': 500},
            'medium': {'docs': 10000, 'vocab': 2000},
            'large': {'docs': 100000, 'vocab': 10000},
            'xlarge': {'docs': 1000000, 'vocab': 50000}
        }
        
        datasets = {}
        for name, config in configurations.items():
            logger.info(f"Generating {name} dataset...")
            datasets[name] = self.generate_dataset(
                num_documents=config['docs'],
                vocab_size=config['vocab']
            )
            
        return datasets


class DatasetAnalyzer:
    """
    Analyzer for computing dataset statistics and properties
    """
    
    @staticmethod
    def analyze_dataset(documents: List[DocumentRecord]) -> DatasetStatistics:
        """
        Analyze dataset and compute comprehensive statistics
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            DatasetStatistics object with computed metrics
        """
        if not documents:
            raise ValueError("Cannot analyze empty dataset")
            
        logger.info(f"Analyzing dataset with {len(documents)} documents")
        
        # Basic statistics
        num_documents = len(documents)
        all_keywords = []
        keywords_per_doc = []
        
        for doc in documents:
            all_keywords.extend(doc.keywords)
            keywords_per_doc.append(len(doc.keywords))
            
        unique_keywords = set(all_keywords)
        num_unique_keywords = len(unique_keywords)
        
        # Keywords per document statistics
        avg_keywords_per_doc = np.mean(keywords_per_doc)
        min_keywords_per_doc = min(keywords_per_doc)
        max_keywords_per_doc = max(keywords_per_doc)
        
        # Keyword frequency distribution
        keyword_freq = Counter(all_keywords)
        
        # Estimate Zipfian parameter
        zipf_param = DatasetAnalyzer._estimate_zipf_parameter(keyword_freq)
        
        statistics = DatasetStatistics(
            num_documents=num_documents,
            num_unique_keywords=num_unique_keywords,
            avg_keywords_per_doc=avg_keywords_per_doc,
            min_keywords_per_doc=min_keywords_per_doc,
            max_keywords_per_doc=max_keywords_per_doc,
            keyword_frequency_distribution=dict(keyword_freq.most_common(100)),
            zipf_parameter=zipf_param
        )
        
        logger.info(f"Dataset analysis completed: {num_documents} docs, "
                   f"{num_unique_keywords} unique keywords, "
                   f"avg {avg_keywords_per_doc:.1f} keywords/doc")
        
        return statistics
        
    @staticmethod
    def _estimate_zipf_parameter(keyword_freq: Counter) -> float:
        """Estimate Zipfian distribution parameter using rank-frequency data"""
        try:
            frequencies = sorted(keyword_freq.values(), reverse=True)
            ranks = np.arange(1, len(frequencies) + 1)
            
            # Use log-log linear regression to estimate Zipf parameter
            log_ranks = np.log(ranks)
            log_freqs = np.log(frequencies)
            
            # Filter out zero frequencies (shouldn't happen, but be safe)
            valid_indices = log_freqs != -np.inf
            
            if np.sum(valid_indices) < 2:
                return 1.0  # Default value
                
            # Linear regression: log(freq) = -alpha * log(rank) + log(C)
            coeffs = np.polyfit(log_ranks[valid_indices], log_freqs[valid_indices], 1)
            alpha = -coeffs[0]  # Zipf parameter is negative of slope
            
            return max(0.1, min(3.0, alpha))  # Clamp to reasonable range
        except Exception:
            return 1.0  # Default Zipf parameter


class DatasetManager:
    """
    High-level manager for dataset operations
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize dataset manager
        
        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.enron_processor = EnronDatasetProcessor(self.base_dir / "enron")
        self.synthetic_generator = SyntheticDatasetGenerator()
        
    def prepare_enron_dataset(self, max_documents: Optional[int] = None,
                            force_download: bool = False) -> List[DocumentRecord]:
        """
        Download, extract, and preprocess Enron dataset
        
        Args:
            max_documents: Maximum documents to process (None for all)
            force_download: Force re-download of dataset
            
        Returns:
            List of processed documents
        """
        # Check if cached processed data exists
        cache_file = self.base_dir / "enron" / "processed_documents.json"
        
        if cache_file.exists() and not force_download:
            logger.info("Loading cached Enron dataset...")
            return self._load_documents_from_cache(cache_file)
            
        # Download and process dataset
        if not self.enron_processor.download_dataset(force_download):
            raise RuntimeError("Failed to download Enron dataset")
            
        if not self.enron_processor.extract_dataset():
            raise RuntimeError("Failed to extract Enron dataset")
            
        documents = self.enron_processor.preprocess_emails(max_documents)
        
        # Cache processed documents
        self._save_documents_to_cache(documents, cache_file)
        
        return documents
        
    def prepare_synthetic_dataset(self, config_name: str = "medium") -> List[DocumentRecord]:
        """
        Generate or load cached synthetic dataset
        
        Args:
            config_name: Configuration name (small, medium, large, xlarge)
            
        Returns:
            List of synthetic documents
        """
        cache_file = self.base_dir / "synthetic" / f"{config_name}_dataset.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        if cache_file.exists():
            logger.info(f"Loading cached synthetic dataset: {config_name}")
            return self._load_documents_from_cache(cache_file)
            
        # Generate new dataset
        configs = {
            'small': {'docs': 1000, 'vocab': 500},
            'medium': {'docs': 10000, 'vocab': 2000}, 
            'large': {'docs': 100000, 'vocab': 10000},
            'xlarge': {'docs': 1000000, 'vocab': 50000}
        }
        
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}")
            
        config = configs[config_name]
        documents = self.synthetic_generator.generate_dataset(**config)
        
        # Cache generated documents
        self._save_documents_to_cache(documents, cache_file)
        
        return documents
        
    def _save_documents_to_cache(self, documents: List[DocumentRecord], cache_file: Path):
        """Save documents to JSON cache file"""
        cache_data = []
        for doc in documents:
            doc_dict = {
                'doc_id': doc.doc_id,
                'keywords': doc.keywords,
                'source': doc.source,
                'metadata': doc.metadata or {}
            }
            cache_data.append(doc_dict)
            
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
            
        logger.info(f"Cached {len(documents)} documents to {cache_file}")
        
    def _load_documents_from_cache(self, cache_file: Path) -> List[DocumentRecord]:
        """Load documents from JSON cache file"""
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        documents = []
        for doc_dict in cache_data:
            doc = DocumentRecord(
                doc_id=doc_dict['doc_id'],
                keywords=doc_dict['keywords'],
                source=doc_dict['source'],
                metadata=doc_dict.get('metadata')
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} documents from cache")
        return documents


def main():
    """Example usage of dataset handling utilities"""
    manager = DatasetManager()
    
    # Prepare Enron dataset (small sample for testing)
    print("Preparing Enron dataset...")
    enron_docs = manager.prepare_enron_dataset(max_documents=1000)
    
    # Analyze Enron dataset
    enron_stats = DatasetAnalyzer.analyze_dataset(enron_docs)
    print(f"Enron dataset: {enron_stats.num_documents} docs, "
          f"{enron_stats.num_unique_keywords} keywords")
    
    # Prepare synthetic dataset
    print("Preparing synthetic dataset...")
    synthetic_docs = manager.prepare_synthetic_dataset('small')
    
    # Analyze synthetic dataset
    synthetic_stats = DatasetAnalyzer.analyze_dataset(synthetic_docs)
    print(f"Synthetic dataset: {synthetic_stats.num_documents} docs, "
          f"{synthetic_stats.num_unique_keywords} keywords")
    

if __name__ == "__main__":
    main()