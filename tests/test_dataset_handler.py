"""
Tests for dataset handling utilities
"""

import pytest
import os
import tempfile
import json
from vep_emmcse.utils.dataset_handler import (
    DatasetManager, EnronDatasetProcessor, SyntheticDatasetGenerator,
    DatasetAnalyzer, DocumentRecord, DatasetStatistics
)

@pytest.fixture
def dataset_manager():
    """Create dataset manager with temporary directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DatasetManager(base_dir=temp_dir)
        yield manager

@pytest.fixture
def synthetic_generator():
    """Create synthetic dataset generator"""
    return SyntheticDatasetGenerator(seed=42)

def test_synthetic_dataset_generation(synthetic_generator):
    """Test synthetic dataset generation"""
    num_docs = 100
    vocab_size = 50
    
    documents = synthetic_generator.generate_dataset(
        num_documents=num_docs,
        vocab_size=vocab_size
    )
    
    assert len(documents) == num_docs
    for doc in documents:
        assert isinstance(doc, DocumentRecord)
        assert doc.doc_id.startswith("synthetic_doc_")
        assert len(doc.keywords) > 0
        assert doc.source == "synthetic"

def test_scalability_datasets(synthetic_generator):
    """Test generation of multiple dataset sizes"""
    datasets = synthetic_generator.generate_scalability_datasets()
    
    assert 'small' in datasets
    assert 'medium' in datasets
    assert 'large' in datasets
    assert 'xlarge' in datasets
    
    # Verify increasing sizes
    sizes = [len(ds) for ds in datasets.values()]
    assert all(sizes[i] < sizes[i+1] for i in range(len(sizes)-1))

def test_dataset_analysis():
    """Test dataset analysis functionality"""
    # Create sample documents
    docs = [
        DocumentRecord(
            doc_id=f"doc_{i}",
            keywords=[f"keyword_{j}" for j in range(5)],
            source="test",
            metadata={"type": "test"}
        )
        for i in range(10)
    ]
    
    stats = DatasetAnalyzer.analyze_dataset(docs)
    assert isinstance(stats, DatasetStatistics)
    assert stats.num_documents == 10
    assert stats.num_unique_keywords == 5
    assert stats.avg_keywords_per_doc == 5.0
    assert stats.zipf_parameter is not None

def test_dataset_caching(dataset_manager):
    """Test dataset caching functionality"""
    # Generate small synthetic dataset
    docs = dataset_manager.prepare_synthetic_dataset('small')
    
    # Verify cache file exists
    cache_file = os.path.join(dataset_manager.base_dir, "synthetic", "small_dataset.json")
    assert os.path.exists(cache_file)
    
    # Load from cache and verify
    cached_docs = dataset_manager.prepare_synthetic_dataset('small')
    assert len(cached_docs) == len(docs)
    assert all(isinstance(doc, DocumentRecord) for doc in cached_docs)

def test_document_record_handling():
    """Test DocumentRecord creation and manipulation"""
    doc = DocumentRecord(
        doc_id="test_doc",
        keywords=["test", "keywords"],
        source="test_source",
        metadata={"type": "test"}
    )
    
    assert doc.doc_id == "test_doc"
    assert len(doc.keywords) == 2
    assert doc.source == "test_source"
    assert doc.metadata["type"] == "test"

def test_enron_processor_initialization():
    """Test Enron dataset processor initialization"""
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = EnronDatasetProcessor(data_dir=temp_dir)
        assert processor.data_dir.exists()
        assert processor.min_keyword_length > 0
        assert processor.max_keyword_length > processor.min_keyword_length
        assert len(processor.stop_words) > 0

def test_keyword_extraction():
    """Test keyword extraction from text"""
    processor = EnronDatasetProcessor()
    
    # Test with email content
    email_content = """
    Subject: Test Email
    
    This is a test email containing some keywords and stop words.
    Important business meeting scheduled for tomorrow.
    Please review the attached documents.
    """
    
    keywords = processor._extract_keywords_from_email(email_content)
    assert len(keywords) > 0
    assert "test" in keywords
    assert "email" in keywords
    assert "the" not in keywords  # Stop word
    assert "a" not in keywords    # Stop word

def test_dataset_save_load(dataset_manager):
    """Test saving and loading datasets"""
    # Generate sample data
    docs = [
        DocumentRecord(
            doc_id=f"doc_{i}",
            keywords=[f"kw_{i}"],
            source="test",
            metadata={"test": True}
        )
        for i in range(5)
    ]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        dataset_manager._save_documents_to_cache(docs, tmp.name)
        
        # Load and verify
        loaded_docs = dataset_manager._load_documents_from_cache(tmp.name)
        
        assert len(loaded_docs) == len(docs)
        for orig, loaded in zip(docs, loaded_docs):
            assert orig.doc_id == loaded.doc_id
            assert orig.keywords == loaded.keywords
            assert orig.source == loaded.source
            
        os.unlink(tmp.name)

def test_error_handling():
    """Test error handling in dataset operations"""
    # Test analyzer with empty dataset
    with pytest.raises(ValueError):
        DatasetAnalyzer.analyze_dataset([])
    
    # Test invalid dataset configuration
    with pytest.raises(ValueError):
        DatasetManager().prepare_synthetic_dataset('invalid_size')
    
    # Test missing cache file
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DatasetManager(base_dir=temp_dir)
        docs = manager._load_documents_from_cache("nonexistent.json")
        assert len(docs) == 0
