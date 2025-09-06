"""
Tests for eMMCSE-PQ baseline scheme
"""

import pytest
from vep_emmcse.schemes.emmcse_baseline import (
    eMMCSE_PQ, BaseSystemParameters, BaseDataSourceKeys,
    BaseClientKeys, BaseSearchToken, BaseSearchResult,
    create_emmcse_pq
)

@pytest.fixture
def scheme():
    """Create eMMCSE-PQ instance for tests"""
    return create_emmcse_pq()

@pytest.fixture
def initialized_scheme(scheme):
    """Create and initialize eMMCSE-PQ instance"""
    source_ids = ["source1", "source2"]
    scheme.setup(security_level=128, data_source_ids=source_ids)
    return scheme

@pytest.fixture
def test_client(initialized_scheme):
    """Create test client"""
    client_id = "client1"
    authorized_sources = ["source1"]
    return initialized_scheme.generate_client_key(client_id, authorized_sources)

def test_scheme_setup(scheme):
    """Test scheme initialization"""
    params = scheme.setup(security_level=128)
    assert isinstance(params, BaseSystemParameters)
    assert params.security_level == 128
    assert len(params.master_secret_key) == 32
    assert params.system_id is not None

def test_data_source_setup(initialized_scheme):
    """Test data source initialization"""
    source_id = "source1"
    assert source_id in initialized_scheme.data_sources
    
    ds_keys = initialized_scheme.data_sources[source_id]
    assert isinstance(ds_keys, BaseDataSourceKeys)
    assert ds_keys.source_id == source_id
    assert len(ds_keys.prf_key) > 0
    assert len(ds_keys.signature_keypair) == 2

def test_client_key_generation(initialized_scheme):
    """Test client key generation"""
    client_id = "test_client"
    authorized_sources = ["source1"]
    
    client_keys = initialized_scheme.generate_client_key(client_id, authorized_sources)
    assert isinstance(client_keys, BaseClientKeys)
    assert client_keys.client_id == client_id
    assert client_keys.authorized_sources == set(authorized_sources)
    assert len(client_keys.access_key) > 0

def test_document_addition(initialized_scheme):
    """Test document addition"""
    source_id = "source1"
    doc_id = "doc1"
    keywords = ["keyword1", "keyword2", "keyword3"]
    
    success = initialized_scheme.update_document(source_id, "add", doc_id, keywords)
    assert success
    assert doc_id in initialized_scheme.encrypted_database
    
    doc_entry = initialized_scheme.encrypted_database[doc_id]
    assert doc_entry['document_id'] == doc_id
    assert doc_entry['source_id'] == source_id
    assert doc_entry['keywords'] == keywords
    assert doc_entry['operation'] == 'add'

def test_document_deletion(initialized_scheme):
    """Test document deletion"""
    # First add a document
    source_id = "source1"
    doc_id = "doc_to_delete"
    keywords = ["keyword1"]
    initialized_scheme.update_document(source_id, "add", doc_id, keywords)
    
    # Then delete it
    success = initialized_scheme.update_document(source_id, "delete", doc_id, keywords)
    assert success
    assert initialized_scheme.encrypted_database[doc_id]['operation'] == 'delete'
    assert initialized_scheme.encrypted_database[doc_id]['deleted'] == True

def test_search_token_generation(initialized_scheme, test_client):
    """Test search token generation"""
    query = "(keyword1 AND keyword2) OR keyword3"
    token = initialized_scheme.search_token_gen("client1", query)
    
    assert isinstance(token, BaseSearchToken)
    assert len(token.query_dnf) == 2  # Two clauses
    assert len(token.encrypted_tokens) == 2
    assert token.token_id is not None

def test_search_execution(initialized_scheme, test_client):
    """Test search execution"""
    # Add test documents
    source_id = "source1"
    doc1_id = "doc1"
    doc2_id = "doc2"
    
    initialized_scheme.update_document(source_id, "add", doc1_id, ["keyword1", "keyword2"])
    initialized_scheme.update_document(source_id, "add", doc2_id, ["keyword1", "keyword3"])
    
    # Generate and execute search
    query = "keyword1 AND keyword2"
    token = initialized_scheme.search_token_gen("client1", query)
    results = initialized_scheme.search(token)
    
    assert isinstance(results, BaseSearchResult)
    assert doc1_id in results.document_ids
    assert doc2_id not in results.document_ids
    assert results.metadata is not None

def test_query_parsing(initialized_scheme):
    """Test Boolean query parsing"""
    # Single keyword
    query1 = "keyword1"
    dnf1 = initialized_scheme._parse_query(query1)
    assert len(dnf1) == 1
    assert len(dnf1[0]) == 1
    assert dnf1[0][0] == "keyword1"
    
    # AND query
    query2 = "keyword1 AND keyword2"
    dnf2 = initialized_scheme._parse_query(query2)
    assert len(dnf2) == 1
    assert len(dnf2[0]) == 2
    assert set(dnf2[0]) == {"keyword1", "keyword2"}
    
    # OR query
    query3 = "keyword1 OR keyword2"
    dnf3 = initialized_scheme._parse_query(query3)
    assert len(dnf3) == 2
    assert len(dnf3[0]) == 1
    assert len(dnf3[1]) == 1
    
    # Complex query
    query4 = "(keyword1 AND keyword2) OR keyword3"
    dnf4 = initialized_scheme._parse_query(query4)
    assert len(dnf4) == 2
    assert len(dnf4[0]) == 2
    assert len(dnf4[1]) == 1

def test_clause_evaluation(initialized_scheme):
    """Test conjunctive clause evaluation"""
    # Add test documents
    source_id = "source1"
    doc1_id = "doc1"
    doc2_id = "doc2"
    
    initialized_scheme.update_document(source_id, "add", doc1_id, ["kw1", "kw2", "kw3"])
    initialized_scheme.update_document(source_id, "add", doc2_id, ["kw1", "kw2"])
    
    # Test single keyword clause
    matches1 = initialized_scheme._evaluate_clause(["kw1"], b"token")
    assert doc1_id in matches1
    assert doc2_id in matches1
    
    # Test AND clause
    matches2 = initialized_scheme._evaluate_clause(["kw1", "kw2"], b"token")
    assert doc1_id in matches2
    assert doc2_id in matches2
    
    # Test clause with non-matching keyword
    matches3 = initialized_scheme._evaluate_clause(["kw1", "kw4"], b"token")
    assert not matches3

def test_error_handling(initialized_scheme):
    """Test error handling"""
    # Test invalid source ID
    with pytest.raises(ValueError):
        initialized_scheme.update_document("invalid_source", "add", "doc1", ["kw1"])
    
    # Test invalid client ID
    with pytest.raises(ValueError):
        initialized_scheme.search_token_gen("invalid_client", "keyword1")
    
    # Test invalid operation
    with pytest.raises(ValueError):
        initialized_scheme.update_document("source1", "invalid_op", "doc1", ["kw1"])
