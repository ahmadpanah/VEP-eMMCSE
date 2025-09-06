"""
Tests for VEP-eMMCSE main scheme implementation
"""

import pytest
from vep_emmcse.schemes.vep_emmcse import (
    VEP_eMMCSE, SystemParameters, DataSourceKeys,
    ClientKeys, SearchToken, SearchResult, create_vep_emmcse
)

@pytest.fixture
def scheme():
    """Create VEP-eMMCSE instance for tests"""
    return create_vep_emmcse()

@pytest.fixture
def initialized_scheme(scheme):
    """Create and initialize VEP-eMMCSE instance"""
    source_ids = ["source1", "source2"]
    scheme.setup(security_level=128, data_source_ids=source_ids)
    return scheme

@pytest.fixture
def test_client(initialized_scheme):
    """Create test client"""
    client_id = "client1"
    authorized_sources = ["source1"]
    return initialized_scheme.aggregate_key(client_id, authorized_sources)

def test_scheme_setup(scheme):
    """Test scheme initialization"""
    params = scheme.setup(security_level=128)
    assert isinstance(params, SystemParameters)
    assert params.security_level == 128
    assert len(params.master_secret_key) == 32
    assert params.public_verification_root is not None

def test_data_source_setup(initialized_scheme):
    """Test data source initialization"""
    source_id = "source1"
    assert source_id in initialized_scheme.data_sources
    
    ds_keys = initialized_scheme.data_sources[source_id]
    assert isinstance(ds_keys, DataSourceKeys)
    assert ds_keys.source_id == source_id
    assert len(ds_keys.prf_keys) > 0
    assert len(ds_keys.signature_keypair) == 2

def test_client_key_generation(initialized_scheme):
    """Test client key generation"""
    client_id = "test_client"
    authorized_sources = ["source1"]
    
    client_keys = initialized_scheme.aggregate_key(client_id, authorized_sources)
    assert isinstance(client_keys, ClientKeys)
    assert client_keys.client_id == client_id
    assert client_keys.access_rights == set(authorized_sources)
    assert len(client_keys.aggregated_key) > 0

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
    
    assert isinstance(token, SearchToken)
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
    
    assert isinstance(results, SearchResult)
    assert doc1_id in results.document_ids
    assert doc2_id not in results.document_ids
    assert len(results.merkle_proofs) > 0

def test_result_verification(initialized_scheme, test_client):
    """Test result verification"""
    # Add a document
    source_id = "source1"
    doc_id = "doc1"
    keywords = ["keyword1", "keyword2"]
    initialized_scheme.update_document(source_id, "add", doc_id, keywords)
    
    # Search
    query = "keyword1 AND keyword2"
    token = initialized_scheme.search_token_gen("client1", query)
    results = initialized_scheme.search(token)
    
    # Verify results
    verified_docs = initialized_scheme.verify_results("client1", results, query)
    assert verified_docs is not None
    assert doc_id in verified_docs

def test_client_revocation(initialized_scheme, test_client):
    """Test client revocation"""
    source_id = "source1"
    client_id = "client1"
    
    success = initialized_scheme.revoke_client(source_id, client_id)
    assert success
    
    # Verify client no longer has access
    client_keys = initialized_scheme.clients[client_id]
    assert source_id not in client_keys.access_rights

def test_system_stats(initialized_scheme):
    """Test system statistics"""
    # Add some test data
    source_id = "source1"
    for i in range(3):
        doc_id = f"doc{i}"
        keywords = [f"keyword{i}"]
        initialized_scheme.update_document(source_id, "add", doc_id, keywords)
        
    stats = initialized_scheme.get_system_stats()
    assert isinstance(stats, dict)
    assert stats['num_data_sources'] == 2
    assert stats['num_documents'] == 3
    assert stats['num_active_documents'] == 3
    assert stats['merkle_tree_info'] is not None
