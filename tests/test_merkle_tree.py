"""
Tests for Merkle tree implementation
"""

import pytest
from vep_emmcse.core.merkle_tree import MerkleHashTree, MerkleNode, MerkleProof

@pytest.fixture
def merkle_tree():
    """Create Merkle tree instance for tests"""
    return MerkleHashTree()

@pytest.fixture
def sample_data():
    """Sample data for tree construction"""
    return [f"data_{i}".encode() for i in range(8)]

def test_empty_tree(merkle_tree):
    """Test empty tree construction"""
    root_hash = merkle_tree.build_tree([])
    assert root_hash is not None
    assert merkle_tree.get_root_hash() == root_hash

def test_single_node_tree(merkle_tree):
    """Test tree with single leaf node"""
    data = [b"single_node"]
    root_hash = merkle_tree.build_tree(data)
    assert root_hash is not None
    assert len(merkle_tree.leaves) == 1

def test_balanced_tree(merkle_tree, sample_data):
    """Test balanced tree construction"""
    root_hash = merkle_tree.build_tree(sample_data)
    assert root_hash is not None
    assert len(merkle_tree.leaves) == len(sample_data)
    
    # Verify tree height
    assert merkle_tree.tree_height == 4  # log2(8) + 1

def test_proof_generation_verification(merkle_tree, sample_data):
    """Test proof generation and verification"""
    merkle_tree.build_tree(sample_data)
    
    # Generate proof for first leaf
    proof = merkle_tree.generate_proof(0)
    assert proof is not None
    assert isinstance(proof, MerkleProof)
    
    # Verify the proof
    assert merkle_tree.verify_proof(proof, sample_data[0])
    
    # Verify fails with wrong data
    assert not merkle_tree.verify_proof(proof, b"wrong_data")

def test_tree_updates(merkle_tree):
    """Test dynamic tree updates"""
    initial_data = [b"data1", b"data2"]
    merkle_tree.build_tree(initial_data)
    initial_root = merkle_tree.get_root_hash()
    
    # Add new leaf
    leaf_idx, new_root = merkle_tree.add_leaf(b"data3")
    assert new_root != initial_root
    assert leaf_idx == 2
    
    # Verify new proof
    proof = merkle_tree.generate_proof(leaf_idx)
    assert merkle_tree.verify_proof(proof, b"data3")

def test_batch_addition(merkle_tree):
    """Test batch addition of leaves"""
    initial_data = [b"data1"]
    merkle_tree.build_tree(initial_data)
    
    # Add batch of leaves
    new_data = [b"data2", b"data3", b"data4"]
    new_root = merkle_tree.batch_add_leaves(new_data)
    assert new_root is not None
    
    # Verify all leaves are present
    assert len(merkle_tree.leaves) == 4

def test_proof_serialization(merkle_tree, sample_data):
    """Test proof serialization and deserialization"""
    merkle_tree.build_tree(sample_data)
    original_proof = merkle_tree.generate_proof(0)
    
    # Serialize
    json_str = merkle_tree.serialize_proof(original_proof)
    assert isinstance(json_str, str)
    
    # Deserialize
    restored_proof = merkle_tree.deserialize_proof(json_str)
    assert isinstance(restored_proof, MerkleProof)
    assert restored_proof.leaf_index == original_proof.leaf_index
    assert restored_proof.leaf_hash == original_proof.leaf_hash
    assert restored_proof.root_hash == original_proof.root_hash

def test_invalid_proof_indices(merkle_tree, sample_data):
    """Test proof generation with invalid indices"""
    merkle_tree.build_tree(sample_data)
    
    # Test negative index
    assert merkle_tree.generate_proof(-1) is None
    
    # Test out of bounds index
    assert merkle_tree.generate_proof(len(sample_data)) is None

def test_tree_info(merkle_tree, sample_data):
    """Test tree information retrieval"""
    merkle_tree.build_tree(sample_data)
    info = merkle_tree.get_tree_info()
    
    assert info['num_leaves'] == len(sample_data)
    assert info['tree_height'] == 4  # log2(8) + 1
    assert not info['is_empty']
    assert info['root_hash'] is not None
