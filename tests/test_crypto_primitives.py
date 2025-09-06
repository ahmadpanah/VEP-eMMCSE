"""
Tests for core cryptographic primitives
"""

import pytest
import os
import secrets
from vep_emmcse.core.crypto_primitives import (
    PostQuantumPRF, PostQuantumSignature, SymmetricEncryption,
    SecureHash, CryptographicAccumulator
)

@pytest.fixture
def prf():
    """Create PRF instance for tests"""
    return PostQuantumPRF()

@pytest.fixture
def signature():
    """Create signature scheme instance for tests"""
    return PostQuantumSignature()

@pytest.fixture
def encryption():
    """Create encryption instance for tests"""
    return SymmetricEncryption()

def test_prf_consistency(prf):
    """Test PRF outputs are consistent"""
    input_data = "test_input"
    output1 = prf.evaluate(input_data)
    output2 = prf.evaluate(input_data)
    assert output1 == output2
    assert len(output1) == 32  # SHA3-256 output size

def test_prf_different_inputs(prf):
    """Test PRF outputs differ for different inputs"""
    output1 = prf.evaluate("input1")
    output2 = prf.evaluate("input2")
    assert output1 != output2

def test_prf_key_stretching(prf):
    """Test PRF key stretching functionality"""
    input_data = "test_input"
    desired_length = 64
    output = prf.evaluate_with_length(input_data, desired_length)
    assert len(output) == desired_length

def test_signature_keygen(signature):
    """Test signature key generation"""
    public_key, secret_key = signature.generate_keypair()
    assert len(public_key) > 0
    assert len(secret_key) > 0
    assert public_key != secret_key

def test_signature_sign_verify(signature):
    """Test signature creation and verification"""
    message = "test message"
    public_key, secret_key = signature.generate_keypair()
    
    # Sign with secret key
    sig = signature.sign(message, secret_key)
    assert len(sig) > 0
    
    # Verify with public key
    assert signature.verify(message, sig, public_key)
    
    # Verify fails with wrong message
    assert not signature.verify("wrong message", sig, public_key)

def test_symmetric_encryption(encryption):
    """Test symmetric encryption and decryption"""
    plaintext = b"test message"
    associated_data = b"additional data"
    
    # Encrypt
    ciphertext, nonce = encryption.encrypt(plaintext, associated_data)
    assert len(ciphertext) > 0
    assert len(nonce) == 12  # GCM nonce size
    
    # Decrypt
    decrypted = encryption.decrypt(ciphertext, nonce, associated_data)
    assert decrypted == plaintext
    
    # Decryption fails with wrong nonce
    wrong_nonce = secrets.token_bytes(12)
    with pytest.raises(Exception):
        encryption.decrypt(ciphertext, wrong_nonce, associated_data)

def test_secure_hash():
    """Test secure hash functionality"""
    data = "test data"
    hash1 = SecureHash.hash(data)
    hash2 = SecureHash.hash(data)
    assert hash1 == hash2
    assert len(hash1) == 32  # SHA3-256 output size
    
    # Test multiple input hashing
    multi_hash = SecureHash.hash_multiple("input1", "input2", "input3")
    assert len(multi_hash) == 32

def test_cryptographic_accumulator():
    """Test cryptographic accumulator functionality"""
    accumulator = CryptographicAccumulator()
    
    # Add elements
    element1 = "test1"
    element2 = "test2"
    digest1 = accumulator.add_element(element1)
    digest2 = accumulator.add_element(element2)
    assert digest1 != digest2
    
    # Generate and verify proofs
    proof1 = accumulator.generate_membership_proof(element1)
    assert proof1 is not None
    assert accumulator.verify_membership(element1, proof1)
    
    # Verify fails for non-member
    proof_nonmember = accumulator.generate_membership_proof("nonmember")
    assert proof_nonmember is None
