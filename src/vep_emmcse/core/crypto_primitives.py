#!/usr/bin/env python3
"""
Post-Quantum Cryptographic Primitives for VEP-eMMCSE

This module implements the core cryptographic building blocks required for
the VEP-eMMCSE framework, including post-quantum PRFs, signatures, and
other security primitives.

Author: VEP-eMMCSE Research Team
License: MIT
"""

import os
import hashlib
from typing import Tuple, Optional, Union, Dict, Any
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidSignature
import secrets

try:
    import pqcrypto.sign.dilithium3 as dilithium
    PQC_AVAILABLE = True
except ImportError:
    print("Warning: pqcrypto not available. Using mock implementations for development.")
    PQC_AVAILABLE = False


class PostQuantumPRF:
    """
    Post-Quantum Pseudorandom Function based on HMAC-SHA3-256
    
    This provides a quantum-resistant PRF construction suitable for
    generating tokens and keys in the VEP-eMMCSE framework.
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize PQ-PRF with a secret key
        
        Args:
            key: 32-byte secret key. If None, generates a random key.
        """
        if key is None:
            key = secrets.token_bytes(32)
        elif len(key) != 32:
            raise ValueError("Key must be exactly 32 bytes")
            
        self.key = key
        
    def evaluate(self, input_data: Union[str, bytes]) -> bytes:
        """
        Evaluate the PRF on input data
        
        Args:
            input_data: Input to the PRF (string or bytes)
            
        Returns:
            32-byte PRF output
        """
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
            
        h = hmac.HMAC(self.key, hashes.SHA3_256())
        h.update(input_data)
        return h.finalize()
        
    def evaluate_with_length(self, input_data: Union[str, bytes], 
                            output_length: int) -> bytes:
        """
        Evaluate PRF with specified output length (using key stretching)
        
        Args:
            input_data: Input to the PRF
            output_length: Desired output length in bytes
            
        Returns:
            PRF output of specified length
        """
        if output_length <= 32:
            return self.evaluate(input_data)[:output_length]
            
        # For longer outputs, use iterative approach
        output = b''
        counter = 0
        
        while len(output) < output_length:
            counter_bytes = counter.to_bytes(4, 'big')
            if isinstance(input_data, str):
                input_with_counter = input_data.encode('utf-8') + counter_bytes
            else:
                input_with_counter = input_data + counter_bytes
                
            chunk = self.evaluate(input_with_counter)
            output += chunk
            counter += 1
            
        return output[:output_length]


class PostQuantumSignature:
    """
    Post-Quantum Digital Signature using CRYSTALS-Dilithium3
    
    Provides quantum-resistant digital signatures for authenticating
    updates and other critical operations in VEP-eMMCSE.
    """
    
    def __init__(self):
        """Initialize the signature scheme"""
        self.public_key: Optional[bytes] = None
        self.secret_key: Optional[bytes] = None
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new key pair
        
        Returns:
            Tuple of (public_key, secret_key)
        """
        if PQC_AVAILABLE:
            self.public_key, self.secret_key = dilithium.keypair()
        else:
            # Mock implementation for development/testing
            self.secret_key = secrets.token_bytes(4000)  # Approximate Dilithium3 secret key size
            self.public_key = secrets.token_bytes(1952)  # Approximate Dilithium3 public key size
            
        return self.public_key, self.secret_key
        
    def sign(self, message: Union[str, bytes], secret_key: Optional[bytes] = None) -> bytes:
        """
        Sign a message
        
        Args:
            message: Message to sign
            secret_key: Secret key to use (uses instance key if None)
            
        Returns:
            Signature bytes
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
            
        key_to_use = secret_key if secret_key is not None else self.secret_key
        if key_to_use is None:
            raise ValueError("No secret key available")
            
        if PQC_AVAILABLE:
            return dilithium.sign(message, key_to_use)
        else:
            # Mock implementation - DO NOT USE IN PRODUCTION
            h = hashlib.sha3_256()
            h.update(key_to_use[:32])  # Use first 32 bytes as "key"
            h.update(message)
            # Return a fixed-size mock signature
            return h.digest() + secrets.token_bytes(3293 - 32)
            
    def verify(self, message: Union[str, bytes], signature: bytes, 
              public_key: Optional[bytes] = None) -> bool:
        """
        Verify a signature
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Public key to use (uses instance key if None)
            
        Returns:
            True if signature is valid, False otherwise
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
            
        key_to_use = public_key if public_key is not None else self.public_key
        if key_to_use is None:
            raise ValueError("No public key available")
            
        if PQC_AVAILABLE:
            try:
                dilithium.verify(message, signature, key_to_use)
                return True
            except:
                return False
        else:
            # Mock implementation - DO NOT USE IN PRODUCTION
            if len(signature) != 3293:
                return False
            expected_hash = signature[:32]
            h = hashlib.sha3_256()
            h.update(key_to_use[:32])  # Use first 32 bytes as "key"
            h.update(message)
            return h.digest() == expected_hash


class SymmetricEncryption:
    """
    AES-256-GCM symmetric encryption for data confidentiality
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize with encryption key
        
        Args:
            key: 32-byte AES key. If None, generates random key.
        """
        if key is None:
            key = secrets.token_bytes(32)
        elif len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")
            
        self.aes = AESGCM(key)
        self.key = key
        
    def encrypt(self, plaintext: Union[str, bytes], 
                associated_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext with AES-256-GCM
        
        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (optional)
            
        Returns:
            Tuple of (ciphertext, nonce)
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
            
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        ciphertext = self.aes.encrypt(nonce, plaintext, associated_data)
        
        return ciphertext, nonce
        
    def decrypt(self, ciphertext: bytes, nonce: bytes,
                associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt ciphertext with AES-256-GCM
        
        Args:
            ciphertext: Encrypted data
            nonce: Nonce used for encryption
            associated_data: Additional authenticated data (optional)
            
        Returns:
            Decrypted plaintext
            
        Raises:
            InvalidSignature: If decryption/authentication fails
        """
        return self.aes.decrypt(nonce, ciphertext, associated_data)


class SecureHash:
    """
    SHA3-256 hash function for integrity and Merkle trees
    """
    
    @staticmethod
    def hash(data: Union[str, bytes]) -> bytes:
        """
        Compute SHA3-256 hash
        
        Args:
            data: Data to hash
            
        Returns:
            32-byte hash digest
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return hashlib.sha3_256(data).digest()
        
    @staticmethod
    def hash_multiple(*args: Union[str, bytes]) -> bytes:
        """
        Hash multiple inputs together
        
        Args:
            args: Variable number of inputs to hash together
            
        Returns:
            32-byte hash digest
        """
        h = hashlib.sha3_256()
        for data in args:
            if isinstance(data, str):
                data = data.encode('utf-8')
            h.update(data)
        return h.digest()


class CryptographicAccumulator:
    """
    Simple cryptographic accumulator for set membership proofs
    
    This is a basic implementation - production systems may want
    more sophisticated accumulators.
    """
    
    def __init__(self):
        """Initialize empty accumulator"""
        self.elements: Dict[bytes, bool] = {}
        self.current_digest = SecureHash.hash(b"EMPTY_ACCUMULATOR")
        
    def add_element(self, element: Union[str, bytes]) -> bytes:
        """
        Add an element to the accumulator
        
        Args:
            element: Element to add
            
        Returns:
            Updated accumulator digest
        """
        if isinstance(element, str):
            element = element.encode('utf-8')
            
        element_hash = SecureHash.hash(element)
        self.elements[element_hash] = True
        
        # Update accumulator digest
        all_elements = sorted(self.elements.keys())
        self.current_digest = SecureHash.hash(b''.join(all_elements))
        
        return self.current_digest
        
    def generate_membership_proof(self, element: Union[str, bytes]) -> Optional[Dict[str, Any]]:
        """
        Generate proof that an element is in the accumulator
        
        Args:
            element: Element to prove membership for
            
        Returns:
            Membership proof or None if element not in set
        """
        if isinstance(element, str):
            element = element.encode('utf-8')
            
        element_hash = SecureHash.hash(element)
        if element_hash not in self.elements:
            return None
            
        # Simple proof: list of all elements (not efficient for large sets)
        return {
            'element': element,
            'element_hash': element_hash,
            'all_elements': list(self.elements.keys()),
            'accumulator_digest': self.current_digest
        }
        
    def verify_membership(self, element: Union[str, bytes], 
                         proof: Dict[str, Any]) -> bool:
        """
        Verify membership proof
        
        Args:
            element: Element to verify
            proof: Membership proof
            
        Returns:
            True if proof is valid, False otherwise
        """
        if isinstance(element, str):
            element = element.encode('utf-8')
            
        element_hash = SecureHash.hash(element)
        
        # Verify element hash matches
        if element_hash != proof['element_hash']:
            return False
            
        # Verify accumulator digest
        expected_digest = SecureHash.hash(b''.join(sorted(proof['all_elements'])))
        if expected_digest != proof['accumulator_digest']:
            return False
            
        # Verify element is in the set
        return element_hash in proof['all_elements']


# Factory functions for easy instantiation
def create_pq_prf(key: Optional[bytes] = None) -> PostQuantumPRF:
    """Create a new PQ-PRF instance"""
    return PostQuantumPRF(key)


def create_pq_signature() -> PostQuantumSignature:
    """Create a new PQ signature instance"""
    return PostQuantumSignature()


def create_symmetric_encryption(key: Optional[bytes] = None) -> SymmetricEncryption:
    """Create a new symmetric encryption instance"""
    return SymmetricEncryption(key)


# Module-level constants
HASH_OUTPUT_SIZE = 32  # SHA3-256 output size
AES_KEY_SIZE = 32      # AES-256 key size  
AES_NONCE_SIZE = 12    # AES-GCM nonce size
PRF_KEY_SIZE = 32      # PQ-PRF key size

if PQC_AVAILABLE:
    DILITHIUM3_PUBLIC_KEY_SIZE = 1952
    DILITHIUM3_SECRET_KEY_SIZE = 4000
    DILITHIUM3_SIGNATURE_SIZE = 3293
else:
    # Approximate sizes for mock implementation
    DILITHIUM3_PUBLIC_KEY_SIZE = 1952
    DILITHIUM3_SECRET_KEY_SIZE = 4000  
    DILITHIUM3_SIGNATURE_SIZE = 3293