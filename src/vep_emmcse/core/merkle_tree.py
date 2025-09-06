#!/usr/bin/env python3
"""
Merkle Hash Tree Implementation for VEP-eMMCSE Verifiability

This module provides a complete Merkle Hash Tree implementation with
authentication path generation and verification for ensuring search
result integrity in the VEP-eMMCSE framework.

"""

import hashlib
from typing import List, Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass
import json
import math

from .crypto_primitives import SecureHash


@dataclass
class MerkleNode:
    """
    Represents a node in the Merkle Hash Tree
    """
    hash_value: bytes
    left_child: Optional['MerkleNode'] = None
    right_child: Optional['MerkleNode'] = None
    is_leaf: bool = False
    data: Optional[bytes] = None  # Original data for leaf nodes


@dataclass
class MerkleProof:
    """
    Authentication path proof for a leaf in the Merkle tree
    """
    leaf_index: int
    leaf_hash: bytes
    sibling_hashes: List[bytes]
    sibling_positions: List[str]  # 'left' or 'right'
    root_hash: bytes


class MerkleHashTree:
    """
    Complete Merkle Hash Tree implementation with verifiability features
    
    This implementation provides:
    - Dynamic tree construction from data items
    - Authentication path generation
    - Proof verification
    - Tree updates and maintenance
    """
    
    def __init__(self):
        """Initialize an empty Merkle Hash Tree"""
        self.root: Optional[MerkleNode] = None
        self.leaves: List[bytes] = []
        self.leaf_data: List[bytes] = []
        self.tree_height: int = 0
        
    def build_tree(self, data_items: List[Union[str, bytes]]) -> bytes:
        """
        Build the Merkle tree from a list of data items
        
        Args:
            data_items: List of data items to include in the tree
            
        Returns:
            Root hash of the constructed tree
        """
        if not data_items:
            # Empty tree has a special root
            self.root = MerkleNode(SecureHash.hash(b"EMPTY_TREE"))
            return self.root.hash_value
            
        # Convert data items to bytes and compute leaf hashes
        self.leaf_data = []
        self.leaves = []
        
        for item in data_items:
            if isinstance(item, str):
                item_bytes = item.encode('utf-8')
            else:
                item_bytes = item
                
            self.leaf_data.append(item_bytes)
            leaf_hash = SecureHash.hash(item_bytes)
            self.leaves.append(leaf_hash)
            
        # Build tree bottom-up
        current_level = [
            MerkleNode(hash_val, is_leaf=True, data=data)
            for hash_val, data in zip(self.leaves, self.leaf_data)
        ]
        
        level_count = 1
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs of nodes
            for i in range(0, len(current_level), 2):
                left_node = current_level[i]
                
                # Handle odd number of nodes by duplicating the last one
                if i + 1 < len(current_level):
                    right_node = current_level[i + 1]
                else:
                    right_node = left_node
                    
                # Compute parent hash
                combined_hash = left_node.hash_value + right_node.hash_value
                parent_hash = SecureHash.hash(combined_hash)
                
                # Create parent node
                parent_node = MerkleNode(
                    hash_value=parent_hash,
                    left_child=left_node,
                    right_child=right_node
                )
                
                next_level.append(parent_node)
                
            current_level = next_level
            level_count += 1
            
        self.root = current_level[0]
        self.tree_height = level_count
        
        return self.root.hash_value
        
    def get_root_hash(self) -> Optional[bytes]:
        """
        Get the root hash of the tree
        
        Returns:
            Root hash or None if tree is empty
        """
        return self.root.hash_value if self.root else None
        
    def generate_proof(self, leaf_index: int) -> Optional[MerkleProof]:
        """
        Generate authentication path proof for a leaf
        
        Args:
            leaf_index: Index of the leaf to prove (0-based)
            
        Returns:
            MerkleProof object or None if index is invalid
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            return None
            
        if not self.root:
            return None
            
        leaf_hash = self.leaves[leaf_index]
        sibling_hashes = []
        sibling_positions = []
        
        # Traverse from leaf to root, collecting sibling hashes
        current_index = leaf_index
        current_node = self.root
        
        # Find path to leaf by traversing tree
        path_to_leaf = self._find_path_to_leaf(leaf_index)
        
        if not path_to_leaf:
            return None
            
        # Build authentication path
        for level in range(len(path_to_leaf) - 1):
            node = path_to_leaf[level]
            
            if current_index % 2 == 0:  # Left child
                # Sibling is right child
                if node.right_child and node.right_child != node.left_child:
                    sibling_hashes.append(node.right_child.hash_value)
                    sibling_positions.append('right')
                else:
                    # Duplicate case
                    sibling_hashes.append(node.left_child.hash_value)
                    sibling_positions.append('right')
            else:  # Right child
                # Sibling is left child
                sibling_hashes.append(node.left_child.hash_value)
                sibling_positions.append('left')
                
            current_index //= 2
            
        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=leaf_hash,
            sibling_hashes=sibling_hashes,
            sibling_positions=sibling_positions,
            root_hash=self.root.hash_value
        )
        
    def _find_path_to_leaf(self, leaf_index: int) -> Optional[List[MerkleNode]]:
        """
        Find the path from root to a specific leaf
        
        Args:
            leaf_index: Index of target leaf
            
        Returns:
            List of nodes from root to leaf, or None if not found
        """
        if not self.root or leaf_index >= len(self.leaves):
            return None
            
        path = []
        
        def traverse(node: MerkleNode, target_index: int, 
                    current_min: int, current_max: int) -> bool:
            path.append(node)
            
            if node.is_leaf:
                return target_index == current_min
                
            mid = (current_min + current_max) // 2
            
            # Try left subtree
            if target_index <= mid and node.left_child:
                if traverse(node.left_child, target_index, current_min, mid):
                    return True
                    
            # Try right subtree
            path.pop()  # Remove this node from path
            if target_index > mid and node.right_child:
                path.append(node)
                if traverse(node.right_child, target_index, mid + 1, current_max):
                    return True
                    
            path.pop()  # Remove this node from path
            return False
            
        # Start traversal
        max_leaf_index = len(self.leaves) - 1
        if traverse(self.root, leaf_index, 0, max_leaf_index):
            return path
        else:
            return None
            
    @staticmethod
    def verify_proof(proof: MerkleProof, original_data: Union[str, bytes]) -> bool:
        """
        Verify a Merkle proof
        
        Args:
            proof: MerkleProof to verify
            original_data: Original data that should hash to the leaf
            
        Returns:
            True if proof is valid, False otherwise
        """
        if isinstance(original_data, str):
            original_data = original_data.encode('utf-8')
            
        # Verify leaf hash matches original data
        expected_leaf_hash = SecureHash.hash(original_data)
        if expected_leaf_hash != proof.leaf_hash:
            return False
            
        # Reconstruct root hash using authentication path
        current_hash = proof.leaf_hash
        
        for sibling_hash, position in zip(proof.sibling_hashes, proof.sibling_positions):
            if position == 'left':
                # Sibling is on the left, current hash is on the right
                combined = sibling_hash + current_hash
            else:
                # Sibling is on the right, current hash is on the left
                combined = current_hash + sibling_hash
                
            current_hash = SecureHash.hash(combined)
            
        # Check if reconstructed root matches expected root
        return current_hash == proof.root_hash
        
    def add_leaf(self, data: Union[str, bytes]) -> Tuple[int, bytes]:
        """
        Add a new leaf to the tree and rebuild
        
        Args:
            data: Data to add as new leaf
            
        Returns:
            Tuple of (leaf_index, new_root_hash)
        """
        # Add to current data
        new_data = list(self.leaf_data)
        if isinstance(data, str):
            new_data.append(data.encode('utf-8'))
        else:
            new_data.append(data)
            
        # Rebuild tree
        new_root = self.build_tree(new_data)
        
        return len(self.leaves) - 1, new_root
        
    def batch_add_leaves(self, data_list: List[Union[str, bytes]]) -> bytes:
        """
        Add multiple leaves efficiently
        
        Args:
            data_list: List of data items to add
            
        Returns:
            New root hash after additions
        """
        # Combine with existing data
        combined_data = list(self.leaf_data)
        for data in data_list:
            if isinstance(data, str):
                combined_data.append(data.encode('utf-8'))
            else:
                combined_data.append(data)
                
        # Rebuild tree with all data
        return self.build_tree(combined_data)
        
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the current tree state
        
        Returns:
            Dictionary with tree statistics
        """
        return {
            'num_leaves': len(self.leaves),
            'tree_height': self.tree_height,
            'root_hash': self.root.hash_value.hex() if self.root else None,
            'is_empty': len(self.leaves) == 0
        }
        
    def serialize_proof(self, proof: MerkleProof) -> str:
        """
        Serialize a proof to JSON string
        
        Args:
            proof: MerkleProof to serialize
            
        Returns:
            JSON string representation
        """
        proof_dict = {
            'leaf_index': proof.leaf_index,
            'leaf_hash': proof.leaf_hash.hex(),
            'sibling_hashes': [h.hex() for h in proof.sibling_hashes],
            'sibling_positions': proof.sibling_positions,
            'root_hash': proof.root_hash.hex()
        }
        
        return json.dumps(proof_dict, indent=2)
        
    @staticmethod
    def deserialize_proof(json_str: str) -> MerkleProof:
        """
        Deserialize a proof from JSON string
        
        Args:
            json_str: JSON string representation
            
        Returns:
            MerkleProof object
        """
        proof_dict = json.loads(json_str)
        
        return MerkleProof(
            leaf_index=proof_dict['leaf_index'],
            leaf_hash=bytes.fromhex(proof_dict['leaf_hash']),
            sibling_hashes=[bytes.fromhex(h) for h in proof_dict['sibling_hashes']],
            sibling_positions=proof_dict['sibling_positions'],
            root_hash=bytes.fromhex(proof_dict['root_hash'])
        )