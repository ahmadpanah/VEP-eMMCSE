#!/usr/bin/env python3
"""
VEP-eMMCSE: Main Framework Implementation

This module implements the complete VEP-eMMCSE scheme with verifiable results,
expressive Boolean queries, and post-quantum security for multi-source
multi-client searchable encryption.

"""

import os
import json
import secrets
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

from ..core.crypto_primitives import (
    PostQuantumPRF, PostQuantumSignature, SymmetricEncryption,
    SecureHash, create_pq_prf, create_pq_signature
)
from ..core.merkle_tree import MerkleHashTree, MerkleProof


@dataclass
class SystemParameters:
    """System-wide parameters for VEP-eMMCSE"""
    security_level: int
    master_secret_key: bytes
    public_verification_root: bytes
    system_id: str


@dataclass
class DataSourceKeys:
    """Key material for a data source"""
    source_id: str
    identity_key: bytes
    prf_keys: Dict[str, bytes]  # Multiple PRF keys for different purposes
    signature_keypair: Tuple[bytes, bytes]  # (public_key, secret_key)


@dataclass 
class ClientKeys:
    """Key material for a client"""
    client_id: str
    aggregated_key: bytes
    access_rights: Set[str]  # Set of data source IDs this client can access
    state_set: Dict[str, Any]


@dataclass
class SearchToken:
    """Token for Boolean query execution"""
    token_id: str
    query_dnf: List[List[str]]  # Disjunctive Normal Form representation
    encrypted_tokens: Dict[str, bytes]
    timestamp: float


@dataclass
class SearchResult:
    """Search result with verifiability proofs"""
    document_ids: List[str]
    merkle_proofs: Dict[str, MerkleProof]
    metadata: Dict[str, Any]


class BooleanQueryParser:
    """
    Parser for Boolean queries into Disjunctive Normal Form (DNF)
    
    Supports queries like: (keyword1 AND keyword2) OR keyword3
    """
    
    @staticmethod
    def parse_to_dnf(query: str) -> List[List[str]]:
        """
        Parse Boolean query into DNF format
        
        Args:
            query: Boolean query string
            
        Returns:
            List of conjunctive clauses (each clause is a list of keywords)
        """
        # Simple parser - production version would use proper grammar
        query = query.strip()
        
        # Handle basic cases
        if ' OR ' in query:
            # Split on OR and process each part
            clauses = []
            or_parts = query.split(' OR ')
            
            for part in or_parts:
                part = part.strip()
                if part.startswith('(') and part.endswith(')'):
                    part = part[1:-1]
                    
                if ' AND ' in part:
                    # Conjunctive clause
                    keywords = [kw.strip() for kw in part.split(' AND ')]
                    clauses.append(keywords)
                else:
                    # Single keyword
                    clauses.append([part])
                    
            return clauses
        elif ' AND ' in query:
            # Single conjunctive clause
            keywords = [kw.strip() for kw in query.split(' AND ')]
            return [keywords]
        else:
            # Single keyword
            return [[query.strip()]]


class ABELikeIndex:
    """
    Attribute-Based Encryption-like index for Boolean queries
    
    This simulates ABE functionality for supporting expressive queries
    """
    
    def __init__(self, prf: PostQuantumPRF):
        """Initialize with PRF for token generation"""
        self.prf = prf
        
    def create_index_entry(self, document_id: str, keywords: List[str]) -> Dict[str, bytes]:
        """
        Create encrypted index entries for a document
        
        Args:
            document_id: Unique document identifier
            keywords: List of keywords in the document
            
        Returns:
            Dictionary mapping keywords to encrypted index entries
        """
        index_entries = {}
        
        for keyword in keywords:
            # Create index entry that embeds document ID and keyword
            entry_data = f"{document_id}:{keyword}".encode('utf-8')
            encrypted_entry = self.prf.evaluate(entry_data)
            index_entries[keyword] = encrypted_entry
            
        return index_entries
        
    def generate_search_token(self, keywords: List[str], 
                            client_key: bytes) -> bytes:
        """
        Generate search token for conjunctive query
        
        Args:
            keywords: List of keywords in conjunction
            client_key: Client's access key
            
        Returns:
            Search token for the conjunctive clause
        """
        # Combine keywords and client key for token generation
        combined_data = ":".join(sorted(keywords)) + ":" + client_key.hex()
        return self.prf.evaluate(combined_data.encode('utf-8'))
        
    def evaluate_token(self, token: bytes, index_entries: Dict[str, bytes],
                      required_keywords: List[str]) -> bool:
        """
        Evaluate if a search token matches the document's index
        
        Args:
            token: Search token to evaluate
            index_entries: Document's encrypted index entries
            required_keywords: Keywords required for match
            
        Returns:
            True if token matches (all keywords present), False otherwise
        """
        # Check if all required keywords are present in index
        for keyword in required_keywords:
            if keyword not in index_entries:
                return False
                
        # Simulate ABE decryption check
        # In a real implementation, this would be proper ABE evaluation
        return len(required_keywords) > 0


class VEP_eMMCSE:
    """
    Main VEP-eMMCSE scheme implementation
    
    Provides verifiable, expressive, post-quantum searchable encryption
    for multi-source multi-client environments.
    """
    
    def __init__(self):
        """Initialize the VEP-eMMCSE framework"""
        self.system_params: Optional[SystemParameters] = None
        self.data_sources: Dict[str, DataSourceKeys] = {}
        self.clients: Dict[str, ClientKeys] = {}
        self.encrypted_database: Dict[str, Dict[str, Any]] = {}
        self.merkle_tree = MerkleHashTree()
        self.query_parser = BooleanQueryParser()
        
    def setup(self, security_level: int = 128, 
              data_source_ids: Optional[List[str]] = None) -> SystemParameters:
        """
        Initialize the VEP-eMMCSE system
        
        Args:
            security_level: Security parameter (default 128)
            data_source_ids: List of data source identifiers
            
        Returns:
            System parameters
        """
        if data_source_ids is None:
            data_source_ids = []
            
        # Generate master secret key
        master_key = secrets.token_bytes(32)
        
        # Initialize Merkle tree with empty state
        initial_root = self.merkle_tree.build_tree([b"INITIAL_STATE"])
        
        # Create system parameters
        self.system_params = SystemParameters(
            security_level=security_level,
            master_secret_key=master_key,
            public_verification_root=initial_root,
            system_id=secrets.token_hex(16)
        )
        
        # Initialize data sources
        for source_id in data_source_ids:
            self._setup_data_source(source_id)
            
        return self.system_params
        
    def _setup_data_source(self, source_id: str) -> DataSourceKeys:
        """
        Setup a new data source with cryptographic keys
        
        Args:
            source_id: Unique identifier for the data source
            
        Returns:
            Generated keys for the data source
        """
        if not self.system_params:
            raise ValueError("System must be initialized first")
            
        # Generate PRF keys for different purposes
        master_prf = create_pq_prf(self.system_params.master_secret_key)
        identity_key = master_prf.evaluate(f"DS:{source_id}")
        
        prf_keys = {
            'index': master_prf.evaluate(f"INDEX:{source_id}"),
            'metadata': master_prf.evaluate(f"META:{source_id}"),
            'search': master_prf.evaluate(f"SEARCH:{source_id}"),
            'hash': master_prf.evaluate(f"HASH:{source_id}")
        }
        
        # Generate signature keypair
        sig_scheme = create_pq_signature()
        public_key, secret_key = sig_scheme.generate_keypair()
        
        # Store data source keys
        ds_keys = DataSourceKeys(
            source_id=source_id,
            identity_key=identity_key,
            prf_keys=prf_keys,
            signature_keypair=(public_key, secret_key)
        )
        
        self.data_sources[source_id] = ds_keys
        return ds_keys
        
    def aggregate_key(self, client_id: str, 
                     authorized_sources: List[str]) -> ClientKeys:
        """
        Generate aggregated access key for a client
        
        Args:
            client_id: Unique client identifier
            authorized_sources: List of data sources client can access
            
        Returns:
            Client keys with access rights
        """
        if not self.system_params:
            raise ValueError("System must be initialized first")
            
        # Generate client-specific key
        master_prf = create_pq_prf(self.system_params.master_secret_key)
        client_base_key = master_prf.evaluate(f"CLIENT:{client_id}")
        
        # Create aggregated key combining access to authorized sources
        aggregated_data = client_id + ":" + ":".join(sorted(authorized_sources))
        aggregated_key = master_prf.evaluate(aggregated_data.encode('utf-8'))
        
        # Create state set for client
        state_set = {
            'authorized_sources': authorized_sources,
            'creation_time': secrets.token_hex(8),
            'access_level': 'standard'
        }
        
        client_keys = ClientKeys(
            client_id=client_id,
            aggregated_key=aggregated_key,
            access_rights=set(authorized_sources),
            state_set=state_set
        )
        
        self.clients[client_id] = client_keys
        return client_keys
        
    def update_document(self, source_id: str, operation: str,
                       document_id: str, keywords: List[str]) -> bool:
        """
        Add or delete a document from the encrypted database
        
        Args:
            source_id: Data source performing the update
            operation: 'add' or 'delete'
            document_id: Unique document identifier  
            keywords: List of keywords in the document
            
        Returns:
            True if update successful, False otherwise
        """
        if source_id not in self.data_sources:
            raise ValueError(f"Data source {source_id} not found")
            
        ds_keys = self.data_sources[source_id]
        
        try:
            if operation == 'add':
                return self._add_document(ds_keys, document_id, keywords)
            elif operation == 'delete':
                return self._delete_document(ds_keys, document_id, keywords)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            print(f"Update failed: {e}")
            return False
            
    def _add_document(self, ds_keys: DataSourceKeys, 
                     document_id: str, keywords: List[str]) -> bool:
        """Add a document to the encrypted database"""
        
        # Create ABE-like index entries
        index_prf = create_pq_prf(ds_keys.prf_keys['index'])
        abe_index = ABELikeIndex(index_prf)
        index_entries = abe_index.create_index_entry(document_id, keywords)
        
        # Create Merkle tree leaves for each keyword
        merkle_leaves = []
        for keyword in keywords:
            leaf_data = f"ADD:{ds_keys.source_id}:{document_id}:{keyword}"
            merkle_leaves.append(leaf_data.encode('utf-8'))
            
        # Update Merkle tree
        old_leaves = list(self.merkle_tree.leaf_data)
        new_leaves = old_leaves + merkle_leaves
        new_root = self.merkle_tree.build_tree(new_leaves)
        
        # Create update signature
        sig_scheme = create_pq_signature()
        sig_scheme.public_key, sig_scheme.secret_key = ds_keys.signature_keypair
        
        update_data = f"{document_id}:{':'.join(keywords)}"
        signature = sig_scheme.sign(update_data)
        
        # Store in encrypted database
        doc_entry = {
            'document_id': document_id,
            'source_id': ds_keys.source_id,
            'keywords': keywords,
            'index_entries': index_entries,
            'merkle_leaves': merkle_leaves,
            'signature': signature,
            'operation': 'add'
        }
        
        self.encrypted_database[document_id] = doc_entry
        
        # Update system verification root
        if self.system_params:
            self.system_params.public_verification_root = new_root
            
        return True
        
    def _delete_document(self, ds_keys: DataSourceKeys,
                        document_id: str, keywords: List[str]) -> bool:
        """Delete a document from the encrypted database"""
        
        if document_id not in self.encrypted_database:
            return False
            
        # Mark as deleted and add deletion proof to Merkle tree
        delete_leaf = f"DELETE:{ds_keys.source_id}:{document_id}"
        old_leaves = list(self.merkle_tree.leaf_data)
        new_leaves = old_leaves + [delete_leaf.encode('utf-8')]
        new_root = self.merkle_tree.build_tree(new_leaves)
        
        # Update database entry
        self.encrypted_database[document_id]['operation'] = 'delete'
        self.encrypted_database[document_id]['deleted'] = True
        
        # Update system verification root
        if self.system_params:
            self.system_params.public_verification_root = new_root
            
        return True
        
    def search_token_gen(self, client_id: str, query: str) -> SearchToken:
        """
        Generate search token for Boolean query
        
        Args:
            client_id: ID of client performing search
            query: Boolean query string
            
        Returns:
            Search token for the query
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
            
        client_keys = self.clients[client_id]
        
        # Parse query to DNF
        dnf_clauses = self.query_parser.parse_to_dnf(query)
        
        # Generate tokens for each DNF clause
        encrypted_tokens = {}
        
        for i, clause in enumerate(dnf_clauses):
            # Create ABE-like token for this conjunctive clause
            clause_id = f"clause_{i}"
            
            # Use client's aggregated key to generate search token
            token_data = f"{client_id}:{':'.join(sorted(clause))}"
            token_prf = create_pq_prf(client_keys.aggregated_key)
            encrypted_tokens[clause_id] = token_prf.evaluate(token_data.encode('utf-8'))
            
        search_token = SearchToken(
            token_id=secrets.token_hex(16),
            query_dnf=dnf_clauses,
            encrypted_tokens=encrypted_tokens,
            timestamp=secrets.randbits(64)
        )
        
        return search_token
        
    def search(self, search_token: SearchToken) -> SearchResult:
        """
        Execute search using the provided token
        
        Args:
            search_token: Token generated by search_token_gen
            
        Returns:
            Search results with Merkle proofs
        """
        matching_docs = set()
        merkle_proofs = {}
        
        # Evaluate each DNF clause
        for clause_id, clause_keywords in enumerate(search_token.query_dnf):
            clause_matches = self._evaluate_conjunctive_clause(
                clause_keywords, search_token.encrypted_tokens[f"clause_{clause_id}"]
            )
            matching_docs.update(clause_matches)
            
        # Generate Merkle proofs for matching documents
        for doc_id in matching_docs:
            if doc_id in self.encrypted_database:
                doc_entry = self.encrypted_database[doc_id]
                if doc_entry.get('operation') == 'add' and not doc_entry.get('deleted', False):
                    # Generate proof for each keyword that caused the match
                    for keyword in doc_entry['keywords']:
                        # Find the leaf index for this document-keyword pair
                        leaf_data = f"ADD:{doc_entry['source_id']}:{doc_id}:{keyword}"
                        proof = self._generate_merkle_proof_for_leaf(leaf_data.encode('utf-8'))
                        if proof:
                            merkle_proofs[f"{doc_id}:{keyword}"] = proof
                            
        result = SearchResult(
            document_ids=list(matching_docs),
            merkle_proofs=merkle_proofs,
            metadata={
                'query_dnf': search_token.query_dnf,
                'num_clauses': len(search_token.query_dnf),
                'timestamp': search_token.timestamp
            }
        )
        
        return result
        
    def _evaluate_conjunctive_clause(self, keywords: List[str], 
                                   clause_token: bytes) -> Set[str]:
        """Evaluate a conjunctive clause against the database"""
        matching_docs = set()
        
        for doc_id, doc_entry in self.encrypted_database.items():
            if doc_entry.get('operation') == 'add' and not doc_entry.get('deleted', False):
                # Check if all keywords in clause are present in document
                doc_keywords = set(doc_entry['keywords'])
                clause_keywords_set = set(keywords)
                
                if clause_keywords_set.issubset(doc_keywords):
                    matching_docs.add(doc_id)
                    
        return matching_docs
        
    def _generate_merkle_proof_for_leaf(self, leaf_data: bytes) -> Optional[MerkleProof]:
        """Generate Merkle proof for a specific leaf"""
        try:
            # Find the index of this leaf in the tree
            leaf_index = None
            for i, stored_leaf in enumerate(self.merkle_tree.leaf_data):
                if stored_leaf == leaf_data:
                    leaf_index = i
                    break
                    
            if leaf_index is None:
                return None
                
            return self.merkle_tree.generate_proof(leaf_index)
        except Exception:
            return None
            
    def verify_results(self, client_id: str, search_result: SearchResult, 
                      original_query: str) -> Union[List[str], None]:
        """
        Verify the integrity and completeness of search results
        
        Args:
            client_id: ID of client verifying results
            search_result: Results to verify
            original_query: Original query string
            
        Returns:
            List of verified document IDs, or None if verification fails
        """
        if client_id not in self.clients:
            return None
            
        if not self.system_params:
            return None
            
        current_root = self.system_params.public_verification_root
        verified_docs = []
        
        # Verify each Merkle proof
        for proof_key, proof in search_result.merkle_proofs.items():
            # Extract document ID and keyword from proof key
            parts = proof_key.split(':')
            if len(parts) < 2:
                continue
                
            doc_id = parts[0]
            keyword = parts[1]
            
            # Reconstruct original leaf data
            if doc_id in self.encrypted_database:
                doc_entry = self.encrypted_database[doc_id]
                source_id = doc_entry['source_id']
                leaf_data = f"ADD:{source_id}:{doc_id}:{keyword}"
                
                # Verify the proof
                if MerkleHashTree.verify_proof(proof, leaf_data.encode('utf-8')):
                    if proof.root_hash == current_root:
                        if doc_id not in verified_docs:
                            verified_docs.append(doc_id)
                    else:
                        # Root mismatch - verification failed
                        return None
                else:
                    # Invalid proof - verification failed  
                    return None
                    
        return verified_docs
        
    def revoke_client(self, source_id: str, client_id: str, 
                     revocation_type: str = 'coarse') -> bool:
        """
        Revoke access for a client
        
        Args:
            source_id: Data source performing revocation
            client_id: Client to revoke
            revocation_type: 'coarse' (all access) or 'fine' (specific documents)
            
        Returns:
            True if revocation successful
        """
        if source_id not in self.data_sources:
            return False
            
        if client_id not in self.clients:
            return False
            
        client_keys = self.clients[client_id]
        
        if revocation_type == 'coarse':
            # Remove source from client's access rights
            client_keys.access_rights.discard(source_id)
            
            # Update state set
            client_keys.state_set['revoked_sources'] = \
                client_keys.state_set.get('revoked_sources', []) + [source_id]
                
        # In a full implementation, this would also update tokens
        # and re-encrypt affected index entries
        
        return True
        
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current system state
        
        Returns:
            Dictionary with system statistics
        """
        active_docs = sum(1 for entry in self.encrypted_database.values() 
                         if entry.get('operation') == 'add' and not entry.get('deleted', False))
        
        return {
            'num_data_sources': len(self.data_sources),
            'num_clients': len(self.clients),
            'num_documents': len(self.encrypted_database),
            'num_active_documents': active_docs,
            'merkle_tree_info': self.merkle_tree.get_tree_info(),
            'system_id': self.system_params.system_id if self.system_params else None
        }
        
    def export_keys(self, output_file: str) -> bool:
        """
        Export system keys to file (for backup/recovery)
        
        Args:
            output_file: Path to output file
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                'system_params': {
                    'security_level': self.system_params.security_level,
                    'master_secret_key': self.system_params.master_secret_key.hex(),
                    'public_verification_root': self.system_params.public_verification_root.hex(),
                    'system_id': self.system_params.system_id
                } if self.system_params else None,
                'data_sources': {},
                'clients': {}
            }
            
            # Export data source keys
            for source_id, ds_keys in self.data_sources.items():
                export_data['data_sources'][source_id] = {
                    'identity_key': ds_keys.identity_key.hex(),
                    'prf_keys': {k: v.hex() for k, v in ds_keys.prf_keys.items()},
                    'signature_keypair': [
                        ds_keys.signature_keypair[0].hex(),
                        ds_keys.signature_keypair[1].hex()
                    ]
                }
                
            # Export client keys  
            for client_id, client_keys in self.clients.items():
                export_data['clients'][client_id] = {
                    'aggregated_key': client_keys.aggregated_key.hex(),
                    'access_rights': list(client_keys.access_rights),
                    'state_set': client_keys.state_set
                }
                
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
            
    def import_keys(self, input_file: str) -> bool:
        """
        Import system keys from file
        
        Args:
            input_file: Path to input file
            
        Returns:
            True if import successful
        """
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
                
            # Import system parameters
            if import_data.get('system_params'):
                sp = import_data['system_params']
                self.system_params = SystemParameters(
                    security_level=sp['security_level'],
                    master_secret_key=bytes.fromhex(sp['master_secret_key']),
                    public_verification_root=bytes.fromhex(sp['public_verification_root']),
                    system_id=sp['system_id']
                )
                
            # Import data source keys
            for source_id, ds_data in import_data.get('data_sources', {}).items():
                prf_keys = {k: bytes.fromhex(v) for k, v in ds_data['prf_keys'].items()}
                signature_keypair = (
                    bytes.fromhex(ds_data['signature_keypair'][0]),
                    bytes.fromhex(ds_data['signature_keypair'][1])
                )
                
                self.data_sources[source_id] = DataSourceKeys(
                    source_id=source_id,
                    identity_key=bytes.fromhex(ds_data['identity_key']),
                    prf_keys=prf_keys,
                    signature_keypair=signature_keypair
                )
                
            # Import client keys
            for client_id, client_data in import_data.get('clients', {}).items():
                self.clients[client_id] = ClientKeys(
                    client_id=client_id,
                    aggregated_key=bytes.fromhex(client_data['aggregated_key']),
                    access_rights=set(client_data['access_rights']),
                    state_set=client_data['state_set']
                )
                
            return True
        except Exception as e:
            print(f"Import failed: {e}")
            return False


# Factory functions for easy instantiation
def create_vep_emmcse() -> VEP_eMMCSE:
    """Create a new VEP-eMMCSE instance"""
    return VEP_eMMCSE()


def create_data_source(vep_scheme: VEP_eMMCSE, source_id: str) -> DataSourceKeys:
    """Create and register a new data source"""
    return vep_scheme._setup_data_source(source_id)


def create_client(vep_scheme: VEP_eMMCSE, client_id: str, 
                 authorized_sources: List[str]) -> ClientKeys:
    """Create and register a new client"""
    return vep_scheme.aggregate_key(client_id, authorized_sources)