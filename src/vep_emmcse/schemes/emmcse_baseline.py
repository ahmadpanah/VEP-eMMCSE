#!/usr/bin/env python3
"""
eMMCSE-PQ: Post-Quantum Multi-source Multi-client Searchable Encryption

This module implements the baseline eMMCSE-PQ scheme from the paper for comparison
with the enhanced VEP-eMMCSE framework. This implementation focuses on the core
functionality without the verifiability extensions.
"""

import os
import json
import secrets
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from collections import defaultdict

from ..core.crypto_primitives import (
    PostQuantumPRF, PostQuantumSignature, SymmetricEncryption,
    SecureHash, create_pq_prf, create_pq_signature
)


@dataclass
class BaseSystemParameters:
    """System parameters for eMMCSE-PQ"""
    security_level: int
    master_secret_key: bytes
    system_id: str


@dataclass
class BaseDataSourceKeys:
    """Key material for a data source in eMMCSE-PQ"""
    source_id: str
    prf_key: bytes
    signature_keypair: Tuple[bytes, bytes]


@dataclass
class BaseClientKeys:
    """Client key material for eMMCSE-PQ"""
    client_id: str
    access_key: bytes
    authorized_sources: Set[str]


@dataclass
class BaseSearchToken:
    """Search token for eMMCSE-PQ"""
    token_id: str
    query_dnf: List[List[str]]
    encrypted_tokens: Dict[str, bytes]


@dataclass
class BaseSearchResult:
    """Search result from eMMCSE-PQ"""
    document_ids: List[str]
    metadata: Dict[str, Any]


class eMMCSE_PQ:
    """
    Implementation of the eMMCSE-PQ baseline scheme
    
    This represents the base scheme without verifiability extensions,
    used as a performance baseline for comparison with VEP-eMMCSE.
    """
    
    def __init__(self):
        """Initialize eMMCSE-PQ instance"""
        self.system_params: Optional[BaseSystemParameters] = None
        self.data_sources: Dict[str, BaseDataSourceKeys] = {}
        self.clients: Dict[str, BaseClientKeys] = {}
        self.encrypted_database: Dict[str, Dict[str, Any]] = {}
        
    def setup(self, security_level: int = 128,
             data_source_ids: Optional[List[str]] = None) -> BaseSystemParameters:
        """
        Setup eMMCSE-PQ system
        
        Args:
            security_level: Security parameter (default 128)
            data_source_ids: List of data source identifiers
            
        Returns:
            System parameters
        """
        # Generate master secret key
        master_key = secrets.token_bytes(32)
        
        self.system_params = BaseSystemParameters(
            security_level=security_level,
            master_secret_key=master_key,
            system_id=secrets.token_hex(16)
        )
        
        # Initialize data sources
        if data_source_ids:
            for source_id in data_source_ids:
                self._setup_data_source(source_id)
                
        return self.system_params
        
    def _setup_data_source(self, source_id: str) -> BaseDataSourceKeys:
        """Setup cryptographic keys for a data source"""
        if not self.system_params:
            raise ValueError("System must be initialized first")
            
        # Generate source-specific keys
        master_prf = create_pq_prf(self.system_params.master_secret_key)
        prf_key = master_prf.evaluate(f"SOURCE:{source_id}")
        
        # Generate signature keypair
        sig_scheme = create_pq_signature()
        public_key, secret_key = sig_scheme.generate_keypair()
        
        # Store data source keys
        ds_keys = BaseDataSourceKeys(
            source_id=source_id,
            prf_key=prf_key,
            signature_keypair=(public_key, secret_key)
        )
        
        self.data_sources[source_id] = ds_keys
        return ds_keys
        
    def generate_client_key(self, client_id: str,
                          authorized_sources: List[str]) -> BaseClientKeys:
        """
        Generate client access key
        
        Args:
            client_id: Unique client identifier
            authorized_sources: List of authorized data sources
            
        Returns:
            Client key material
        """
        if not self.system_params:
            raise ValueError("System must be initialized first")
            
        # Generate client access key
        master_prf = create_pq_prf(self.system_params.master_secret_key)
        access_key = master_prf.evaluate(f"CLIENT:{client_id}")
        
        client_keys = BaseClientKeys(
            client_id=client_id,
            access_key=access_key,
            authorized_sources=set(authorized_sources)
        )
        
        self.clients[client_id] = client_keys
        return client_keys
        
    def update_document(self, source_id: str, operation: str,
                       document_id: str, keywords: List[str]) -> bool:
        """
        Add or delete document
        
        Args:
            source_id: Data source ID
            operation: 'add' or 'delete'
            document_id: Document identifier
            keywords: Document keywords
            
        Returns:
            True if successful
        """
        if source_id not in self.data_sources:
            raise ValueError(f"Data source {source_id} not found")
            
        ds_keys = self.data_sources[source_id]
        source_prf = create_pq_prf(ds_keys.prf_key)
        
        try:
            if operation == 'add':
                # Generate encrypted index entries
                index_entries = {}
                for keyword in keywords:
                    token_input = f"{document_id}:{keyword}"
                    index_entries[keyword] = source_prf.evaluate(token_input.encode())
                    
                # Sign update
                sig_scheme = create_pq_signature()
                sig_scheme.secret_key = ds_keys.signature_keypair[1]
                signature = sig_scheme.sign(f"{document_id}:{':'.join(keywords)}")
                
                # Store document
                self.encrypted_database[document_id] = {
                    'document_id': document_id,
                    'source_id': source_id,
                    'keywords': keywords,
                    'index_entries': index_entries,
                    'signature': signature,
                    'operation': 'add'
                }
                
            elif operation == 'delete':
                if document_id in self.encrypted_database:
                    self.encrypted_database[document_id]['operation'] = 'delete'
                    self.encrypted_database[document_id]['deleted'] = True
                    
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            return True
            
        except Exception as e:
            print(f"Update failed: {e}")
            return False
            
    def search_token_gen(self, client_id: str, query: str) -> BaseSearchToken:
        """
        Generate search token
        
        Args:
            client_id: Client performing search
            query: Boolean query string
            
        Returns:
            Search token
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
            
        client_keys = self.clients[client_id]
        
        # Parse query into DNF
        dnf_clauses = self._parse_query(query)
        
        # Generate tokens for each clause
        encrypted_tokens = {}
        client_prf = create_pq_prf(client_keys.access_key)
        
        for i, clause in enumerate(dnf_clauses):
            clause_id = f"clause_{i}"
            token_input = f"{client_id}:{':'.join(sorted(clause))}"
            encrypted_tokens[clause_id] = client_prf.evaluate(token_input.encode())
            
        return BaseSearchToken(
            token_id=secrets.token_hex(16),
            query_dnf=dnf_clauses,
            encrypted_tokens=encrypted_tokens
        )
        
    def search(self, search_token: BaseSearchToken) -> BaseSearchResult:
        """
        Execute search
        
        Args:
            search_token: Search token
            
        Returns:
            Search results
        """
        matching_docs = set()
        
        # Evaluate each DNF clause
        for clause_id, clause_keywords in enumerate(search_token.query_dnf):
            clause_matches = self._evaluate_clause(
                clause_keywords,
                search_token.encrypted_tokens[f"clause_{clause_id}"]
            )
            matching_docs.update(clause_matches)
            
        # Filter deleted documents
        active_docs = [doc_id for doc_id in matching_docs
                      if doc_id in self.encrypted_database and
                      self.encrypted_database[doc_id].get('operation') == 'add' and
                      not self.encrypted_database[doc_id].get('deleted', False)]
                      
        return BaseSearchResult(
            document_ids=active_docs,
            metadata={
                'query_dnf': search_token.query_dnf,
                'num_clauses': len(search_token.query_dnf)
            }
        )
        
    def _evaluate_clause(self, keywords: List[str], 
                        clause_token: bytes) -> Set[str]:
        """Evaluate a conjunctive clause against the database"""
        matching_docs = set()
        
        # Check each document in database
        for doc_id, doc_entry in self.encrypted_database.items():
            if doc_entry.get('operation') == 'add' and not doc_entry.get('deleted', False):
                # Check if all keywords in clause are present
                doc_keywords = set(doc_entry['keywords'])
                if all(kw in doc_keywords for kw in keywords):
                    matching_docs.add(doc_id)
                    
        return matching_docs
        
    def _parse_query(self, query: str) -> List[List[str]]:
        """Parse Boolean query into DNF format"""
        if ' OR ' in query:
            clauses = []
            for part in query.split(' OR '):
                part = part.strip('()')
                if ' AND ' in part:
                    keywords = [kw.strip() for kw in part.split(' AND ')]
                    clauses.append(keywords)
                else:
                    clauses.append([part.strip()])
            return clauses
        elif ' AND ' in query:
            keywords = [kw.strip() for kw in query.split(' AND ')]
            return [keywords]
        else:
            return [[query.strip()]]


# Factory function
def create_emmcse_pq() -> eMMCSE_PQ:
    """Create new eMMCSE-PQ instance"""
    return eMMCSE_PQ()
