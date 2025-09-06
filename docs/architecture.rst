System Architecture
===================

This document describes the architecture of the VEP-eMMCSE framework.

Overview
--------

VEP-eMMCSE extends the traditional eMMCSE scheme with three main enhancements:

1. Post-quantum cryptographic primitives
2. Verifiable search results
3. Expressive Boolean query support

The system consists of several core components:

Component Hierarchy
-----------------

.. code-block:: text

    VEP-eMMCSE
    ├── Core Components
    │   ├── Cryptographic Primitives
    │   │   ├── Post-Quantum PRF
    │   │   ├── Post-Quantum Signatures
    │   │   ├── Symmetric Encryption
    │   │   └── Secure Hash Functions
    │   │
    │   ├── Merkle Tree
    │   │   ├── Tree Construction
    │   │   ├── Proof Generation
    │   │   └── Verification
    │   │
    │   └── Boolean Query Parser
    │       ├── Query Normalization
    │       └── DNF Conversion
    │
    ├── Scheme Components
    │   ├── Setup Phase
    │   ├── Key Generation
    │   ├── Document Update
    │   ├── Search Token Generation
    │   ├── Search Execution
    │   └── Result Verification
    │
    └── Utilities
        ├── Dataset Handling
        ├── Benchmarking
        └── Performance Analysis

Key Components
------------

Cryptographic Primitives
~~~~~~~~~~~~~~~~~~~~~~~

The core cryptographic operations are implemented in ``crypto_primitives.py``:

- **PostQuantumPRF**: Implements quantum-resistant pseudorandom functions
- **PostQuantumSignature**: Provides post-quantum digital signatures
- **SymmetricEncryption**: Handles data encryption/decryption
- **SecureHash**: Implements cryptographic hash functions
- **CryptographicAccumulator**: Provides set membership proofs

Merkle Tree
~~~~~~~~~~

The Merkle tree implementation in ``merkle_tree.py`` provides:

- Dynamic tree construction
- Efficient proof generation
- Result verification
- Serialization support

Main Scheme
~~~~~~~~~~

The main scheme implementation in ``vep_emmcse.py`` includes:

- System setup and initialization
- Key generation and management
- Document addition and deletion
- Search token generation
- Search execution
- Result verification

Data Flow
--------

1. Setup Phase
~~~~~~~~~~~~~

.. code-block:: text

    Setup
    ├── Generate master secret key
    ├── Initialize Merkle tree
    └── Setup data sources
        ├── Generate source-specific keys
        └── Initialize source state

2. Document Update
~~~~~~~~~~~~~~~~

.. code-block:: text

    Update
    ├── Process document keywords
    ├── Generate index entries
    ├── Update Merkle tree
    └── Sign update

3. Search Process
~~~~~~~~~~~~~~~

.. code-block:: text

    Search
    ├── Parse Boolean query
    ├── Generate search token
    ├── Execute search
    └── Generate verification proofs

Security Architecture
------------------

The security of VEP-eMMCSE relies on several components:

1. **Post-Quantum Security**
   - Quantum-resistant PRF
   - Post-quantum signatures
   - Forward security mechanisms

2. **Verifiability**
   - Merkle tree authentication
   - Cryptographic accumulators
   - Digital signatures

3. **Access Control**
   - Fine-grained permissions
   - Source-specific access rights
   - Key aggregation

Implementation Details
-------------------

Core Data Structures
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class SystemParameters:
        security_level: int
        master_secret_key: bytes
        public_verification_root: bytes
        system_id: str

    @dataclass
    class DataSourceKeys:
        source_id: str
        identity_key: bytes
        prf_keys: Dict[str, bytes]
        signature_keypair: Tuple[bytes, bytes]

    @dataclass
    class ClientKeys:
        client_id: str
        aggregated_key: bytes
        access_rights: Set[str]
        state_set: Dict[str, Any]

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~

1. **Search Optimizations**
   - Efficient Boolean query parsing
   - Optimized index structure
   - Parallel search execution

2. **Verification Optimizations**
   - Batch proof generation
   - Proof aggregation
   - Efficient tree updates

3. **Memory Optimizations**
   - Lazy loading
   - Caching strategies
   - Memory-efficient data structures

Extension Points
-------------

The architecture supports several extension points:

1. **Cryptographic Primitives**
   - New post-quantum algorithms
   - Alternative PRF constructions
   - Custom signature schemes

2. **Query Processing**
   - Extended query languages
   - Custom query optimizers
   - New index structures

3. **Verification Mechanisms**
   - Alternative proof systems
   - Custom accumulator schemes
   - New tree structures
