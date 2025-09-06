API Reference
============

This document provides detailed API documentation for the VEP-eMMCSE framework.

Core Components
-------------

Cryptographic Primitives
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vep_emmcse.core.crypto_primitives
   :members:
   :undoc-members:
   :show-inheritance:

Merkle Tree
~~~~~~~~~~

.. automodule:: vep_emmcse.core.merkle_tree
   :members:
   :undoc-members:
   :show-inheritance:

Main Scheme
----------

VEP-eMMCSE Scheme
~~~~~~~~~~~~~~~

.. automodule:: vep_emmcse.schemes.vep_emmcse
   :members:
   :undoc-members:
   :show-inheritance:

eMMCSE-PQ Baseline
~~~~~~~~~~~~~~~~

.. automodule:: vep_emmcse.schemes.emmcse_baseline
   :members:
   :undoc-members:
   :show-inheritance:

Utility Modules
-------------

Dataset Handler
~~~~~~~~~~~~~

.. automodule:: vep_emmcse.utils.dataset_handler
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark Tools
~~~~~~~~~~~~~

.. automodule:: vep_emmcse.utils.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

Data Types
---------

System Parameters
~~~~~~~~~~~~~~

.. autoclass:: vep_emmcse.schemes.vep_emmcse.SystemParameters
   :members:
   :undoc-members:

Data Source Keys
~~~~~~~~~~~~~~

.. autoclass:: vep_emmcse.schemes.vep_emmcse.DataSourceKeys
   :members:
   :undoc-members:

Client Keys
~~~~~~~~~

.. autoclass:: vep_emmcse.schemes.vep_emmcse.ClientKeys
   :members:
   :undoc-members:

Search Token
~~~~~~~~~~

.. autoclass:: vep_emmcse.schemes.vep_emmcse.SearchToken
   :members:
   :undoc-members:

Search Result
~~~~~~~~~~~

.. autoclass:: vep_emmcse.schemes.vep_emmcse.SearchResult
   :members:
   :undoc-members:

Factory Functions
--------------

.. autofunction:: vep_emmcse.schemes.vep_emmcse.create_vep_emmcse

.. autofunction:: vep_emmcse.schemes.emmcse_baseline.create_emmcse_pq

.. autofunction:: vep_emmcse.utils.dataset_handler.create_dataset_handler

Exceptions
---------

.. autoexception:: vep_emmcse.core.crypto_primitives.InvalidSignature

Type Hints
---------

Common type hints used throughout the API:

.. code-block:: python

    from typing import Dict, List, Tuple, Optional, Set, Union, Any

    # Common type aliases
    DocumentId = str
    SourceId = str
    ClientId = str
    Keywords = List[str]
    Proof = Dict[str, Any]
    KeyMaterial = bytes
