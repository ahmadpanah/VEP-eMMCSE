VEP-eMMCSE Documentation
=======================

Verifiable, Expressive, and Post-Quantum enhanced Multi-source Multi-client Searchable Encryption

VEP-eMMCSE is a comprehensive framework that extends the traditional eMMCSE scheme with:

1. Post-quantum security guarantees
2. Verifiable search results using Merkle trees
3. Expressive Boolean query support
4. Multi-source and multi-client capabilities

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   api_reference
   cryptographic_primitives
   security_analysis
   performance
   examples


Installation
------------

To install VEP-eMMCSE::

    pip install vep-emmcse

For development installation::

    git clone https://github.com/ahmadpanah/vep-emmcse.git
    cd vep-emmcse
    pip install -e .[dev]


Quick Start
----------

Here's a simple example of using VEP-eMMCSE::

    from vep_emmcse.schemes.vep_emmcse import create_vep_emmcse
    
    # Create and initialize scheme
    scheme = create_vep_emmcse()
    scheme.setup(data_source_ids=["source1"])
    
    # Add a document
    scheme.update_document(
        source_id="source1",
        operation="add",
        document_id="doc1",
        keywords=["keyword1", "keyword2"]
    )
    
    # Generate search token
    token = scheme.search_token_gen(
        client_id="client1",
        query="keyword1 AND keyword2"
    )
    
    # Execute search
    results = scheme.search(token)
    
    # Verify results
    verified_docs = scheme.verify_results("client1", results, "keyword1 AND keyword2")


Features
--------

- **Post-Quantum Security**: Built on post-quantum cryptographic primitives
- **Verifiable Search Results**: Uses Merkle trees for result verification
- **Expressive Queries**: Supports complex Boolean queries
- **Multi-Source**: Handles multiple data sources
- **Multi-Client**: Supports multiple clients with different access rights
- **Performance**: Optimized for large-scale deployments


Contributing
-----------

Contributions are welcome! Please feel free to submit a Pull Request.


License
-------

This project is licensed under the MIT License - see the LICENSE file for details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
