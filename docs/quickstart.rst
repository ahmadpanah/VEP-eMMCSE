Quick Start Guide
===============

This guide will help you get started with VEP-eMMCSE quickly.

Basic Usage
----------

1. Initialize the Scheme
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vep_emmcse.schemes.vep_emmcse import create_vep_emmcse
    
    # Create scheme instance
    scheme = create_vep_emmcse()
    
    # Initialize with data sources
    params = scheme.setup(
        security_level=128,
        data_source_ids=["source1", "source2"]
    )

2. Setup Clients
~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate client keys
    client_keys = scheme.aggregate_key(
        client_id="client1",
        authorized_sources=["source1"]
    )

3. Add Documents
~~~~~~~~~~~~~~

.. code-block:: python

    # Add a document with keywords
    scheme.update_document(
        source_id="source1",
        operation="add",
        document_id="doc1",
        keywords=["keyword1", "keyword2", "keyword3"]
    )

4. Search Documents
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate search token
    token = scheme.search_token_gen(
        client_id="client1",
        query="keyword1 AND keyword2"
    )
    
    # Execute search
    results = scheme.search(token)

5. Verify Results
~~~~~~~~~~~~~~~

.. code-block:: python

    # Verify search results
    verified_docs = scheme.verify_results(
        client_id="client1",
        search_result=results,
        original_query="keyword1 AND keyword2"
    )

Complete Example
--------------

Here's a complete example putting it all together:

.. code-block:: python

    from vep_emmcse.schemes.vep_emmcse import create_vep_emmcse
    
    def main():
        # Create and initialize scheme
        scheme = create_vep_emmcse()
        params = scheme.setup(
            security_level=128,
            data_source_ids=["source1", "source2"]
        )
        
        # Setup client
        client_keys = scheme.aggregate_key(
            client_id="client1",
            authorized_sources=["source1"]
        )
        
        # Add documents
        docs = [
            {
                "id": "doc1",
                "keywords": ["research", "cryptography", "security"]
            },
            {
                "id": "doc2",
                "keywords": ["cryptography", "privacy", "blockchain"]
            }
        ]
        
        for doc in docs:
            scheme.update_document(
                source_id="source1",
                operation="add",
                document_id=doc["id"],
                keywords=doc["keywords"]
            )
        
        # Search
        query = "cryptography AND security"
        token = scheme.search_token_gen("client1", query)
        results = scheme.search(token)
        
        # Verify
        verified_docs = scheme.verify_results("client1", results, query)
        
        print(f"Found and verified documents: {verified_docs}")
    
    if __name__ == "__main__":
        main()

Using Boolean Queries
------------------

VEP-eMMCSE supports complex Boolean queries:

.. code-block:: python

    # Simple AND query
    "keyword1 AND keyword2"
    
    # Simple OR query
    "keyword1 OR keyword2"
    
    # Complex query
    "(keyword1 AND keyword2) OR (keyword3 AND keyword4)"

Performance Considerations
-----------------------

1. Batch document updates when possible
2. Cache frequently used search tokens
3. Use appropriate keyword selectivity
4. Consider result set size for verification overhead

Next Steps
---------

- Read the Architecture documentation
- Review Security Analysis
- Check Performance Benchmarks
- Explore Advanced Examples
