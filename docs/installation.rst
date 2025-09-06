Installation Guide
==================

Prerequisites
------------

VEP-eMMCSE requires:

- Python 3.10 or later
- pip package manager
- A Unix-like environment (Linux, macOS) or Windows

Basic Installation
----------------

Install VEP-eMMCSE using pip::

    pip install vep-emmcse

Development Installation
----------------------

For development, clone the repository and install in editable mode::

    git clone https://github.com/ahmadpanah/vep-emmcse.git
    cd vep-emmcse
    pip install -e .[dev]

This will install additional dependencies needed for development:

- pytest for testing
- black for code formatting
- flake8 for linting
- mypy for type checking
- sphinx for documentation

Installing Optional Components
---------------------------

For running experiments and benchmarks::

    pip install -e .[experiments]

For building documentation::

    pip install -e .[docs]

Verification
-----------

To verify the installation::

    python -c "from vep_emmcse.schemes.vep_emmcse import create_vep_emmcse; print('Installation successful!')"

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. Import errors:
   
   - Ensure Python version is 3.10 or later
   - Check if package is installed using ``pip list | grep vep-emmcse``

2. Cryptographic backend issues:
   
   - Install system-level dependencies: ``apt-get install build-essential python3-dev``
   - Try reinstalling cryptography: ``pip install --force-reinstall cryptography``

Getting Help
~~~~~~~~~~~

If you encounter issues:

1. Check the GitHub issues page
2. Join our community discussion
3. Contact the maintainers
