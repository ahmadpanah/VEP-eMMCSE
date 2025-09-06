# VEP-eMMCSE: Verifiable, Expressive, and Post-Quantum Enhanced Multi-source Multi-client Searchable Encryption

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/ahmadpanah/vep-emmcse/workflows/CI/badge.svg)](https://github.com/ahmadpanah/vep-emmcse/actions)

A comprehensive implementation of the VEP-eMMCSE framework for secure, verifiable, and expressive searchable encryption in Cloud-IoT environments with post-quantum security guarantees.

## 🚀 Features

- **Post-Quantum Security**: Built with lattice-based cryptographic primitives resistant to quantum attacks
- **Verifiable Results**: Cryptographic proofs ensure search result integrity and completeness
- **Boolean Query Support**: Complex search expressions with AND, OR, and NOT operations
- **Multi-Source Multi-Client**: Supports collaborative Cloud-IoT environments
- **Forward/Backward Privacy**: Advanced privacy guarantees against adaptive adversaries
- **Practical Performance**: Optimized for real-world deployment scenarios

## 📋 Requirements

- **Hardware**: Intel i9-12900K or equivalent, 64GB RAM (for full experiments)
- **OS**: Ubuntu 22.04 LTS or compatible Linux distribution
- **Python**: 3.10 or higher
- **Storage**: 100GB available space for datasets

## 🛠️ Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/ahmadpanah/vep-emmcse.git
cd vep-emmcse

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,experiments]"

# Run tests to verify installation
pytest tests/
```

### System Dependencies

```bash
# Install system-level dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake libssl-dev

# For post-quantum cryptography (optional - uses Python bindings by default)
sudo apt install liboqs-dev
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from vep_emmcse import VEP_eMMCSE, DataSource, Client

# Initialize the framework
scheme = VEP_eMMCSE()
setup_params = scheme.setup(security_level=128)

# Create data source and client
ds = DataSource("source_1", setup_params)
client = Client("client_1", setup_params)

# Add encrypted documents
documents = [
    ("doc_1", ["keyword1", "keyword2", "keyword3"]),
    ("doc_2", ["keyword2", "keyword4"]),
]

for doc_id, keywords in documents:
    ds.add_document(doc_id, keywords)

# Perform Boolean search
query = "(keyword1 AND keyword2) OR keyword4"
results = client.search(query)

# Verify results
verified_results = client.verify_results(results, query)
print(f"Found {len(verified_results)} verified documents")
```

### Performance Benchmarking

```python
from vep_emmcse.experiments import PerformanceBenchmark

# Run comprehensive benchmarks
benchmark = PerformanceBenchmark()
results = benchmark.run_all_experiments()

# Generate performance report
benchmark.generate_report("performance_results.html")
```

## 📊 Experimental Reproduction

Reproduce the results from the research paper:

```bash
# Download required datasets
python scripts/download_datasets.py

# Run full experimental suite
python scripts/run_experiments.py --config experiments/paper_reproduction.yaml

# Generate comparison tables and figures
python scripts/generate_paper_figures.py
```

Expected performance benchmarks:
- Update time (100 keywords): ~198ms
- Search time scaling: Linear with query complexity
- Verification overhead: ~201ms for 500-document result set

## 🏗️ Architecture

### Core Components

- **`crypto_primitives.py`**: Post-quantum cryptographic building blocks
- **`merkle_tree.py`**: Verifiable data structure implementation
- **`searchable_encryption.py`**: Core SSE protocols
- **`vep_emmcse.py`**: Main framework implementation

### Key Algorithms

1. **Setup**: Initialize system parameters and keys
2. **AggKey**: Generate client access keys
3. **Update**: Add/delete documents with PQ signatures
4. **Search**: Process Boolean queries with ABE-like indexes  
5. **Verify**: Validate results using Merkle proofs

## 🔒 Security Features

### Threat Model
- **Malicious Cloud Server**: May return incorrect or incomplete results
- **Quantum Adversaries**: Resistant to Shor's algorithm and variants
- **Adaptive Attacks**: Forward/backward privacy against inference

### Cryptographic Primitives
- **PQ-PRF**: HMAC-SHA3-256 construction
- **PQ-Signatures**: CRYSTALS-Dilithium3 (NIST Level 3)
- **Symmetric Encryption**: AES-256-GCM
- **Hash Functions**: SHA3-256

## 📈 Performance

Performance characteristics on reference hardware (i9-12900K, 64GB RAM):

| Operation | Baseline (ms) | VEP-eMMCSE (ms) | Overhead |
|-----------|---------------|------------------|----------|
| Update (10 keywords) | 12.91 | 20.18 | 1.56x |
| Search (4 clauses) | 575.21 | 575.21 | 1.0x |
| Verification (100 docs) | - | 40.28 | New feature |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vep_emmcse tests/

# Run specific test categories
pytest tests/test_crypto.py -v
pytest tests/test_schemes.py -v
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)**: Detailed setup instructions
- **[Usage Manual](docs/usage.md)**: Comprehensive API documentation
- **[Performance Guide](docs/performance.md)**: Optimization and benchmarking
- **[Security Analysis](docs/security.md)**: Formal security guarantees

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ahmadpanah/vep-emmcse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ahmadpanah/vep-emmcse/discussions)

## 🔮 Roadmap

- [ ] Hardware acceleration support
- [ ] Additional post-quantum primitives (SPHINCS+, FALCON)
- [ ] Range query support
- [ ] Distributed deployment tools
- [ ] GUI interface for non-technical users

---

**Disclaimer**: This is research software intended for academic and experimental use. Please conduct thorough security reviews before production deployment.