# Documentation

This directory contains comprehensive documentation for the SG-MRM project.

## Available Documentation

### [Architecture Guide](architecture.md)

Detailed description of the system architecture, design principles, and layer implementations.

### [API Reference](api_reference.md)

Complete API documentation for all modules, classes, and functions.

### [Development Guide](development_guide.md)

Guide for developers contributing to the project, including coding standards and testing procedures.

## Quick Links

- [Main README](../README.md)
- [Configuration Guide](../configs/README.md)
- [Data Guide](../data/README.md)
- [Examples](../examples/README.md)

## Additional Resources

### ITU-R Recommendations

- [ITU-R P.618](https://www.itu.int/rec/R-REC-P.618/): Propagation data and prediction methods
- [ITU-R P.531](https://www.itu.int/rec/R-REC-P.531/): Ionospheric propagation data
- [ITU-R P.526](https://www.itu.int/rec/R-REC-P.526/): Propagation by diffraction

### External Documentation

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

## Contributing to Documentation

When adding new features, please update the relevant documentation:

1. Add docstrings to all new functions and classes
2. Update API reference if needed
3. Add examples for new functionality
4. Update architecture guide for structural changes

## Building Documentation

For generating HTML documentation (future):

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```
