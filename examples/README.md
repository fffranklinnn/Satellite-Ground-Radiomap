# Examples Directory

This directory contains example scripts demonstrating how to use the SG-MRM system.

## Available Examples

### 1. basic_usage.py

Basic usage example showing how to:
- Create layers programmatically
- Initialize the aggregator
- Compute a single radio map
- Visualize results

**Run:**
```bash
python examples/basic_usage.py
```

### 2. v1_static_link.py

V1.0 milestone example demonstrating static link closure with:
- L1 Macro Layer (satellite positioning)
- L3 Urban Layer (building shadows)
- Composite map generation

**Run:**
```bash
python examples/v1_static_link.py
```

## Output

All examples save their output to the `output/examples/` or `output/v1_static_link/` directories.

Output includes:
- PNG visualizations of radio maps
- Layer comparison plots
- Raw numpy arrays (.npy files) for further processing

## Creating Custom Examples

To create your own example:

1. Copy one of the existing examples
2. Modify the configuration parameters
3. Add your custom processing logic
4. Run and visualize results

Example template:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import L1MacroLayer
from src.engine import RadioMapAggregator
from src.utils import plot_radio_map

def main():
    # Your code here
    pass

if __name__ == '__main__':
    main()
```

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```
