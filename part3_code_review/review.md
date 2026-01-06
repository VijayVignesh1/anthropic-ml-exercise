# Code Review

## Bugs & Optimizations

Code Bugs:

* Missing optimizer.zero\_grad(). It can lead to gradient accumulation.  
* No batch\_first=True in TransformerEncoderLayer input shape mismatch crash  
* No positional encodings which means transformer learns nothing (it is position blind)  
* Missing model.train()   
* No padding masks → padded tokens corrupt attention  
* No device handling → CPU/GPU tensor mismatch

Optimizations:

* Adding batch\_first=True can lead to memory reduction and proper GPU utilization  
* Mixed precision (AMP) leading to increased training speed and reduced memory

Bottlenecks:

* Sequential CPU to GPU transfer.  
* No padding masks leading to wasted computation on pad tokens  
* Fixed small batches leading to poor GPU utilization

## Best Practices 

Code Quality:

* Add type hints, docstrings, input validation  
* Use torch.manual\_seed() for reproducibility  
* Proper logging instead of print()  
* Configuration via dataclass/argparse
