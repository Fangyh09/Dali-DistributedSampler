
# Dali-DistributedSampler
A generic DistributedSampler for Dali, support all kinds of input source. 

# Why it?
Dali does not provide a generic DistributedSampler currently.

# Usage
4 gpus, with total dataset size 8000
```python
from sampler import DistributedDaliSampler
rankid = get_rankid()
sampler = DistributedDaliSampler(dataset_size=8000, num_replicas=4, rank=rankid)
self.iter = iter(sampler)

for _ in range(self.batch_size):
    cur_index = next(self.iter)
```
