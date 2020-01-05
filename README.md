# Dali-DistributedSampler

# Usage
4 gpus, with total dataset size 8000
```python
rankid = get_rankid()
sampler = DistributedDaliSampler(dataset_size=8000, num_replicas=4, rank=rankid)
self.iter = iter(sampler)

for _ in range(self.batch_size):
    cur_index = next(self.iter)
```
