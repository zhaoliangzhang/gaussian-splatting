import torch
import torch.utils.benchmark as benchmark
from utils.prune_utils import _gumbel_sigmoid

x = torch.rand(1000).cuda()
t0 = benchmark.Timer(stmt='_gumbel_sigmoid(x)',
                    setup='from utils.prune_utils import _gumbel_sigmoid',
                    globals={'x': x},
                    )

render_time0 = t0.timeit(100)
print(render_time0.median)

t1 = benchmark.Timer(stmt='torch.sigmoid(x)',
                    setup='import torch',
                    globals={'x': x},
                    )

render_time1 = t1.timeit(100)
print(render_time1.median)                    