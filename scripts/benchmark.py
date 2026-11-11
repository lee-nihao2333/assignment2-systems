import cs336_basics.model
import cs336_basics.optimizer
import cs336_basics.nn_utils
import cs336_basics.data
import torch
import timeit
import numpy as np

class CFG:
    model_size = "medium"
    context_length = 512

    d_model = {
        "small": 768,
        "medium": 1024,
        "large": 1280,
        "xl": 2560,
        "10B": 4608
    }

    d_ff = {
        "small": 3072,
        "medium": 4096,
        "large": 5120,
        "xl": 10240,
        "10B": 12288
    }

    num_layers = {
        "small": 12,
        "medium": 24,
        "large": 36,
        "xl": 32,
        "10B": 50
    }

    num_heads = {
        "small": 12,
        "medium": 16,
        "large": 20,
        "xl": 32,
        "10B": 36
    }

    batch_size = 4
    vocab_size = 10000
    device = "cuda:0"

    warmup = 5
    test_count = 10
    epochs = 10

print("test model size:", CFG.model_size)

model = cs336_basics.model.BasicsTransformerLM(
    vocab_size=CFG.vocab_size,
    context_length=CFG.context_length,
    d_model=CFG.d_model[CFG.model_size],
    num_layers=CFG.num_layers[CFG.model_size],
    num_heads=CFG.num_heads[CFG.model_size],
    d_ff=CFG.d_ff[CFG.model_size],
).to(CFG.device)

dataset = np.array(range(CFG.vocab_size))
data, target = cs336_basics.data.get_batch(dataset,CFG.batch_size, CFG.context_length, CFG.device)

with torch.no_grad(): 
    for _ in range(CFG.warmup): 
        output = model(data)

torch.cuda.synchronize()


test_time = []

with torch.no_grad(): 
    for _ in range(CFG.test_count): 
        start = timeit.default_timer()
        output = model(data)
        torch.cuda.synchronize()
        now = timeit.default_timer()

        test_time.append(now-start)
        # print(f"forward {i+1}: {time[i]}s")

test_time = np.array(test_time)

print("test time:")
print("sum:", test_time.sum())
print("mean:", test_time.mean())
print("std:", np.std(test_time))

optimizer = cs336_basics.optimizer.AdamW(model.parameters(), lr=0.001)

for epoch in range(CFG.warmup):
    output = model(data)
    loss = cs336_basics.nn_utils.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

forward_time = []
backward_time = []
optimizer_step_time  = []

for epoch in range(CFG.epochs):
    torch.cuda.synchronize()
    start = timeit.default_timer()
    output = model(data)
    torch.cuda.synchronize()
    now = timeit.default_timer()

    forward_time.append(now-start)

    start = timeit.default_timer()
    loss = cs336_basics.nn_utils.cross_entropy(output, target)
    loss.backward()
    torch.cuda.synchronize()
    now = timeit.default_timer()

    backward_time.append(now-start)

    start = timeit.default_timer()
    optimizer.step()
    torch.cuda.synchronize()
    now = timeit.default_timer()

    optimizer_step_time.append(now-start)
    optimizer.zero_grad()

    # print("loss:", loss)

forward_time = np.array(forward_time)

print("forward time:")
print("sum:", forward_time.sum())
print("mean:", forward_time.mean())
print("std:", np.std(forward_time))

backward_time = np.array(backward_time)

print("backward time:")
print("sum:", backward_time.sum())
print("mean:", backward_time.mean())
print("std:", np.std(backward_time))

optimizer_step_time = np.array(optimizer_step_time)

print("optimizer step time:")
print("sum:", optimizer_step_time.sum())
print("mean:", optimizer_step_time.mean())
print("std:", np.std(optimizer_step_time))

train_time = forward_time + backward_time + optimizer_step_time

print("train time:")
print("sum:", train_time.sum())
print("mean:", train_time.mean())
print("std:", np.std(train_time))