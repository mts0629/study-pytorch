"""Measure the model's runtime and memory consumption by the profiler."""

import torch
import torchvision.models as models

from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == "__main__":
    model = models.resnet18()

    inputs = torch.randn(8, 3, 224, 224)  # 8 batches

    # Get a profile of CPU activities (with shape info)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):  # Label with "model_inference"
            model(inputs)

    # Print stats with sorting by total CPU time
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # With input shapes
    print(
        prof.key_averages(group_by_input_shape=True)\
        .table(sort_by="cpu_time_total", row_limit=10)
    )

    # Get a profile with memory consumption
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        model(inputs)

    # Sort by a memory amount allocated by the operator itself
    print(
        prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
    )
    # Sort by a memory amount allocated by the operator and its children
    print(
        prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)
    )

    # Get a profile of CPU and GPU activities (with shape info)
    ### Doesn't work on WSL? ###
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True
    # ) as prof:
    #     with record_function("model_inference"):
    #         model(inputs)
    # Sort by total GPU time
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export a trace file for the Chrome trace viewer
    prof.export_chrome_trace("./work/trace.json")

    # Python and TorchScript stack traces
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        model(inputs)

    print(
        prof.key_averages(group_by_stack_n=5)\
            .table(sort_by="self_cpu_time_total", row_limit=2)
    )
