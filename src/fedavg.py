
import torch


def fedavg(state_dicts, sample_counts):
    """
    Aggregate client model weights using sample-weighted averaging.

    Args:
        state_dicts: List of client model state dictionaries.
        sample_counts: Number of training samples per client.

    Returns:
        Aggregated global model state dictionary.
    """
    if not state_dicts:
        raise ValueError("No client models provided.")

    if len(state_dicts) != len(sample_counts):
        raise ValueError("Number of models and sample counts must match.")

    if any(count <= 0 for count in sample_counts):
        raise ValueError("Sample counts must be positive.")

    keys = state_dicts[0].keys()

    for state in state_dicts[1:]:
        if state.keys() != state_dicts[0].keys():
            raise ValueError("Client models have different parameters.")

    total_samples = sum(sample_counts)
    global_state = {}

    for key in keys:
        first_tensor = state_dicts[0][key]

        if torch.is_floating_point(first_tensor):
            averaged = torch.zeros_like(
                first_tensor, dtype=torch.float32, device="cpu"
            )

            for state, count in zip(state_dicts, sample_counts):
                weight = count / total_samples
                averaged += state[key].detach().cpu().float() * weight

            global_state[key] = averaged.to(dtype=first_tensor.dtype)
        else:
            # Keep non-floating buffers from the first client.
            global_state[key] = first_tensor.detach().cpu().clone()

    return global_state


if __name__ == "__main__":
    client_states = [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([3.0, 4.0])},
    ]

    counts = [3, 1]
    result = fedavg(client_states, counts)

    print("FedAvg test")
    print("Aggregated weights:", result["weight"])
    print("Expected weights: tensor([1.5, 2.5])")

    assert torch.allclose(
        result["weight"], torch.tensor([1.5, 2.5])
    )
    print("FedAvg test passed.")
