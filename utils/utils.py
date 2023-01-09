import torch

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32)
    target = torch.nn.functional.one_hot(torch.randint(0, 10, (10,)), 10)
    print(target)
    out = mixup(x, target, 0.5)
    print(out[1])
    print(out[0].shape, out[1].shape)