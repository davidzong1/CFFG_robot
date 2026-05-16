import torch

from class_free_guide.network.base.transformer_block.attention import AttentionBlock


def test_attention_block_self_attention_shape_and_finite():
    torch.manual_seed(0)
    block = AttentionBlock(hidden_dim=32, num_attention_heads=4, max_token_length=16)
    hidden = torch.randn(2, 8, 32)

    out = block(hidden)

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


def test_attention_block_cross_attention_shape_and_mask():
    torch.manual_seed(1)
    block = AttentionBlock(hidden_dim=32, cross_hidden_dim=24, num_attention_heads=4, max_token_length=16)
    hidden = torch.randn(2, 6, 32)
    cross = torch.randn(2, 5, 24)

    query_mask = torch.ones(2, 6, dtype=torch.bool)
    key_mask = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    mask2d = query_mask.unsqueeze(2) & key_mask.unsqueeze(1)
    out = block(hidden, cross_input=cross, mask2d=mask2d)

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_attention_block_self_attention_shape_and_finite()
    print("Self-attention shape and finite test passed.")
    test_attention_block_cross_attention_shape_and_mask()
    print("Cross-attention shape and mask test passed.")
