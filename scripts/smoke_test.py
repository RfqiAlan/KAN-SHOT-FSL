"""
Smoke test for KANProtoNet, MLPProtoNet, KANLinear, and gradient flow isolation.
Tests all critical components end-to-end with synthetic tensors.
"""
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fhist'))
from src.kan.kan_layer import BSplineBasis, KANLinear, KANHead
from src.methods.kan_protonet import KANProtoNet
from src.methods.mlp_protonet import MLPProtoNet


def test_kan_layers():
    print("--- 1. Testing KAN Base Layers ---")
    x = torch.randn(4, 512)
    
    basis = BSplineBasis(grid_size=5, spline_order=3)
    b_out = basis(x)
    assert b_out.shape == (4, 512, 8), f"Expected (4, 512, 8), got {b_out.shape}"
    print("✅ BSplineBasis forward OK.")

    kan_lin = KANLinear(512, 512, grid_size=5, spline_order=3)
    out = kan_lin(x)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"
    print(f"✅ KANLinear forward OK. Params = {kan_lin.num_parameters:,}")

    kan_head = KANHead(512, 512, hidden_features=256)
    head_out = kan_head(x)
    assert head_out.shape == (4, 512), f"Expected (4, 512), got {head_out.shape}"
    print(f"✅ KANHead 2-layer forward OK. Params = {kan_head.num_parameters:,}")


def test_methods_and_gradient_flow():
    print("\n--- 2. Testing KANProtoNet & MLPProtoNet with Frozen Backbone ---")
    device = torch.device("cpu")

    # ResNet18 backbone
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    args = argparse.Namespace(
        kan_mode='transform',
        kan_temperature=1.0,
        feat_dim=512,
        kan_out_dim=512,
        kan_grid_size=5,
        kan_spline_order=3,
        kan_hidden=None,
        kan_dropout=0.0
    )

    # 1. KANProtoNet
    kan_method = KANProtoNet(args).to(device)
    print(f"✅ KANProtoNet params = {kan_method.count_kan_parameters():,}")

    # 2. MLPProtoNet
    mlp_method = MLPProtoNet(args).to(device)
    print(f"✅ MLPProtoNet params = {mlp_method.count_parameters():,}")

    # Synthetic 5-way 5-shot episode with 15 query per class (25 support, 75 query)
    x_s = torch.randn(1, 25, 3, 84, 84)
    x_q = torch.randn(1, 75, 3, 84, 84)
    y_s = torch.tensor([[0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5])
    y_q = torch.tensor([[0]*15 + [1]*15 + [2]*15 + [3]*15 + [4]*15])

    # Test KANProtoNet forward + backward + gradient isolation
    opt_kan = torch.optim.Adam(kan_method.parameters(), lr=1e-3)
    opt_kan.zero_grad()
    loss_k, preds_k = kan_method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)
    assert preds_k.shape == (1, 75), f"Expected (1, 75), got {preds_k.shape}"
    loss_k.mean().backward()

    # Assert backbone receives ZERO gradients
    assert all(p.grad is None for p in model.parameters()), "❌ Backbone received gradient!"
    # Assert KAN receives gradients
    assert any(p.grad is not None for p in kan_method.parameters()), "❌ KAN received no gradient!"
    
    before_k = {n: p.detach().clone() for n, p in kan_method.named_parameters()}
    opt_kan.step()
    changed_k = [not torch.equal(before_k[n], p.detach()) for n, p in kan_method.named_parameters()]
    assert any(changed_k), "❌ KAN parameters did not change after optimizer step!"
    print("✅ KANProtoNet forward, backward, freeze assertion & parameter update PASSED!")

    # Test MLPProtoNet forward + backward + gradient isolation
    opt_mlp = torch.optim.Adam(mlp_method.parameters(), lr=1e-3)
    opt_mlp.zero_grad()
    loss_m, preds_m = mlp_method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)
    assert preds_m.shape == (1, 75), f"Expected (1, 75), got {preds_m.shape}"
    loss_m.mean().backward()

    assert all(p.grad is None for p in model.parameters()), "❌ Backbone received gradient during MLP!"
    assert any(p.grad is not None for p in mlp_method.parameters()), "❌ MLP received no gradient!"
    
    before_m = {n: p.detach().clone() for n, p in mlp_method.named_parameters()}
    opt_mlp.step()
    changed_m = [not torch.equal(before_m[n], p.detach()) for n, p in mlp_method.named_parameters()]
    assert any(changed_m), "❌ MLP parameters did not change after optimizer step!"
    print("✅ MLPProtoNet forward, backward, freeze assertion & parameter update PASSED!")


if __name__ == "__main__":
    test_kan_layers()
    test_methods_and_gradient_flow()
    print("\n🎉 ALL SMOKE TESTS PASSED SUCCESSFULLY!")
