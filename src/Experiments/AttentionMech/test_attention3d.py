import torch
from Attention3d import CoordAtt3D, SE3D, make_attn

torch.manual_seed(0)
# Non-cubic on purpose: a cube hides D/H/W mix-ups.
X = torch.randn(2, 16, 32, 24, 8)


def count(m):
    return sum(p.numel() for p in m.parameters())


for pooling in ("avg", "avgmax"):
    print(f"\n===== CoordAtt3D, pooling={pooling} =====")
    m = CoordAtt3D(16, reduction=32, pooling=pooling)

    # 1. shape
    y = m(X)
    assert y.shape == X.shape, y.shape
    print("shape OK:", tuple(y.shape))

    # 2. gate range: output / input must lie in (0, 1) wherever input is not ~0
    ratio = (y / X)[X.abs() > 1e-3]
    assert ratio.min() > 0 and ratio.max() < 1, (ratio.min().item(), ratio.max().item())
    print(f"gate range OK: [{ratio.min().item():.3f}, {ratio.max().item():.3f}]")

    # 3. initial gate scale (product of three sigmoids)
    print(f"mean combined gate at init (gate_bias=0): {ratio.mean().item():.3f}")
    m_hi = CoordAtt3D(16, reduction=32, pooling=pooling, gate_bias=2.0)
    r_hi = (m_hi(X) / X)[X.abs() > 1e-3]
    print(f"mean combined gate at init (gate_bias=2): {r_hi.mean().item():.3f}")

    # 4. roll test: no spatial convs, so rolling the input along an axis rolls the output
    m.eval()
    for dim, name in ((2, "D"), (3, "H"), (4, "W")):
        a = m(X.roll(3, dims=dim))
        b = m(X).roll(3, dims=dim)
        print(f"roll along {name}: equivariant = {torch.allclose(a, b, atol=1e-5)}")

    # 5. gradients reach every parameter
    m.train()
    m.zero_grad()
    m(X).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing
    print("gradients OK for all parameters")

    # 6. parameter count at the channel sizes used in the network
    for c in (16, 32, 64):
        print(f"params at {c:>2} channels: {count(CoordAtt3D(c, 32, pooling)):>6}"
              f"   (SE3D: {count(SE3D(c, 32)):>5})")

# make_attn
assert isinstance(make_attn(None, 16), torch.nn.Identity)
assert isinstance(make_attn("se", 16), SE3D)
assert make_attn("ca", 16).pooling == "avg"
assert make_attn("ca_max", 16).pooling == "avgmax"
print("\nmake_attn OK")
