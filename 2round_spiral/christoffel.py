import torch

def compute_jacobian(v, x):
    """
    v: [B, d]
    x: [B, d] (requires grad)
    returns: J_v [B, d, d]
    """
    B, d = x.shape
    J = []

    for i in range(d):
        grad_i = torch.autograd.grad(
            v[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]
        J.append(grad_i)

    # J: list of [B, d] -> [B, d, d]
    return torch.stack(J, dim=1)


def compute_metric(Jv, alpha=1.0):
    """
    Induced metric: g = I + alpha * J_v^T J_v
    """
    B, d, _ = Jv.shape
    I = torch.eye(d, device=Jv.device).unsqueeze(0).expand(B, d, d)
    g = I + alpha * torch.bmm(Jv.transpose(1, 2), Jv)
    return g


def compute_christoffel(g, x):
    """
    Computes Christoffel symbols Γ^k_{ij}
    g: [B, d, d]
    x: [B, d] (requires grad)
    returns: Γ [B, d, d, d]
    """
    B, d, _ = g.shape
    g_inv = torch.inverse(g)

    # ∂_i g_{jl}
    dg = []
    for j in range(d):
        for l in range(d):
            grad_jl = torch.autograd.grad(
                g[:, j, l].sum(),
                x,
                create_graph=True,
                retain_graph=True
            )[0]
            dg.append(grad_jl)

    dg = torch.stack(dg, dim=1).view(B, d, d, d)  # [B, j, l, i]

    Gamma = torch.zeros(B, d, d, d, device=x.device)

    for k in range(d):
        for i in range(d):
            for j in range(d):
                term = 0
                for l in range(d):
                    term += g_inv[:, k, l] * (
                        dg[:, j, l, i] +
                        dg[:, i, l, j] -
                        dg[:, i, j, l]
                    )
                Gamma[:, k, i, j] = 0.5 * term

    return Gamma


def christoffel_correction(v, x, alpha=1.0):
    """
    Returns Γ[v, v]
    """
    Jv = compute_jacobian(v, x)
    g = compute_metric(Jv, alpha)
    Gamma = compute_christoffel(g, x)

    # Γ[v, v]
    # v: [B, d]
    corr = torch.einsum('bkij,bi,bj->bk', Gamma, v, v)
    return corr
