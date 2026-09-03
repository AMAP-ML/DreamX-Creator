"""Per-layer LQ-anchor influence, measured without touching the attention code.

The question this answers: *which blocks is the anchor actually driving?* The anchor
and the rolling KV cache share one softmax, so whatever mass the anchor takes is mass
the history lost. If that competition is what weakens generation, it will not be
uniform across depth — and knowing where it concentrates is what picks the k for
``--lq_anchor_layers``.

The honest measurement would be ``Σp_anchor`` read straight out of the softmax, but
that lives inside ``wan/modules/sr_dit/attention.py``, which the queued H20 training
arms execute (the job runs the live NAS workdir at dequeue time, not the uploaded
snapshot). So this measures the *counterfactual* instead, from a forward hook:

    rel = ||out_with_anchor - out_without_anchor|| / ||out_without_anchor||

i.e. how far this block's attention output moves if this block alone loses its anchor,
with its input held fixed. It is not ``Σp_anchor`` — a block can carry a lot of anchor
mass and still barely move its output if the anchor's V happens to agree with the
history's. Read it as "how much this block's output *depends* on the anchor", which is
the quantity the k sweep trades away.

Why re-invoking the module is safe here: ``WanSelfAttention.forward`` writes only its
``_flex_*`` mask cache (keyed, so a repeat is idempotent), never mutates its inputs,
and hands the KV cache back through its return value rather than appending in place.
Three properties are asserted by the tests and enforced below:

* the host run's output stays bit-identical (a probe that perturbs the run measures
  something other than the run);
* no recursion — calling a module inside its own forward hook would otherwise
  re-enter the hook forever;
* no RNG consumption — inference is seeded, and stealing draws would change the noise
  of every later chunk.
"""

import torch

__all__ = ["AnchorInfluenceProbe"]


class AnchorInfluenceProbe:
    """Record per-block anchor influence via counterfactual forward hooks.

    Args:
        blocks: iterable of transformer blocks (``model.blocks``).
        attr: attribute name of the self-attention submodule on each block.
        eps: floor on the denominator of the relative norm.

    Use as a context manager, then read :meth:`summary`. Blocks that never received an
    anchor are absent from the summary.
    """

    def __init__(self, blocks, attr="self_attn", eps=1e-12):
        self.blocks = list(blocks)
        self.attr = attr
        self.eps = float(eps)
        self._handles = []
        self._busy = False
        self._records = {}

    # ────────────────────────────── lifecycle ──────────────────────────────
    def attach(self):
        if self._handles:
            return self
        for idx, block in enumerate(self.blocks):
            module = getattr(block, self.attr, None)
            if module is None:
                raise AttributeError(
                    f"block {idx} has no {self.attr!r} submodule; pass attr= for this "
                    f"architecture")
            self._handles.append(module.register_forward_hook(
                self._make_hook(idx), with_kwargs=True))
        return self

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    # ──────────────────────────────── hook ────────────────────────────────
    def _make_hook(self, idx):
        def hook(module, args, kwargs, output):
            # Re-entrancy guard: the counterfactual call below goes through this same
            # module, so without this the hook would fire on itself forever.
            if self._busy:
                return None
            if kwargs.get("anchor_k", None) is None:
                return None      # this block got no anchor — nothing to compare

            with_anchor = _primary(output)
            off_kwargs = dict(kwargs)
            off_kwargs["anchor_k"] = None
            off_kwargs["anchor_v"] = None

            self._busy = True
            try:
                with torch.no_grad(), _frozen_rng(with_anchor.device):
                    without = _primary(module(*args, **off_kwargs))
            finally:
                self._busy = False

            if without.shape != with_anchor.shape:
                raise RuntimeError(
                    f"block {idx}: dropping the anchor changed the output shape "
                    f"{tuple(with_anchor.shape)} -> {tuple(without.shape)}; the probe "
                    f"cannot compare these")

            a = with_anchor.detach().to(torch.float64)
            b = without.to(torch.float64)
            denom = torch.linalg.vector_norm(b)
            rel = (torch.linalg.vector_norm(a - b)
                   / denom.clamp_min(self.eps)).item()
            self._records.setdefault(idx, []).append(rel)
            # Return None: the host keeps its own output untouched.
            return None

        return hook

    # ─────────────────────────────── read-out ───────────────────────────────
    def summary(self):
        """``{block_index: {"n": calls, "rel": mean, "rel_per_call": [...]}}``.

        JSON-serialisable (plain ints/floats/lists) so it can be dumped next to the
        run's videos.
        """
        out = {}
        for idx, rels in sorted(self._records.items()):
            out[int(idx)] = {
                "n": len(rels),
                "rel": float(sum(rels) / len(rels)),
                "rel_max": float(max(rels)),
                "rel_per_call": [float(r) for r in rels],
            }
        return out

    def reset(self):
        self._records = {}
        return self


def _primary(output):
    """The attention output tensor. The streaming path returns ``(y, cache_k, cache_v)``."""
    while isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("self-attention returned an empty sequence")
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(
            f"expected the self-attention output to be a tensor, got {type(output).__name__}")
    return output


class _frozen_rng:
    """Restore CPU (and, if relevant, CUDA) RNG state on exit.

    The counterfactual call runs the same code as the host, which may draw random
    numbers. Inference is seeded, so a stolen draw would silently change the noise of
    every subsequent chunk — the probe would rewrite the very video it is measuring.
    """

    def __init__(self, device):
        self.device = torch.device(device) if device is not None else None

    def __enter__(self):
        self._cpu = torch.get_rng_state()
        self._cuda = None
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            self._cuda = torch.cuda.get_rng_state(self.device)
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state(self._cuda, self.device)
        return False
