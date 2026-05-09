"""K-1592 P26 `skinny_N1024` K-COMPLEMENT 30-cell alias frozenset.

P26 is a **documentation alias only** — `_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30`
is defined as the explicit set-union P16 ∪ {(2048,1024,4096,bf16)} and is
NOT wired into `_k971_route_to_hbl`.  Both upstream layers already cover
all 30 cells (P16 routes 29/30 at chain position 10; R-K979 P5 Clause-1
routes the 30th at position 4), so a separate predicate function and
chain entry would be unreachable dead code.

What this fixture pins:
  (1) cardinality is exactly 30 (P16's 29 + the 1 witness);
  (2) P26 is the set-union P16 ∪ {(2048,1024,4096,bf16)} — strict superset
      of P16, differing by exactly the K-1429-reject witness cell;
  (3) the K-1592 30-cell sweep grid matches the documented envelope
      M ∈ {2048, 4096, 8192} × N=1024 × K ∈ {2048, 4096, 8192, 16384,
      32768} × {bf16, fp16};
  (4) every P26 cell is routed by an upstream layer — for each cell, at
      least one of {P16 strict-equality, R-K979 P5 Clause-1} returns True;
  (5) the single P5-covered cell (2048,1024,4096,bf16) is admitted by
      R-K979 P5 Clause-1 (alias invariant rests on this);
  (6) NEGATIVE cases — cells outside the 30-element envelope are NOT in
      P26 (sibling-N rows, off-grid M / K, dtype carve-out);
  (7) NO predicate function symbol is exported (P26 is data, not a
      dispatch entry);
  (8) NO chain wiring — P26 symbol is not referenced in
      `_k971_route_to_hbl` / `k971_route_decision`.
"""
import ast
import inspect
import textwrap

import pytest

from tritonblas._route_predicate import (
    R_K979_P5_route_to_hbl,
    _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29,
    _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30,
)
from tritonblas import _route_predicate as _rp


# ---------------------------------------------------------------------------
# (1) cardinality
# ---------------------------------------------------------------------------
def test_p26_cardinality_is_thirty():
    assert len(_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30) == 30


# ---------------------------------------------------------------------------
# (2) P26 = P16 ∪ {witness}
# ---------------------------------------------------------------------------
def test_p26_is_strict_superset_of_p16():
    assert _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29.issubset(
        _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30)
    assert (_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30
            != _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)


def test_p26_minus_p16_is_exactly_the_p5_witness_cell():
    extra = (_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30
             - _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29)
    assert extra == frozenset({(2048, 1024, 4096, "torch.bfloat16")})


# ---------------------------------------------------------------------------
# (3) sweep grid
# ---------------------------------------------------------------------------
def test_p26_admit_set_is_full_n1024_kcompl_grid():
    expected = {
        (M, 1024, K, dtype)
        for M in (2048, 4096, 8192)
        for K in (2048, 4096, 8192, 16384, 32768)
        for dtype in ("torch.bfloat16", "torch.float16")
    }
    assert _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30 == expected


# ---------------------------------------------------------------------------
# (4) every P26 cell is covered by some upstream layer (P16 ⨄ P5 Clause-1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell", sorted(_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30)
)
def test_p26_alias_stack_covered_by_p16_or_p5(cell):
    M, N, K, dtype = cell
    in_p16 = cell in _K1429_P16_SKINNY_N1024_KCOMPL_ROUTEOUT_29
    in_p5 = R_K979_P5_route_to_hbl(M, N, K, dtype)
    assert in_p16 or in_p5, (
        f"P26 alias cell {cell} not covered by any upstream layer; "
        "alias invariant violated.")


# ---------------------------------------------------------------------------
# (5) the single P16-rejected cell is admitted structurally by P5 Clause-1
# ---------------------------------------------------------------------------
def test_p26_p16_reject_cell_is_admitted_by_p5_clause1():
    """(2048,1024,4096,bf16) — K-1429 reject at r=1.015 — sits in R-K979 P5
    Clause-1's mid-rect LDS-bound regime: minMN=1024 ∈ [256,2304],
    maxMN=2048 ∈ [1792,3072], K=4096 ∈ [1240,8064], bf16."""
    assert R_K979_P5_route_to_hbl(2048, 1024, 4096, "torch.bfloat16") is True


# ---------------------------------------------------------------------------
# (6) NEGATIVE cases — cells outside the envelope are NOT in P26.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        # sibling-N rows (envelope is N=1024 only)
        (2048,   128, 2048, "torch.bfloat16"),
        (2048,   256, 2048, "torch.bfloat16"),
        (2048,   512, 2048, "torch.bfloat16"),
        (2048,  2048, 2048, "torch.bfloat16"),
        (2048,  4096, 2048, "torch.bfloat16"),
        (2048, 16384, 2048, "torch.bfloat16"),
        (2048, 32768, 2048, "torch.bfloat16"),
        # off-grid K (envelope K ∈ {2048, 4096, 8192, 16384, 32768})
        (2048,  1024,  1024, "torch.bfloat16"),
        (2048,  1024,  3072, "torch.bfloat16"),
        (2048,  1024,  6144, "torch.bfloat16"),
        (2048,  1024, 12288, "torch.bfloat16"),
        (2048,  1024, 65536, "torch.bfloat16"),
        # off-grid M (envelope M ∈ {2048, 4096, 8192})
        (1024,  1024, 2048, "torch.bfloat16"),
        (3072,  1024, 2048, "torch.bfloat16"),
        (6144,  1024, 2048, "torch.bfloat16"),
        (16384, 1024, 2048, "torch.bfloat16"),
        # dtype carve-out — only bf16 / fp16 in envelope
        (2048,  1024, 2048, "torch.float32"),
        (2048,  1024, 2048, "torch.float8_e4m3fnuz"),
    ],
)
def test_p26_envelope_rejects_outside_cells(cell):
    """NEGATIVE case: every cell outside the 30-element envelope MUST NOT
    appear in P26 (alias must not silently absorb sibling-N rows, off-grid
    M/K, or unsupported dtypes)."""
    assert cell not in _K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30


# ---------------------------------------------------------------------------
# (7) NO predicate function exported (P26 is data, not a dispatch entry)
# ---------------------------------------------------------------------------
def test_p26_does_not_export_a_predicate_function():
    """Per the K-1605 reviewer Minimalist consensus: P26 is an alias
    frozenset only — there is no `_k1592_p26_..._routeout` callable. If a
    future change re-introduces the predicate, this guard catches it so the
    alias-vs-load-bearing distinction stays visible."""
    fn_names = [name for name in dir(_rp)
                if name.startswith("_k1592_p26") and callable(getattr(_rp, name, None))]
    assert fn_names == [], (
        f"unexpected K-1592 P26 predicate functions exported: {fn_names}; "
        "P26 must remain a documentation-only alias.")


# ---------------------------------------------------------------------------
# (8) NO chain wiring — `_k971_route_to_hbl` / `k971_route_decision` does
#     not consult P26.  This is enforced by source inspection so a future
#     edit that wires P26 into the dispatch chain fails loudly here rather
#     than turning the alias into silent dead code.
# ---------------------------------------------------------------------------
def _executable_body_unparsed(fn):
    """Return the function body unparsed back to source, with the docstring
    (first Expr/Constant statement) stripped so docstring text is excluded."""
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fn_node = tree.body[0]  # FunctionDef
    body = fn_node.body
    if (body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_p26_symbol_not_used_in_dispatch_chain():
    """Inspect the executable body (docstring stripped via AST) of
    `k971_route_decision` and assert P26 is neither referenced nor invoked.
    The function's docstring intentionally documents P26 as an alias-only
    handle — we don't want to forbid that.  What we forbid is any
    executable reference: `_K1592_P26_…` mentioned in code, or
    `_k1592_p26_…` invoked as a callable."""
    body_only = _executable_body_unparsed(_rp.k971_route_decision)
    assert "_K1592_P26_SKINNY_N1024_KCOMPL_ALIASSTACK_30" not in body_only, (
        "k971_route_decision body references the K-1592 P26 frozenset — "
        "the alias was wired into dispatch.  P16 + R-K979 P5 already cover "
        "every P26 cell at earlier chain positions, so wiring P26 "
        "introduces dead code.")
    assert "_k1592_p26" not in body_only.lower(), (
        "k971_route_decision body invokes a `_k1592_p26_*` callable — "
        "P26 must remain a documentation-only alias.")
