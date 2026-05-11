"""Skinny-N K-COMPLEMENT alias-stack admission set (residue-48 ladder).

This module enumerates the wave-misaligned skinny-N problem-shape rungs that
benefit from the K-COMPLEMENT alias-stack tile-shaping path on gfx942
(MI300X). All admitted N values satisfy ``N % 64 == 48`` — they straddle a
wave boundary by 48 columns and therefore alias-stack better than the default
persistent-GEMM dispatch.

Each successive rung was profiled and admitted in a separate task in the
K-2xxx series:

    Rung  1 (P45 / N=  880)  baseline admission
    Rung  2 (P46 / N=  944)
    Rung  3 (P47 / N= 1008)
    Rung  4 (P48 / N= 1072)
    Rung  5 (P49 / N= 1136)
    Rung  6 (P50 / N= 1200)
    Rung  7 (P51 / N= 1264)
    Rung  8 (P52 / N= 1328)
    Rung  9 (P53 / N= 1392)
    Rung 10 (P54 / N= 1456)
    Rung 11 (P55 / N= 1520)  K-2127 baseline
    Rung 12 (P56 / N= 1584)  K-2156 (this task)

The ladder is contiguous (stride 64) so membership can be tested either by
``N in KC_ALIAS_STACK_N_RES48`` (O(1) hash lookup, used by the dispatcher) or
by the equivalent arithmetic guard ``880 <= N <= 1584 and N % 64 == 48``.
The frozenset form is preferred so that future rungs can be admitted by a
single ``+1`` line diff without altering the arithmetic guard, and so that
any non-contiguous holes (should profiling later reveal one) can be expressed
without code surgery.
"""

KC_ALIAS_STACK_N_RES48 = frozenset({
     880,   # P45 rung  1
     944,   # P46 rung  2
    1008,   # P47 rung  3
    1072,   # P48 rung  4
    1136,   # P49 rung  5
    1200,   # P50 rung  6
    1264,   # P51 rung  7
    1328,   # P52 rung  8
    1392,   # P53 rung  9
    1456,   # P54 rung 10
    1520,   # P55 rung 11  (K-2127 baseline)
    1584,   # P56 rung 12  (K-2156 — this admission)
})
