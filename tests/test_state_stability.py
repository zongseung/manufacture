"""Identified state labels and seed diagnostics on strongly separated synthetic states."""
import numpy as np
from test_hmm import synthetic_panel

from gmst import hmm


def test_three_seed_summaries_preserve_order_and_state_agreement():
    # Given three well separated within-day residual regimes.
    p = synthetic_panel()
    p['Y'] = np.tile(np.repeat([-20., 0., 20.], 32), (28, 1)) + np.random.default_rng(0).normal(0, .25, (28, 96))
    # When fitting three initialized models using the same fixed zero centre.
    result = hmm.state_stability(p, m=np.zeros_like(p['Y']), max_epochs=8, device='cpu')
    # Then labels remain ordered and posthoc paths agree across seeds.
    assert len(result['rows']) == 3
    assert all(row['same_order'] and row['agreement'] > .9 for row in result['rows'])
    assert all(np.isclose(summary['occ'].sum(), 1) for summary in result['per_seed'].values())


def test_transition_table_averages_each_hours_four_quarters():
    # Given a conditional model and one operating weekday cell.
    p = synthetic_panel()
    model = hmm.CondHMM(seed=2).double()
    # When constructing the interpretability table.
    table = hmm.transition_table(model, p, np.array([2]))
    # Then each hour and originating state is represented with valid probabilities.
    assert table.height == 24 * 3
    assert table['p_to_high'].min() >= 0 and table['p_to_high'].max() <= 1
    assert hmm.conclusion(table, 3)[:2] == (1, 'wk')
