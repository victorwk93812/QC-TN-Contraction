"""fast_elim.eliminate_fast must reproduce core.eliminate step for step."""
import networkx as nx
import pytest

from traces import graphs as gr
from traces.core import column_sweep, eliminate
from traces.fast_elim import eliminate_fast

GRAPHS = {"grid6_int": gr.grid_int(6), "grid5": gr.grid(5), "cubic30": gr.cubic(30, seed=2),
          "sep": gr.separator_graph(), "petersen": nx.petersen_graph()}


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("rule", ["minfill", "mindeg"])
@pytest.mark.parametrize("seed", range(4))
def test_same_trace_greedy(name, rule, seed):
    G = GRAPHS[name]
    assert eliminate_fast(G, rule=rule, seed=seed) == eliminate(G, rule=rule, seed=seed)


def test_same_trace_fixed_order():
    G = gr.grid(6)
    assert eliminate_fast(G, order=column_sweep(6)) == eliminate(G, order=column_sweep(6))
