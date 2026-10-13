"""Invariants of the algorithms that later figures rely on."""
import networkx as nx

from traces import graphs as gr
from traces.carving import tree_stats
from traces.circuits import circuit_tn, default_circuit, simple_tensor_graph
from traces.core import index_graph, separator_report
from traces.flowcutter_lite import flowcutter
from traces.gn import girvan_newman


def test_flowcutter_7x7_front():
    G = gr.grid(7)
    fc = flowcutter(G, (0, 0), (6, 6))
    for s in fc["steps"]:
        assert separator_report(G, s["cut"])["separator"]
    sizes = [s["size"] for s in fc["steps"]]
    assert sizes == sorted(sizes)
    front = [(fc["steps"][i]["size"], fc["steps"][i]["small_side"]) for i in fc["pareto"]]
    assert front == [(k, k * (k - 1) // 2) for k in range(2, 8)]   # diagonal corner cuts


def test_girvan_newman_dendrogram_is_contraction_tree():
    T = gr.bowtie_tn()
    TG = nx.Graph([(a, b) for a, b, _ in gr.BOWTIE_BONDS])
    g = girvan_newman(TG)
    assert len(g["steps"]) == TG.number_of_edges()
    assert len(g["splits"]) == TG.number_of_nodes() - 1
    st = tree_stats(T, g["tree"])
    assert st["max_rank"] == 2


def test_default_circuit():
    T = circuit_tn(default_circuit())
    assert len(T) == 33 and len(index_graph(T)) == 54
    assert max(len(v) for v in T.values()) == 4
    assert all(d["weight"] == 1 for *_, d in simple_tensor_graph(T).edges(data=True))
