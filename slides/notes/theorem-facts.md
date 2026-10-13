# Theorem statements, checked against sources (Oct 1, 2026)

Checked by a research pass over primary papers/abstracts (arXiv, Crossref, OpenAlex, GitHub).
"(secondary)" = confirmed only through another paper. Use these forms on slides.

## Act I–II
- **Markov–Shi**, SIAM J. Comput. 38(3):963–981 (2008); arXiv quant-ph/0511069v7. Prop. 4.2: cc(G) = tw(G*) (G* = line graph).
  Thm 1.1 / 4.6: simulation in T^{O(1)} exp[O(tw(G_C))], tw of the *circuit graph*. Lemma 4.4/Thm 4.5: (tw(G)−1)/2 ≤ tw(G*) ≤ Δ(G)(tw(G)+1)−1.
- **Arnborg–Corneil–Proskurowski**, SIAM J. Alg. Disc. Meth. 8(2):277–284 (1987): deciding tw ≤ k (k in the input) is NP-complete; O(n^{k+2}) for fixed k.
- **Excluded grid**: Robertson–Seymour, JCTB 41 (1986). Chekuri–Chuzhoy, J. ACM 63(5) (2016): polynomial f, δ ≥ 1/98. Best known: Chuzhoy–Tan, JCTB 146 (2021): tw ≥ O(g⁹ polylog g) ⇒ g×g grid minor; lower bound Ω(g² log g).
- **cw ≥ Δ**: Belmonte et al., DAM 2013, Obs. 1. **Step cost ≤ χ^{1.5 cw}**: O'Gorman, TQC 2019 (LIPIcs 135:10), Thm 1: time n·2^{1.5 ec(G)+1}, space 2^{ec(G)+1} (ec = carving width). Uses ec ≤ vc ≤ 1.5 ec (Bienstock 1990).
- **Line graphs**: Harvey–Wood, JCTB 132:157–179 (2018): tw(L(G)) ≥ Δ(G)−1; tw(L(G)) ≥ tw(G)−1 (Prop 2.3); tw(L(G)) ≤ (tw(G)+1)Δ(G)−1 (eq. 2, Atserias/Bienstock/Călinescu–Fernandes–Reed).
- **Carving vs line graph**: Dudek–Dueñas-Osorio–Vardi (arXiv:1908.04381) Thm 3: contraction tree of max rank w ⟺ carving decomposition of width w; Thm 4: cw(G) ≤ tw(L(G))+1. With vc ≤ 1.5 ec: cw ≤ tw(L(G))+1 ≤ 1.5 cw.
- **Chordal / PEO**: Fulkerson–Gross, Pacific J. Math. 15(3):835–855 (1965) §7; "perfect elimination" term: Rose 1970.
- **Yannakakis**, SIAM J. Alg. Disc. Meth. 2(1):77–79 (1981): minimum fill-in NP-complete.
- **Simplicial rule**: Bodlaender–Koster–van den Eijkhof, Comput. Intell. 21(3) (2005), Rule 1: simplicial v of degree d ⇒ tw(G) = max(d, tw(G−v)).
- **Hypercube** tw(Q_d) = Θ(2^d/√d): Chandran–Kavitha, Discrete Math. 306(3) (2006) (secondary). **tw ≤ 3 minors**: Arnborg–Proskurowski–Corneil, Discrete Math. 80 (1990); Satyanarayana–Tung, Networks 20 (1990).
- **Lattice** tw([L]^d) = Θ(L^{d−1}): Hickingbotham–Wood, JGT 109(2) (2023) Cor. 26. Genus: tw ≤ 2√((2g+3)n) − 1 (Dujmović–Morin–Wood 2017). Random cubic Θ(n) (expanders: Grohe–Marx 2009; pw ≤ n/6+o(n) Fomin–Høie 2006).

## Deep dives (to use in M4–M5)
- **Menger** Fund. Math. 10 (1927); **Ford–Fulkerson** Canad. J. Math. 8 (1956) (also Elias–Feinstein–Shannon 1956).
- **Balanced separator from tw**: Robertson–Seymour GM II, J. Algorithms 7 (1986): |X| ≤ tw+1, every component of G−X has ≤ (n−|X|)/2 vertices (secondary).
- **Dvořák–Norin**, JCTB 137:137–144 (2019): tw ≤ 15·sn(G); balanced = both sides ≤ 2n/3, for every subgraph.
- **Lipton–Tarjan**, SIAM J. Appl. Math. 36(2) (1979): |A|,|B| ≤ 2n/3, |C| ≤ 2√2·√n, O(n) time. tw = O(√n) is a later consequence (Fomin–Thilikos: tw ≤ 3.182√n, secondary).
- **George**, SIAM J. Numer. Anal. 10(2) (1973): O(N^{3/2}) operations, O(N log N) storage on the n×n mesh.
- **Brandes**, J. Math. Sociol. 25(2) (2001): O(nm) unweighted. **Girvan–Newman**, PNAS 99(12) (2002): worst case O(m²n).
- **Modularity** in the A_ij form: Clauset–Newman–Moore, PRE 70:066111 (2004) Eq. 4 (Newman–Girvan PRE 69:026113 (2004) defines Q = Tr e − ‖e²‖).
- **Bodlaender**, SIAM J. Comput. 25(6) (1996): linear time for fixed k; "2^{O(k³)} n" as stated by Korhonen–Lokshtanov 2023.
- **Exact treewidth**: Fomin–Kratsch–Todinca–Villanger, SIAM J. Comput. 38(3) (2008): O(1.8899^n). Fomin–Villanger, Combinatorica 32(3) (2012): O(1.7549^n) (exp. space), O(2.6151^n) poly space. **O(1.7347^n)**: Fomin–Todinca–Villanger, SIAM J. Comput. 44(1) (2015). PROPOSAL's "Fomin–Villanger 2012, 1.7347" is wrong.
- **Bouchitté–Todinca**, SIAM J. Comput. 31(1) (2001): potential maximal cliques characterize minimal triangulations' maximal cliques; tw/fill-in polynomial where PMCs are polynomially listable. "Minimal separators of a minimal triangulation are minimal separators of G": Parra–Scheffler, DAM 79 (1997).
- **Feige–Hajiaghayi–Lee**, SIAM J. Comput. 38(2) (2008): width O(k√log k), i.e. O(√log tw) approximation.
- **FlowCutter**: Hamann–Strasser, ACM JEA 23 (2018) (ALENEX 2016). Report the smaller side's min cut, absorb that side, pierce one cut node avoiding the other side's reachable set, tie-break max dist(p,t) − dist(s,p); cuts in increasing size and improving balance; q = 20 random s–t pairs; O(c·m).
- **PACE 2017 heuristic track** (Dell et al., IPEC 2017, LIPIcs 89:30): 1 Tamaki et al. (Meiji, TCS-Meiji/PACE2017-TrackA), 2 Strasser FlowCutter, 3 htd, 4 Jdrasil, 5 Jégou–Kanso–Terrioux (td-mrs/minfill_mrs, minfillbg_mrs), 6 Larisch–Salfelder.
- **Terrioux et al.**: minfill_mrs = one MinFill pass, ties by "neighbor future degree"; minfillbg_mrs = H-TD-WT framework (ICTAI 2015). PACE-specific changes undocumented.
- **Tamaki heuristic**: greedy start, safe separators, then local improvement around the largest bag (exact tw / exact pw / greedy on that subgraph). `./tw-heuristic -s SEED < g.gr > g.td`, prints `c width = …` on improvements, best .td on SIGTERM.
- **Jdrasil heuristic**: ≤ 30 greedy-degree runs, reduction, stochastic greedy permutations, local search; `./tw-heuristic -s 42 < g.gr`, best on SIGTERM.
- **PACE formats**: .gr `p tw n m` + 1-based edges; .td `s td N maxbag n`, `b i v…`, tree edges.
