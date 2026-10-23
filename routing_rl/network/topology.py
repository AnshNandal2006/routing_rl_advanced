"""Topology builders — diamond, mesh, geometric, scale-free, hierarchical, and custom."""
from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional
import random
import networkx as nx
from .router import Router
from .link import Link

TOPOLOGY_CATALOG = {
    "geometric": "Random connected geometric graph (10–30 nodes) with physical (x, y) coords",
    "diamond": "Classic two-path diamond (S0–A/B–D0) with asymmetric capacities",
    "mesh": "Fully connected core mesh with edge sources/destinations",
    "scale_free": "Barabási–Albert scale-free graph (realistic Internet-like)",
    "hierarchical": "Three-tier hierarchy (core / aggregation / access)",
    "grid": "3×3 grid",
    "ring": "6-node bidirectional ring",
}

_GEOMETRIC_ALIASES = {"geometric", "random_geometric", "random"}


def _mk_routers(
    ids: List[str],
    queue_size: int,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Router]:
    positions = positions or {}
    return {
        rid: Router(rid, max_queue_size=queue_size, pos=positions.get(rid))
        for rid in ids
    }


def _mk_links(
    spec: List[Tuple[str, str, float]],
    prop_delay: float = 0.005,
) -> Dict[Tuple[str, str], Link]:
    links: Dict[Tuple[str, str], Link] = {}
    for src, dst, cap in spec:
        links[(src, dst)] = Link(src, dst, capacity_pps=cap, propagation_delay=prop_delay)
    return links


def _wire_neighbors(routers: Dict[str, Router], links: Dict[Tuple[str, str], Link]) -> None:
    for (src, dst) in links:
        if dst not in routers[src].neighbors:
            routers[src].neighbors.append(dst)


def _ensure_connected(G: nx.Graph, pos: Dict) -> nx.Graph:
    """Add nearest unused geometric edges until the unit-disk graph is connected."""
    if nx.is_connected(G):
        return G
    nodes = list(G.nodes)
    candidates = []
    for i, u in enumerate(nodes):
        pu = pos[u]
        for v in nodes[i + 1 :]:
            if G.has_edge(u, v):
                continue
            pv = pos[v]
            dist = (pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2
            candidates.append((dist, u, v))
    candidates.sort()
    for _, u, v in candidates:
        G.add_edge(u, v)
        if nx.is_connected(G):
            break
    return G


def _farthest_node_pair(G: nx.Graph) -> Tuple[Any, Any]:
    """Return the two nodes with the longest shortest-path distance (hops)."""
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    best_d = -1
    best: Tuple[Any, Any] = (None, None)
    for u in sorted(G.nodes()):
        for v in sorted(G.nodes()):
            if u >= v:
                continue
            d = lengths[u][v]
            if d > best_d:
                best_d = d
                best = (u, v)
    return best


def _build_geometric_topology(
    queue_size: int,
    prop_delay: float,
    **kwargs,
) -> Tuple[Dict[str, Router], Dict[Tuple[str, str], Link], Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """Random connected geometric graph with physical coordinates and S/D endpoints."""
    cap_lo = float(kwargs.get("cap_min", 20.0))
    cap_hi = float(kwargs.get("cap_max", 100.0))
    n_min = int(kwargs.get("n_nodes_min", 10))
    n_max = int(kwargs.get("n_nodes_max", 30))
    if n_min > n_max:
        n_min, n_max = n_max, n_min

    seed = kwargs.get("seed", None)
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    seed = int(seed)
    rng = random.Random(seed)

    if kwargs.get("n_nodes") is not None:
        n = int(kwargs["n_nodes"])
    else:
        n = rng.randint(n_min, n_max)
    n = max(2, n)

    radius = float(kwargs.get("radius", 0.45))
    G_und = None
    pos: Dict[Any, Any] = {}
    for _ in range(40):
        g_seed = rng.randint(0, 2**31 - 1)
        G_und = nx.random_geometric_graph(n, radius, seed=g_seed)
        pos = nx.get_node_attributes(G_und, "pos")
        if nx.is_connected(G_und):
            break
        radius = min(0.99, radius + 0.05)
    assert G_und is not None
    G_und = _ensure_connected(G_und, pos)
    pos = nx.get_node_attributes(G_und, "pos")
    for node, xy in list(pos.items()):
        G_und.nodes[node]["pos"] = (float(xy[0]), float(xy[1]))
        pos[node] = (float(xy[0]), float(xy[1]))

    source, dest = _farthest_node_pair(G_und)
    mapping: Dict[Any, str] = {source: "S", dest: "D"}
    r_idx = 0
    for node in sorted(G_und.nodes()):
        if node in mapping:
            continue
        mapping[node] = f"R{r_idx}"
        r_idx += 1

    G_und = nx.relabel_nodes(G_und, mapping)
    positions = {
        mapping[node]: (float(xy[0]), float(xy[1])) for node, xy in pos.items()
    }

    router_ids = list(G_und.nodes)
    routers = _mk_routers(router_ids, queue_size, positions)
    link_spec: List[Tuple[str, str, float]] = []
    for u, v in sorted(G_und.edges()):
        fwd = rng.uniform(cap_lo, cap_hi)
        rev = rng.uniform(cap_lo, cap_hi)
        link_spec += [(u, v, fwd), (v, u, rev)]
    links = _mk_links(link_spec, prop_delay)
    meta = {
        "n_nodes": n,
        "topology_seed": seed,
        "topology": "geometric",
    }
    return routers, links, positions, meta


def traffic_defaults(
    name: str,
    routers: Dict[str, Router],
) -> Tuple[List[str], List[str], List[str]]:
    """Sources, destinations, and a representative failure link for a topology."""
    name = name.lower().strip()
    if name == "diamond":
        return ["S0"], ["D0"], ["S0", "B"]
    if name == "mesh":
        return ["S0", "S1"], ["D0", "D1"], ["C0", "C1"]
    if name == "hierarchical":
        return ["S0", "S1"], ["D0", "D1"], ["Core0", "Agg1"]
    if name in _GEOMETRIC_ALIASES or ("S" in routers and "D" in routers):
        neigh = routers["S"].neighbors if "S" in routers else []
        fail = ["S", neigh[0]] if neigh else ["S", "D"]
        return ["S"], ["D"], fail
    ids = sorted(routers.keys())
    fail = [ids[0], ids[min(1, len(ids) - 1)]]
    return [ids[0]], [ids[-1]], fail


def build_topology(
    name: str,
    queue_size: int = 150,
    capacity: float = 80.0,
    prop_delay: float = 0.005,
    **kwargs,
) -> Tuple[Dict[str, Router], Dict[Tuple[str, str], Link], nx.DiGraph]:
    name = name.lower().strip()
    positions: Dict[str, Tuple[float, float]] = {}
    meta: Dict[str, Any] = {"topology": name}

    if name in _GEOMETRIC_ALIASES:
        routers, links, positions, meta = _build_geometric_topology(
            queue_size, prop_delay, **kwargs
        )

    elif name == "diamond":
        # Asymmetric two-path: upper path is bottleneck, lower is fat
        router_ids = ["S0", "A", "B", "D0"]
        routers = _mk_routers(router_ids, queue_size)
        link_spec = [
            ("S0", "A", 120.0), ("A", "S0", 120.0),
            ("A", "D0", 15.0),  ("D0", "A", 15.0),   # bottleneck
            ("S0", "B", 120.0), ("B", "S0", 120.0),
            ("B", "D0", 120.0), ("D0", "B", 120.0),  # fat path
        ]
        links = _mk_links(link_spec, prop_delay)

    elif name == "mesh":
        # 4 core routers fully meshed + 2 sources + 2 sinks
        core = ["C0", "C1", "C2", "C3"]
        sources = ["S0", "S1"]
        dests = ["D0", "D1"]
        router_ids = sources + core + dests
        routers = _mk_routers(router_ids, queue_size)
        link_spec = []
        # Core mesh
        for i, u in enumerate(core):
            for v in core[i + 1 :]:
                link_spec += [(u, v, capacity), (v, u, capacity)]
        # Attach sources
        for s in sources:
            for c in core[:2]:
                link_spec += [(s, c, capacity * 1.5), (c, s, capacity * 1.5)]
        # Attach destinations
        for d in dests:
            for c in core[2:]:
                link_spec += [(c, d, capacity * 1.5), (d, c, capacity * 1.5)]
        links = _mk_links(link_spec, prop_delay)

    elif name == "scale_free":
        n = kwargs.get("n_nodes", 12)
        m = kwargs.get("m_attach", 2)
        G_und = nx.barabasi_albert_graph(n, m, seed=kwargs.get("seed", 42))
        mapping = {i: f"R{i}" for i in range(n)}
        G_und = nx.relabel_nodes(G_und, mapping)
        router_ids = list(G_und.nodes)
        routers = _mk_routers(router_ids, queue_size)
        link_spec = []
        for u, v in G_und.edges:
            # Heterogeneous capacities based on degree (hubs faster)
            deg = G_und.degree(u) + G_und.degree(v)
            cap = capacity * (0.6 + 0.15 * deg)
            link_spec += [(u, v, cap), (v, u, cap)]
        links = _mk_links(link_spec, prop_delay)
        # Designate low-degree nodes as sources / high-degree as possible sinks
        degrees = sorted(G_und.degree, key=lambda x: x[1])
        # We keep the topology generic; traffic config selects S/D

    elif name == "hierarchical":
        # Core (2) – Aggregation (4) – Access (4 sources + 2 dests)
        core = ["Core0", "Core1"]
        agg = ["Agg0", "Agg1", "Agg2", "Agg3"]
        access_src = ["S0", "S1", "S2", "S3"]
        access_dst = ["D0", "D1"]
        router_ids = core + agg + access_src + access_dst
        routers = _mk_routers(router_ids, queue_size)
        link_spec = []
        # Core interconnect
        link_spec += [("Core0", "Core1", capacity * 2), ("Core1", "Core0", capacity * 2)]
        # Core ↔ Agg
        for i, a in enumerate(agg):
            c = core[i % 2]
            link_spec += [(c, a, capacity * 1.5), (a, c, capacity * 1.5)]
        # Agg ↔ Access sources
        for i, s in enumerate(access_src):
            a = agg[i]
            link_spec += [(s, a, capacity), (a, s, capacity)]
        # Agg ↔ Destinations
        for i, d in enumerate(access_dst):
            a = agg[i + 2]
            link_spec += [(a, d, capacity * 1.2), (d, a, capacity * 1.2)]
        links = _mk_links(link_spec, prop_delay)

    elif name == "grid":
        G = nx.grid_2d_graph(3, 3)
        G = nx.relabel_nodes(G, lambda n: f"R{n[0]}{n[1]}")
        router_ids = list(G.nodes)
        routers = _mk_routers(router_ids, queue_size)
        link_spec = []
        for u, v in G.edges:
            link_spec += [(u, v, capacity), (v, u, capacity)]
        links = _mk_links(link_spec, prop_delay)

    elif name == "ring":
        n = 6
        router_ids = [f"R{i}" for i in range(n)]
        routers = _mk_routers(router_ids, queue_size)
        link_spec = []
        for i in range(n):
            a, b = router_ids[i], router_ids[(i + 1) % n]
            link_spec += [(a, b, capacity), (b, a, capacity)]
        links = _mk_links(link_spec, prop_delay)

    else:
        raise ValueError(f"Unknown topology '{name}'. Choose from: {list(TOPOLOGY_CATALOG)}")

    _wire_neighbors(routers, links)

    G = nx.DiGraph()
    G.graph.update(meta)
    G.graph["n_nodes"] = len(routers)
    for rid, router in routers.items():
        node_pos = positions.get(rid) or getattr(router, "pos", None)
        if node_pos is not None:
            G.add_node(rid, pos=(float(node_pos[0]), float(node_pos[1])))
        else:
            G.add_node(rid)
    for (src, dst), link in links.items():
        if link.is_up:
            G.add_edge(src, dst, weight=1, capacity=link.capacity_pps)
    return routers, links, G


def get_layout_positions(name: str, routers: Dict[str, Router]) -> Dict[str, Tuple[float, float]]:
    """Hand-crafted, stored geometric, or spring layouts for visualization."""
    stored = {}
    for rid, router in routers.items():
        if getattr(router, "pos", None) is not None:
            stored[rid] = (float(router.pos[0]), float(router.pos[1]))
    if len(stored) == len(routers) and stored:
        return stored

    name = name.lower()
    if name == "diamond":
        return {"S0": (0, 1), "A": (1, 2), "B": (1, 0), "D0": (2, 1)}
    if name == "mesh":
        return {
            "S0": (0, 1.5), "S1": (0, 0.5),
            "C0": (1, 2), "C1": (1, 1), "C2": (2, 2), "C3": (2, 1),
            "D0": (3, 1.5), "D1": (3, 0.5),
        }
    if name == "hierarchical":
        return {
            "Core0": (1.5, 3), "Core1": (2.5, 3),
            "Agg0": (0.5, 2), "Agg1": (1.5, 2), "Agg2": (2.5, 2), "Agg3": (3.5, 2),
            "S0": (0.5, 1), "S1": (1.5, 1), "S2": (2.5, 1), "S3": (3.5, 1),
            "D0": (1.5, 0), "D1": (2.5, 0),
        }
    # Fallback: spring layout
    G = nx.Graph()
    G.add_nodes_from(routers.keys())
    for r in routers.values():
        for n in r.neighbors:
            G.add_edge(r.id, n)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
