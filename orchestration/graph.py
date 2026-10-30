"""LangGraph StateGraph Integration with Automatic Offline Fallback.

Attempts to import and instantiate `langgraph.graph.StateGraph` and `END` directly from the
`langgraph` package (`LANGGRAPH_AVAILABLE = True`). When running in an offline environment
where `langgraph` is not installed, transparently falls back to `FallbackStateGraph` with
identical node, edge, conditional-edge, and bounded-step semantics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

try:
    from langgraph.graph import StateGraph as _LangGraphStateGraph, END as _LANGGRAPH_END  # type: ignore
    LANGGRAPH_AVAILABLE = True
    END = _LANGGRAPH_END
except ImportError:
    _LangGraphStateGraph = None
    LANGGRAPH_AVAILABLE = False
    END = "__END__"


class CompiledFallbackStateGraph:
    """Executes a compiled multi-agent state graph when `langgraph` is not installed."""

    def __init__(
        self,
        entry_point: str,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]],
        max_steps: int = 15,
    ) -> None:
        self.entry_point = entry_point
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.max_steps = max_steps

    def invoke(self, initial_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        limit = (config or {}).get("recursion_limit", self.max_steps)
        state = dict(initial_state)
        current_node = self.entry_point
        steps = 0

        while current_node != END:
            if steps >= limit:
                raise RuntimeError(
                    f"Bounded execution limit ({limit} steps) exceeded in StateGraph."
                )
            if current_node not in self.nodes:
                raise KeyError(f"Unknown graph node: {current_node}")

            node_fn = self.nodes[current_node]
            updates = node_fn(state)
            if isinstance(updates, dict):
                state.update(updates)

            steps += 1

            if current_node in self.conditional_edges:
                router_fn, path_map = self.conditional_edges[current_node]
                route_key = router_fn(state)
                current_node = path_map.get(route_key, route_key)
            elif current_node in self.edges:
                current_node = self.edges[current_node]
            else:
                current_node = END

        return state


class FallbackStateGraph:
    """Fallback StateGraph builder matching `langgraph.graph.StateGraph`."""

    def __init__(self, state_schema: Any = dict) -> None:
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Any] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, action: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.nodes[name] = action

    def add_edge(self, start_key: str, end_key: str) -> None:
        self.edges[start_key] = end_key

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[[Dict[str, Any]], str],
        path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.conditional_edges[source] = (path, path_map or {})

    def set_entry_point(self, key: str) -> None:
        self.entry_point = key

    def compile(self) -> CompiledFallbackStateGraph:
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling StateGraph.")
        return CompiledFallbackStateGraph(
            entry_point=self.entry_point,
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
        )


def create_state_graph(state_schema: Any, graph_cls_override: Any = None) -> Tuple[Any, str]:
    """Returns `(graph_builder, backend_name)` using genuine `langgraph.graph.StateGraph` when available."""
    if graph_cls_override is not None:
        return graph_cls_override(state_schema), "langgraph.StateGraph"
    if LANGGRAPH_AVAILABLE and _LangGraphStateGraph is not None:
        return _LangGraphStateGraph(state_schema), "langgraph.StateGraph"
    return FallbackStateGraph(state_schema), "FallbackStateGraph"


# Maintain backward-compatible alias
StateGraph = _LangGraphStateGraph if (LANGGRAPH_AVAILABLE and _LangGraphStateGraph is not None) else FallbackStateGraph
