"""
knowledge_graph.py
===================
Builds and visualises a Knowledge Graph (KG) from the entities extracted
(entity_extractor.py) out of the Top-N re-ranked search results.

Graph schema
------------
Nodes  : documents (doc_id) and entities (surface form), colour-coded by
         node type / entity category.
Edges  : an edge connects a document node to an entity node if that entity
         appears in the document. Edge weight = number of co-occurrences
         (always 1 here, but kept generic for future extension, e.g.
         co-occurrence across multiple documents strengthens an
         entity-entity edge).

Additionally, two entities that co-occur within the same document are
linked directly, letting the graph reveal relationships such as
"Remdesivir" <-> "SARS-CoV-2" without needing the document node in between.
"""

import logging
import os
import sys
from itertools import combinations

import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colour palette used when rendering the graph (matplotlib colour names)
_NODE_COLOR_MAP = {
    "DOCUMENT": "#4C72B0",
    "VIRUS": "#C44E52",
    "DRUG": "#55A868",
    "VACCINE": "#8172B2",
    "PROTEIN": "#CCB974",
    "ORGANIZATION": "#64B5CD",
    "LOCATION": "#DD8452",
    "PERSON": "#937860",
    "GROUP": "#DA8BC3",
}
_DEFAULT_NODE_COLOR = "#999999"


def build_knowledge_graph(documents: list, doc_entities: dict) -> nx.Graph:
    """
    Build a NetworkX graph linking documents to their extracted entities,
    plus entity-entity co-occurrence edges within the same document.

    Parameters
    ----------
    documents : list[dict]
        Each dict needs at least 'doc_id' and 'title' keys (title used as
        a short, human-readable node label).
    doc_entities : dict
        Mapping doc_id -> list of {'text': ..., 'label': ...} entity dicts,
        as produced by entity_extractor.extract_entities_from_documents.

    Returns
    -------
    networkx.Graph
        Graph with 'type' and 'label' node attributes.

    Raises
    ------
    ValueError
        If no entities were found across all documents.
    """
    graph = nx.Graph()
    title_lookup = {doc["doc_id"]: doc.get("title", doc["doc_id"]) for doc in documents}

    total_entities = 0
    for doc_id, entities in doc_entities.items():
        if not entities:
            continue

        doc_node = f"DOC::{doc_id}"
        short_title = (title_lookup.get(doc_id, doc_id) or doc_id)[:60]
        graph.add_node(doc_node, type="DOCUMENT", label=short_title)

        entity_nodes_in_doc = []
        for entity in entities:
            entity_node = f"{entity['label']}::{entity['text'].lower()}"
            graph.add_node(entity_node, type=entity["label"], label=entity["text"])
            graph.add_edge(doc_node, entity_node, weight=1, relation="mentions")
            entity_nodes_in_doc.append(entity_node)
            total_entities += 1

        # Link entities that co-occur in the same document to reveal
        # potential relationships (e.g. drug <-> virus, vaccine <-> org).
        for node_a, node_b in combinations(set(entity_nodes_in_doc), 2):
            if graph.has_edge(node_a, node_b):
                graph[node_a][node_b]["weight"] += 1
            else:
                graph.add_edge(node_a, node_b, weight=1, relation="co-occurs")

    if total_entities == 0:
        raise ValueError("No entities were found; cannot build a knowledge graph.")

    logger.info(
        "Knowledge graph built: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def visualize_knowledge_graph(graph: nx.Graph, save_path: str = None, figsize=(14, 10)):
    """
    Render the knowledge graph to a Matplotlib figure using a spring
    layout, colour-coded by node type.

    Parameters
    ----------
    graph : networkx.Graph
    save_path : str, optional
        If given, saves the figure as a PNG to this path.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the graph has no nodes.
    """
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty knowledge graph.")

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph, k=0.6, seed=42, weight="weight")

    node_colors = [
        _NODE_COLOR_MAP.get(graph.nodes[n].get("type"), _DEFAULT_NODE_COLOR)
        for n in graph.nodes
    ]
    node_sizes = [
        900 if graph.nodes[n].get("type") == "DOCUMENT" else 500 for n in graph.nodes
    ]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)

    labels = {n: graph.nodes[n].get("label", n)[:20] for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

    # Build a legend showing the colour used per entity type
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
        for label, color in _NODE_COLOR_MAP.items()
        if any(graph.nodes[n].get("type") == label for n in graph.nodes)
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)

    ax.set_title("COVID-19 Research Knowledge Graph (Documents & Extracted Entities)")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Knowledge graph image saved to %s", save_path)

    return fig


def get_graph_statistics(graph: nx.Graph) -> dict:
    """
    Compute basic descriptive statistics about the knowledge graph.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    dict
        Keys: num_nodes, num_edges, num_documents, num_entities,
        entity_type_counts, most_connected_entities (top 5 by degree).
    """
    entity_type_counts = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "UNKNOWN")
        if node_type != "DOCUMENT":
            entity_type_counts[node_type] = entity_type_counts.get(node_type, 0) + 1

    degrees = dict(graph.degree())
    entity_nodes = [n for n in graph.nodes if graph.nodes[n].get("type") != "DOCUMENT"]
    top_entities = sorted(
        ((n, degrees[n]) for n in entity_nodes), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_documents": sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "DOCUMENT"),
        "num_entities": len(entity_nodes),
        "entity_type_counts": entity_type_counts,
        "most_connected_entities": [
            {"entity": graph.nodes[n]["label"], "degree": deg} for n, deg in top_entities
        ],
    }


if __name__ == "__main__":
    demo_docs = [{"doc_id": "1", "title": "Remdesivir efficacy against SARS-CoV-2"}]
    demo_entities = {
        "1": [
            {"text": "Remdesivir", "label": "DRUG"},
            {"text": "SARS-CoV-2", "label": "VIRUS"},
            {"text": "WHO", "label": "ORGANIZATION"},
        ]
    }
    g = build_knowledge_graph(demo_docs, demo_entities)
    print(get_graph_statistics(g))
