"""Near-duplicate detection core.

Similarity model: the set of all consecutive 5-character substrings
(char 5-grams / shingles) of the raw article text; two articles are linked
by a direct similarity edge when the Jaccard similarity of their shingle
sets reaches 0.8. Groups are connected components over those edges.
"""

SHINGLE_SIZE = 5
SIMILARITY_THRESHOLD = 0.8


def shingles(text: str, k: int = SHINGLE_SIZE) -> frozenset:
    """All consecutive k-character substrings of *text* as a set."""
    if len(text) < k:
        return frozenset()
    return frozenset(text[i : i + k] for i in range(len(text) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def compute_edges(articles):
    """Direct similarity edges for *articles* (iterable of ``(id, text)``).

    Returns a list of ``(id_a, id_b, similarity)`` tuples with
    ``similarity >= SIMILARITY_THRESHOLD``, ordered deterministically.
    """
    items = list(articles)
    shingle_sets = {aid: shingles(text) for aid, text in items}
    edges = []
    for i in range(len(items)):
        id_a = items[i][0]
        set_a = shingle_sets[id_a]
        if not set_a:
            continue
        for j in range(i + 1, len(items)):
            id_b = items[j][0]
            sim = jaccard(set_a, shingle_sets[id_b])
            if sim >= SIMILARITY_THRESHOLD:
                edges.append((id_a, id_b, round(sim, 6)))
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def connected_groups(article_ids, edges):
    """Connected components (size >= 2) of the similarity graph.

    Returns a list of sorted member-id lists, ordered by first member so
    group numbering is deterministic for a given corpus snapshot.
    """
    parent = {aid: aid for aid in article_ids}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, _ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    components = {}
    for aid in article_ids:
        components.setdefault(find(aid), []).append(aid)
    groups = [sorted(m) for m in components.values() if len(m) >= 2]
    groups.sort(key=lambda members: members[0])
    return groups
