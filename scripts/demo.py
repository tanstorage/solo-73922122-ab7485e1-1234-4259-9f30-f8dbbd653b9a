#!/usr/bin/env python3
"""Demo data + scenario runner for the corpus contamination checker.

Scenarios
---------
seed        load the demo corpus (direct near-duplicate pair, a *bridge*
            article connecting a train and a test article, clean articles)
bridge      delete the bridge article -> its contamination group splits;
            re-add it -> the group forms again
concurrent  submit batches while a background comparison is running:
            the API keeps accepting writes, answers 202 until verification
            covers the latest corpus, and stale computations are discarded
restart     submit a batch, restart the API processes, and watch
            verification complete automatically after the restart

Runs against any base URL (host port or in-compose service name) and uses
only the standard library, so it also works inside the slim API image.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.similarity import jaccard, shingles  # noqa: E402

DEFAULT_BASE = os.environ.get("BASE_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# demo corpus generation (deterministic)
# ---------------------------------------------------------------------------

VOCAB = (
    "finance market stocks bonds trade economy growth inflation rate bank "
    "government policy election senate vote law court judge jury appeal "
    "climate energy carbon solar wind oil gas emission temperature "
    "health vaccine hospital doctor patient virus treatment study "
    "technology software data network chip silicon startup venture "
    "sports league match season coach player victory defeat score "
    "editor reporter journal column headline daily press media"
).split()

SEED = 20261114


def pseudo_article(rng, target_len=200):
    words = []
    total = 0
    while total < target_len:
        word = rng.choice(VOCAB)
        words.append(word)
        total += len(word) + 1
    return " ".join(words)


def alter(text, positions, rng):
    """Change single characters at *positions* (kept >= 5 apart and away
    from both ends, so each change touches exactly 5 shingles)."""
    chars = list(text)
    for pos in positions:
        choices = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != chars[pos]]
        chars[pos] = rng.choice(choices)
    return "".join(chars)


def build_demo_articles():
    rng = random.Random(SEED)

    # Bridge cluster: the bridge article itself is near-duplicate of both a
    # train and a test article which are NOT near-duplicates of each other.
    # 3 single-char changes per endpoint: endpoint~bridge ~= 0.86 (>= 0.8),
    # endpoint~endpoint ~= 0.74 (< 0.8) -- only the bridge links them.
    bridge_base = pseudo_article(rng)
    train_bridge_a = alter(bridge_base, [30, 60, 90], rng)
    test_bridge_b = alter(bridge_base, [130, 160, 190], rng)

    # Direct contamination pair (1 char difference).
    direct_base = pseudo_article(rng)
    test_direct = alter(direct_base, [100], rng)

    # Unrelated clean articles.
    train_clean = pseudo_article(rng)
    test_clean = pseudo_article(rng)

    return [
        {"id": "train-direct-1", "side": "train", "text": direct_base},
        {"id": "test-direct-1", "side": "test", "text": test_direct},
        {"id": "train-bridge-a", "side": "train", "text": train_bridge_a},
        {"id": "test-bridge", "side": "test", "text": bridge_base},
        {"id": "test-bridge-b", "side": "test", "text": test_bridge_b},
        {"id": "train-clean-1", "side": "train", "text": train_clean},
        {"id": "test-clean-1", "side": "test", "text": test_clean},
    ]


EXPECTED_EDGES = {
    tuple(sorted(pair))
    for pair in [
        ("train-direct-1", "test-direct-1"),
        ("train-bridge-a", "test-bridge"),
        ("test-bridge", "test-bridge-b"),
    ]
}


def verify_demo_articles(articles):
    """Fail loudly if the generated data does not produce the intended graph."""
    by_id = {a["id"]: a["text"] for a in articles}
    ids = sorted(by_id)
    actual = set()
    sims = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            sim = jaccard(shingles(by_id[a]), shingles(by_id[b]))
            sims[(a, b)] = sim
            if sim >= 0.8:
                actual.add((a, b))
    if actual != EXPECTED_EDGES:
        table = "\n".join(
            f"  {a} ~ {b}: {s:.3f}" for (a, b), s in sorted(sims.items())
        )
        raise SystemExit(
            "demo data self-check failed\n"
            f"expected edges: {sorted(EXPECTED_EDGES)}\nactual edges: {sorted(actual)}\n"
            f"all similarities:\n{table}"
        )
    return sims


# ---------------------------------------------------------------------------
# tiny HTTP client
# ---------------------------------------------------------------------------


def http(base, method, path, payload=None):
    req = urllib.request.Request(base + path, method=method)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data, timeout=30) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw.decode(errors="replace")}
        return exc.code, body


def wait_ready(base, timeout=180):
    """Block until published results cover the whole committed corpus."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, body = http(base, "GET", "/status")
            if code == 200 and body["up_to_date"]:
                return body
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.3)
    raise SystemExit("timed out waiting for verification to catch up")


def get_contamination(base):
    code, body = http(base, "GET", "/contamination")
    return code, body


def show_groups(report):
    if not report["contaminated"]:
        print("  -> no cross-set contamination")
        return
    for group in report["groups"]:
        print(
            f"  group {group['group_id']}: train={group['train_members']} "
            f"test={group['test_members']}"
        )
        for edge in group["edges"]:
            print(f"    edge {edge['a']} ~ {edge['b']}  sim={edge['similarity']:.3f}")


def expect(condition, message):
    if not condition:
        raise SystemExit(f"SCENARIO FAILED: {message}")
    print(f"  ok: {message}")


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def reset_to_seed(base):
    """Atomically replace the whole corpus with the demo seed articles."""
    seed = build_demo_articles()
    sims = verify_demo_articles(seed)
    code, body = http(base, "GET", "/articles")
    existing = [a["id"] for a in body.get("articles", [])]
    code, body = http(
        base,
        "POST",
        "/articles/batch",
        {"upserts": seed, "deletes": [i for i in existing if i not in {a["id"] for a in seed}]},
    )
    expect(code == 200, f"seed batch committed (generation {body['generation']})")
    wait_ready(base)
    print("  shingle similarities inside the seed corpus:")
    for (a, b), s in sorted(sims.items()):
        marker = " <-- edge" if s >= 0.8 else ""
        print(f"    {a} ~ {b}: {s:.3f}{marker}")
    return seed


def scenario_seed(base):
    print("== seed: loading demo corpus ==")
    reset_to_seed(base)
    code, report = get_contamination(base)
    expect(code == 200, "verification covers the seeded corpus")
    print("  contamination groups after seeding:")
    show_groups(report)
    expect(report["contamination_group_count"] == 2, "two contamination groups found")
    print("seed done.\n")


def scenario_bridge(base):
    print("== bridge: deleting the bridge article splits its group ==")
    seed = reset_to_seed(base)
    bridge = next(a for a in seed if a["id"] == "test-bridge")

    code, report = get_contamination(base)
    expect(code == 200, "report available before deletion")
    print("  before deletion:")
    show_groups(report)

    code, body = http(base, "POST", "/articles/batch", {"deletes": ["test-bridge"]})
    expect(code == 200 and body["deleted"] == 1, "bridge article deleted")
    wait_ready(base)
    code, report = get_contamination(base)
    print("  after deleting test-bridge:")
    show_groups(report)
    grouped = {m for g in report["groups"] for m in g["train_members"] + g["test_members"]}
    expect(
        "train-bridge-a" not in grouped and "test-bridge-b" not in grouped,
        "bridge cluster split: train-bridge-a and test-bridge-b are ungrouped",
    )
    expect(
        report["contamination_group_count"] == 1,
        "only the direct pair remains contaminated",
    )

    code, body = http(base, "POST", "/articles/batch", {"upserts": [bridge]})
    expect(code == 200, "bridge article re-added")
    wait_ready(base)
    code, report = get_contamination(base)
    print("  after re-adding test-bridge:")
    show_groups(report)
    expect(report["contamination_group_count"] == 2, "bridge group formed again")
    print("bridge scenario done.\n")


def scenario_concurrent(base):
    print("== concurrent: submitting while a comparison is running ==")
    _, status = http(base, "GET", "/status")
    delay = status["config"]["compare_delay_ms"]
    if delay <= 0:
        print("  (COMPARE_DELAY_MS is 0; set it e.g. to 3000 to observe the pending window)")
    reset_to_seed(base)

    rng = random.Random(SEED + 1)
    extra_base = pseudo_article(rng)
    train_extra = {"id": "train-extra-1", "side": "train", "text": extra_base}
    test_extra = {
        "id": "test-extra-1",
        "side": "test",
        "text": alter(extra_base, [100], rng),
    }

    code, body = http(base, "POST", "/articles/batch", {"upserts": [train_extra]})
    gen_first = body["generation"]
    expect(code == 200, f"first batch committed during compare (generation {gen_first})")

    code, report = get_contamination(base)
    expect(code == 202, f"report is 202 pending right after the batch (got {code})")
    print(
        f"  pending state: current={report['current_generation']} "
        f"published={report['published_generation']}"
    )

    # Second batch lands while the worker is still computing the first one.
    code, body = http(base, "POST", "/articles/batch", {"upserts": [test_extra]})
    gen_second = body["generation"]
    expect(code == 200, f"second batch accepted while comparing (generation {gen_second})")

    code, report = get_contamination(base)
    expect(code == 202, "still 202 while the newest generation is unverified")

    final = wait_ready(base)
    expect(
        final["current_generation"] == gen_second
        and final["published_generation"] == gen_second,
        "stale computation discarded; published results match the newest batch",
    )
    code, report = get_contamination(base)
    expect(code == 200 and report["generation"] == gen_second, "report serves the newest generation")
    print("  final groups:")
    show_groups(report)
    grouped_pairs = {
        tuple(sorted((e["a"], e["b"])))
        for g in report["groups"]
        for e in g["edges"]
    }
    expect(
        ("test-extra-1", "train-extra-1") in grouped_pairs,
        "the extra near-duplicate pair submitted mid-compare was detected",
    )
    print("concurrent scenario done.\n")


def scenario_restart(base, restart_cmd):
    print("== restart: verification resumes automatically after a restart ==")
    reset_to_seed(base)
    rng = random.Random(SEED + 2)
    probe_base = pseudo_article(rng)
    upserts = [
        {"id": "train-restart-1", "side": "train", "text": probe_base},
        {"id": "test-restart-1", "side": "test", "text": alter(probe_base, [100], rng)},
    ]
    code, body = http(base, "POST", "/articles/batch", {"upserts": upserts})
    gen = body["generation"]
    expect(code == 200, f"probe batch committed (generation {gen})")

    if restart_cmd:
        print(f"  restarting API processes: {restart_cmd}")
        subprocess.run(restart_cmd, shell=True, check=True)
    else:
        input("  restart the API processes now (e.g. `docker compose restart api1 api2`), then press Enter...")

    final = wait_ready(base, timeout=300)
    expect(
        final["published_generation"] == gen,
        "after restart, verification caught up without resubmitting anything",
    )
    code, report = get_contamination(base)
    expect(code == 200 and report["generation"] == gen, "report consistent after restart")
    grouped_pairs = {
        tuple(sorted((e["a"], e["b"])))
        for g in report["groups"]
        for e in g["edges"]
    }
    expect(
        ("test-restart-1", "train-restart-1") in grouped_pairs,
        "probe pair detected after restart",
    )
    print("restart scenario done.\n")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario",
        choices=["seed", "bridge", "concurrent", "restart", "all", "wait", "status"],
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument(
        "--restart-cmd",
        default=os.environ.get("RESTART_CMD", ""),
        help="shell command used by the restart scenario",
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    if args.scenario == "wait":
        wait_ready(base)
        print("verification is up to date")
    elif args.scenario == "status":
        code, body = http(base, "GET", "/status")
        print(json.dumps(body, indent=2))
    elif args.scenario == "seed":
        scenario_seed(base)
    elif args.scenario == "bridge":
        scenario_bridge(base)
    elif args.scenario == "concurrent":
        scenario_concurrent(base)
    elif args.scenario == "restart":
        scenario_restart(base, args.restart_cmd)
    elif args.scenario == "all":
        scenario_seed(base)
        scenario_bridge(base)
        scenario_concurrent(base)
        scenario_restart(base, args.restart_cmd)


if __name__ == "__main__":
    main()
