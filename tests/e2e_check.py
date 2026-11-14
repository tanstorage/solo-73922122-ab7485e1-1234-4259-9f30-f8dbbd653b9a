#!/usr/bin/env python3
"""End-to-end requirement checks against two API processes sharing one DB.

Spawns two uvicorn processes on the same SQLite file (like the two compose
services), then verifies: input validation, batch atomicity, the 300-article
limit, 202 pending gating, contamination groups/edges, bridge-article group
splitting, modify-while-comparing (stale results discarded), cross-process
consistency, crash/restart recovery and no stale/mixed published results.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.demo import build_demo_articles  # noqa: E402

COMPARE_DELAY_MS = 1500
FAILURES = []


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class Server:
    def __init__(self, port, db_path, workdir):
        self.port = port
        self.db_path = db_path
        # The app hard-exits when this file appears (chaos hook); used to
        # simulate crashes and to stop servers without signals.
        self.exit_file = os.path.join(workdir, f"exit-{port}")
        self.proc = None
        self.log = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        env = dict(
            os.environ,
            DB_PATH=self.db_path,
            INSTANCE_ID=f"local-{self.port}",
            COMPARE_DELAY_MS=str(COMPARE_DELAY_MS),
            WORKER_INTERVAL_MS="300",
            EXIT_WHEN_FILE_EXISTS=self.exit_file,
        )
        self.log = open(f"/tmp/corpus-e2e-{self.port}.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.base}/status", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.port} died; see {self.log.name}")
            time.sleep(0.2)
        raise RuntimeError(f"server {self.port} not ready in time")

    def kill(self):
        """Simulate a hard crash (the process os._exit()s on its own)."""
        if self.proc and self.proc.poll() is None:
            open(self.exit_file, "w").close()
            self.proc.wait(timeout=15)
        if os.path.exists(self.exit_file):
            os.unlink(self.exit_file)
        if self.log:
            self.log.close()


def wait_ready(base, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            body = httpx.get(f"{base}/status", timeout=5).json()
            if body["up_to_date"]:
                return body
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{base} did not become up-to-date")


def main():
    workdir = tempfile.mkdtemp(prefix="corpus-e2e-")
    db_path = os.path.join(workdir, "corpus.db")
    s1 = Server(free_port(), db_path, workdir)
    s2 = Server(free_port(), db_path, workdir)
    s1.start()
    s2.start()
    base1, base2 = s1.base, s2.base
    print(f"two API processes up: {base1} {base2} (db={db_path})\n")

    try:
        run_checks(base1, base2, s1, s2)
    finally:
        s1.kill()
        s2.kill()
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


def run_checks(base1, base2, s1, s2):
    seed = build_demo_articles()

    # ---- input validation -------------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcd"}]})
    check("text shorter than 5 chars rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "a" * 2001}]})
    check("text longer than 2000 chars rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "valid", "text": "abcde"}]})
    check("side outside train/test rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcde"},
                                     {"id": "x", "side": "test", "text": "abcde"}]})
    check("duplicate ids inside one batch rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcde"}],
                         "deletes": ["x"]})
    check("same id upserted and deleted rejected (422)", r.status_code == 422)
    r = httpx.get(f"{base1}/articles")
    check("no invalid article leaked into the corpus", r.json()["count"] == 0)

    # ---- batch atomicity ---------------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "good-1", "side": "train", "text": "valid text here"},
                                     {"id": "bad-1", "side": "train", "text": "no"}]})
    check("batch with one invalid article rejected", r.status_code == 422)
    r = httpx.get(f"{base1}/articles")
    check("failed batch applied nothing (atomicity)", r.json()["count"] == 0)

    # boundary lengths accepted
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "min5", "side": "train", "text": "abcde"},
                                     {"id": "max2000", "side": "test", "text": "z" * 2000}]})
    check("5-char and 2000-char texts accepted", r.status_code == 200)

    # ---- 300-article limit --------------------------------------------------
    big = [{"id": f"bulk-{i:03d}", "side": "train" if i % 2 else "test",
            "text": f"bulk article number {i} with enough text"} for i in range(298)]
    r = httpx.post(f"{base1}/articles/batch", json={"upserts": big})
    check("bulk upsert up to 300 articles accepted",
          r.status_code == 200 and r.json()["articles"] == 300)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "one-too-many", "side": "test",
                                      "text": "one article too many"}]})
    check("301st article rejected (409)", r.status_code == 409)
    r = httpx.get(f"{base1}/articles")
    check("corpus still exactly 300 after rejected batch", r.json()["count"] == 300)

    # clear corpus again (single atomic delete batch)
    ids = [a["id"] for a in r.json()["articles"]]
    r = httpx.post(f"{base1}/articles/batch", json={"deletes": ids})
    check("bulk delete accepted", r.status_code == 200 and r.json()["articles"] == 0)
    wait_ready(base1)

    # ---- contamination report across the two processes ---------------------
    r = httpx.post(f"{base1}/articles/batch", json={"upserts": seed})
    check("demo seed committed via api1", r.status_code == 200)
    gen_seed = r.json()["generation"]

    r = httpx.get(f"{base2}/contamination")
    check("api2 answers 202 while verification is pending",
          r.status_code == 202 and r.json()["status"] == "pending")

    wait_ready(base2)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("api2 serves 200 once results cover the corpus",
          r.status_code == 200 and report["generation"] == gen_seed)
    check("two contamination groups found", report["contamination_group_count"] == 2)
    member_sets = {frozenset(g["train_members"] + g["test_members"]) for g in report["groups"]}
    check("direct pair grouped together",
          frozenset({"train-direct-1", "test-direct-1"}) in member_sets)
    check("bridge cluster grouped together",
          frozenset({"train-bridge-a", "test-bridge", "test-bridge-b"}) in member_sets)
    bridge_group = next(g for g in report["groups"] if "test-bridge" in g["test_members"])
    edge_pairs = {tuple(sorted((e["a"], e["b"]))) for e in bridge_group["edges"]}
    check("bridge group exposes direct similarity edges",
          edge_pairs == {("test-bridge", "train-bridge-a"), ("test-bridge", "test-bridge-b")},
          str(edge_pairs))
    check("every edge similarity >= 0.8",
          all(e["similarity"] >= 0.8 for g in report["groups"] for e in g["edges"]))

    # ---- deleting the bridge article splits the group ----------------------
    r = httpx.post(f"{base2}/articles/batch", json={"deletes": ["test-bridge"]})
    check("bridge deletion committed via api2", r.status_code == 200)
    wait_ready(base1)
    r = httpx.get(f"{base1}/contamination")
    report = r.json()
    grouped = {m for g in report["groups"] for m in g["train_members"] + g["test_members"]}
    check("bridge cluster split after deleting the bridge article",
          "train-bridge-a" not in grouped and "test-bridge-b" not in grouped)
    check("direct pair still reported", report["contamination_group_count"] == 1)

    # ---- modify while comparing: stale results must not overwrite ----------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "stale-probe", "side": "train",
                                      "text": "a completely fresh train article about nothing"}]})
    gen_a = r.json()["generation"]
    r = httpx.get(f"{base1}/contamination")
    check("202 right after a new batch (old groups not served with new corpus)",
          r.status_code == 202)
    # second batch lands while the worker computes generation gen_a
    r = httpx.post(f"{base2}/articles/batch",
                   json={"upserts": [{"id": "stale-probe-2", "side": "test",
                                      "text": "a completely fresh train article about nothink"}]})
    gen_b = r.json()["generation"]
    check("second batch accepted while background compare running",
          gen_b == gen_a + 1)
    final = wait_ready(base2)
    check("published generation caught up with the newest batch",
          final["published_generation"] == gen_b)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("report serves exactly the newest generation (no stale/mixed results)",
          report["generation"] == gen_b)
    # the two probe articles differ by 1 char in 43 -> shingle jaccard ~0.74,
    # so they must NOT be linked; the point is the report reflects gen_b state.
    check("corpus content matches the newest batch",
          httpx.get(f"{base2}/articles").json()["count"] == 8)

    # ---- crash + restart recovery ------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "restart-a", "side": "train",
                                      "text": "restart probe article with some unique wording"},
                                     {"id": "restart-b", "side": "test",
                                      "text": "restart probe article with some unique wordinx"}]})
    gen_c = r.json()["generation"]
    s1.kill()  # crash both processes mid-verification
    s2.kill()
    time.sleep(0.5)
    s1.start()
    s2.start()
    final = wait_ready(base1, timeout=90)
    check("after crash+restart verification completes automatically",
          final["published_generation"] == gen_c)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("post-restart report is consistent (single generation)",
          r.status_code == 200 and report["generation"] == gen_c)
    pairs = {tuple(sorted((e["a"], e["b"]))) for g in report["groups"] for e in g["edges"]}
    check("restart probe pair detected after recovery",
          ("restart-a", "restart-b") in pairs)

    # ---- clean verdict only when truly clean --------------------------------
    ids = [a["id"] for a in httpx.get(f"{base1}/articles").json()["articles"]]
    contaminated = [i for i in ids if i not in ("train-clean-1", "test-clean-1")]
    httpx.post(f"{base1}/articles/batch", json={"deletes": contaminated})
    r = httpx.get(f"{base1}/contamination")
    check("202 (not a premature clean verdict) right after deletions",
          r.status_code == 202)
    wait_ready(base1)
    r = httpx.get(f"{base1}/contamination")
    check("clean corpus reported as uncontaminated once verified",
          r.status_code == 200 and r.json()["contaminated"] is False
          and r.json()["groups"] == [])


if __name__ == "__main__":
    main()
