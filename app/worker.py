"""Background verification worker.

Every API process runs one worker thread; jobs live in the shared database
and are claimed under a lease, so exactly one process computes a given
generation while the others keep serving requests. Corpus submissions stay
possible while a comparison is running: mutations only take a short write
transaction, and any result computed for a superseded generation is
discarded instead of being published over the newer corpus state.
"""

import logging
import threading
import time

from .db import connect, get_meta
from .similarity import compute_edges, connected_groups

log = logging.getLogger("corpus.worker")


class CompareWorker(threading.Thread):
    def __init__(
        self,
        db_path: str,
        instance_id: str,
        interval_ms: int = 1000,
        compare_delay_ms: int = 0,
        lease_seconds: float = 30.0,
    ):
        super().__init__(name=f"compare-worker-{instance_id}", daemon=True)
        self.db_path = db_path
        self.instance_id = instance_id
        self.interval = max(interval_ms, 100) / 1000.0
        # Artificial compute delay (demo: makes the pending window observable
        # and proves mutations are still accepted while comparing).
        self.compare_delay = max(compare_delay_ms, 0) / 1000.0
        self.lease_seconds = lease_seconds
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info(
            "worker started (instance=%s, interval=%.2fs, compare_delay=%.2fs)",
            self.instance_id,
            self.interval,
            self.compare_delay,
        )
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("worker tick failed")
            self._stop_event.wait(self.interval)

    # ------------------------------------------------------------------
    # job bookkeeping
    # ------------------------------------------------------------------
    def ensure_pending_job(self) -> None:
        """Guarantee a job exists whenever the corpus has unpublished changes.

        Runs on startup (restart recovery) and on every tick. Stale jobs at
        or below the published generation are garbage-collected here.
        """
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            current, published = get_meta(conn)
            conn.execute("DELETE FROM compute_jobs WHERE generation <= ?", (published,))
            if current > published:
                conn.execute(
                    "INSERT OR IGNORE INTO compute_jobs (generation, status)"
                    " VALUES (?, 'pending')",
                    (current,),
                )
            conn.commit()
        finally:
            conn.close()

    def claim_job(self):
        """Atomically claim the newest claimable job; return its generation."""
        now = time.time()
        expiry = now - self.lease_seconds
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT generation FROM compute_jobs
                WHERE status = 'pending' OR (status = 'running' AND claimed_at < ?)
                ORDER BY generation DESC
                LIMIT 1
                """,
                (expiry,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            generation = row["generation"]
            cur = conn.execute(
                """
                UPDATE compute_jobs
                SET status = 'running', claimed_by = ?, claimed_at = ?
                WHERE generation = ? AND (status = 'pending' OR claimed_at < ?)
                """,
                (self.instance_id, now, generation, expiry),
            )
            conn.commit()
            if cur.rowcount == 1:
                log.info("claimed job for generation %d", generation)
                return generation
            return None
        finally:
            conn.close()

    def drop_job(self, generation: int) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute("DELETE FROM compute_jobs WHERE generation = ?", (generation,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # main loop body
    # ------------------------------------------------------------------
    def tick(self) -> None:
        self.ensure_pending_job()
        generation = self.claim_job()
        if generation is None:
            return

        # Consistent snapshot of the corpus for exactly this generation.
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN")
            current, _ = get_meta(conn)
            if current != generation:
                conn.rollback()
                log.info(
                    "job for generation %d superseded (current is %d); dropping",
                    generation,
                    current,
                )
                self.drop_job(generation)
                return
            articles = conn.execute(
                "SELECT id, side, text FROM articles ORDER BY id"
            ).fetchall()
            conn.commit()
        finally:
            conn.close()

        log.info("computing generation %d: %d articles", generation, len(articles))
        edges = compute_edges([(row["id"], row["text"]) for row in articles])
        groups = connected_groups([row["id"] for row in articles], edges)
        side_of = {row["id"]: row["side"] for row in articles}

        # Simulated slow comparison; submissions remain possible meanwhile.
        if self.compare_delay:
            self._stop_event.wait(self.compare_delay)

        self.publish(generation, edges, groups, side_of)

    def publish(self, generation, edges, groups, side_of) -> bool:
        """Atomically swap in the new results.

        The publish transaction re-checks that the corpus has not moved on:
        a stale computation never overwrites newer modifications, and the
        published pointer flips together with the data, so readers never see
        a mix of generations.
        """
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            current, _ = get_meta(conn)
            if current != generation:
                conn.rollback()
                log.warning(
                    "discarding stale results for generation %d (current is %d)",
                    generation,
                    current,
                )
                self.drop_job(generation)
                return False
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM article_groups")
            conn.executemany(
                "INSERT INTO edges (generation, a, b, similarity) VALUES (?, ?, ?, ?)",
                [(generation, a, b, sim) for a, b, sim in edges],
            )
            conn.executemany(
                "INSERT INTO article_groups (generation, group_id, article_id, side)"
                " VALUES (?, ?, ?, ?)",
                [
                    (generation, group_id, member, side_of[member])
                    for group_id, members in enumerate(groups, start=1)
                    for member in members
                ],
            )
            conn.execute(
                "UPDATE meta SET published_generation = ? WHERE id = 1", (generation,)
            )
            conn.execute("DELETE FROM compute_jobs WHERE generation <= ?", (generation,))
            conn.commit()
            log.info(
                "published generation %d: %d edges, %d groups",
                generation,
                len(edges),
                len(groups),
            )
            return True
        finally:
            conn.close()
