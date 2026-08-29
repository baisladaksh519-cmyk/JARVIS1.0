# Improved MemoryStore with lazy embedding model loading, thread-safety, robust serialization, and LRU updates

import sqlite3
import time
import json
import threading
from typing import List, Optional
from io import BytesIO

# Do not import heavy ML libs at module import time. Lazy-load when needed.
_EMBED_MODEL = None
_embed_lock = threading.Lock()


def _load_embed_model_if_needed():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    with _embed_lock:
        if _EMBED_MODEL is not None:
            return _EMBED_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer('all-MiniLM-L6-v2')
            _EMBED_MODEL = model
            return _EMBED_MODEL
        except Exception:
            _EMBED_MODEL = None
            return None


class MemoryStore:
    def __init__(self, db_path: str = 'friday_memory.db', max_entries: int = 500):
        self.db_path = db_path
        self.max_entries = max_entries
        # per-instance lock for thread-safety around sqlite operations
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        # use check_same_thread=False because we may access from multiple threads
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at REAL,
                last_used REAL
            )
        ''')
        self.conn.commit()

    # serialization helpers
    def _vector_to_blob(self, v) -> bytes:
        # Use numpy.save to preserve dtype & shape
        import numpy as _np
        buf = BytesIO()
        _np.save(buf, v, allow_pickle=False)
        buf.seek(0)
        return buf.read()

    def _blob_to_vector(self, b: bytes):
        import numpy as _np
        buf = BytesIO(b)
        buf.seek(0)
        return _np.load(buf, allow_pickle=False)

    def add(self, text: str, metadata: Optional[dict] = None):
        embedding_blob = None
        model = _load_embed_model_if_needed()
        if model is not None:
            try:
                # SentenceTransformer returns numpy array
                vec = model.encode([text])[0]
                # ensure float32
                embedding_blob = self._vector_to_blob(vec.astype('float32'))
            except Exception:
                embedding_blob = None
        created = time.time()
        meta_json = json.dumps(metadata or {})
        with self._lock:
            c = self.conn.cursor()
            c.execute('INSERT INTO memories (text, embedding, metadata, created_at, last_used) VALUES (?,?,?,?,?)',
                      (text, embedding_blob, meta_json, created, created))
            self.conn.commit()
        self._prune_if_needed()

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """If embedding model available, use cosine similarity. Otherwise, use simple LIKE search.
        """
        c = self.conn.cursor()
        results = []
        model = _load_embed_model_if_needed()
        if model is not None:
            try:
                qvec = model.encode([query])[0].astype('float32')
            except Exception:
                qvec = None

            if qvec is not None:
                # fetch rows with embeddings
                with self._lock:
                    c.execute('SELECT id, text, embedding, metadata FROM memories WHERE embedding IS NOT NULL')
                    rows = c.fetchall()
                sims = []
                import numpy as np
                for r in rows:
                    try:
                        vec = self._blob_to_vector(r[2])
                        # cosine similarity
                        denom = (np.linalg.norm(qvec) * (np.linalg.norm(vec) + 1e-10))
                        score = float(np.dot(qvec, vec) / denom) if denom > 0 else 0.0
                        sims.append((score, r))
                    except Exception:
                        continue
                sims.sort(reverse=True, key=lambda x: x[0])
                for score, r in sims[:top_k]:
                    results.append({'id': r[0], 'text': r[1], 'metadata': json.loads(r[3] or '{}'), 'score': float(score)})
            # update last_used for returned results
            if results:
                now = time.time()
                ids = [(now, r['id']) for r in results]
                with self._lock:
                    c.executemany('UPDATE memories SET last_used = ? WHERE id = ?', ids)
                    self.conn.commit()
            return results
        else:
            # fallback text search by LIKE
            with self._lock:
                c.execute('SELECT id, text, metadata FROM memories WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?', (f'%{query}%', top_k))
                rows = c.fetchall()
            results = [{'id': r[0], 'text': r[1], 'metadata': json.loads(r[2] or '{}'), 'score': None} for r in rows]
            if results:
                now = time.time()
                ids = [(now, r['id']) for r in results]
                with self._lock:
                    c.executemany('UPDATE memories SET last_used = ? WHERE id = ?', ids)
                    self.conn.commit()
            return results

    def _prune_if_needed(self):
        with self._lock:
            c = self.conn.cursor()
            c.execute('SELECT COUNT(1) FROM memories')
            count = c.fetchone()[0]
            if count <= self.max_entries:
                return
            # delete oldest entries by created_at
            to_delete = count - self.max_entries
            c.execute('DELETE FROM memories WHERE id IN (SELECT id FROM memories ORDER BY created_at ASC LIMIT ?)', (to_delete,))
            self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
