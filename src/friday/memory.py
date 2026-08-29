# Simple SQLite-backed memory store for FRIDAY

import sqlite3
import time
import json
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_EMBED = True
    _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _HAS_EMBED = False
    _EMBED_MODEL = None


def _vector_to_blob(v: 'numpy.ndarray') -> bytes:
    return v.tobytes()


def _blob_to_vector(b: bytes, dtype='float32'):
    import numpy as np
    return np.frombuffer(b, dtype=dtype)


class MemoryStore:
    def __init__(self, db_path: str = 'friday_memory.db', max_entries: int = 500):
        self.db_path = db_path
        self.max_entries = max_entries
        self._init_db()

    def _init_db(self):
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

    def add(self, text: str, metadata: Optional[dict] = None):
        embedding_blob = None
        if _HAS_EMBED and _EMBED_MODEL is not None:
            vec = _EMBED_MODEL.encode([text])[0]
            embedding_blob = _vector_to_blob(vec.astype('float32'))
        created = time.time()
        meta_json = json.dumps(metadata or {})
        c = self.conn.cursor()
        c.execute('INSERT INTO memories (text, embedding, metadata, created_at, last_used) VALUES (?,?,?,?,?)',
                  (text, embedding_blob, meta_json, created, created))
        self.conn.commit()
        self._prune_if_needed()

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """If embedding model available, use cosine similarity. Otherwise, use simple LIKE search.
        """
        c = self.conn.cursor()
        if _HAS_EMBED and _EMBED_MODEL is not None:
            qvec = _EMBED_MODEL.encode([query])[0].astype('float32')
            # naive linear scan: fetch all embeddings (small footprint expected)
            c.execute('SELECT id, text, embedding, metadata FROM memories')
            rows = c.fetchall()
            sims = []
            import numpy as np
            for r in rows:
                if r[2] is None:
                    continue
                vec = _blob_to_vector(r[2])
                # cosine similarity
                dot = np.dot(qvec, vec)
                denom = (np.linalg.norm(qvec) * (np.linalg.norm(vec) + 1e-10))
                score = dot / denom if denom > 0 else 0
                sims.append((score, r))
            sims.sort(reverse=True, key=lambda x: x[0])
            results = []
            for score, r in sims[:top_k]:
                results.append({'id': r[0], 'text': r[1], 'metadata': json.loads(r[3] or '{}'), 'score': float(score)})
            return results
        else:
            # fallback text search by LIKE
            c.execute('SELECT id, text, metadata FROM memories WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?', (f'%{query}%', top_k))
            rows = c.fetchall()
            return [{'id': r[0], 'text': r[1], 'metadata': json.loads(r[2] or '{}'), 'score': None} for r in rows]

    def _prune_if_needed(self):
        c = self.conn.cursor()
        c.execute('SELECT COUNT(1) FROM memories')
        count = c.fetchone()[0]
        if count <= self.max_entries:
            return
        # delete oldest entries
        to_delete = count - self.max_entries
        c.execute('DELETE FROM memories WHERE id IN (SELECT id FROM memories ORDER BY created_at ASC LIMIT ?)', (to_delete,))
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

