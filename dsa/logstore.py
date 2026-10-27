"""Log-structured hash table (the design of Bitcask): an append-only file of records plus
a hash map in memory from each key to where its newest record starts in the file.

    put(key, value, ttl)   O(1)   appends one line at the end of the file; the map points at it
    get(key)               O(1)   one seek and one read of that line
    delete(key)            O(1)   appends a "deleted" record
    compact()              O(live records)   rewrites the file with only the newest, unexpired records

The file is only ever appended to, so writing is fast and a crash can at worst leave a
half-written last line, which loading skips (and cuts off). An update leaves the old
line behind as garbage; when there are more than twice as many lines as live keys, or
the file is over its size limit, compact() rewrites it, keeping the newest records
(and, if it is still too big, dropping the oldest ones, like an LRU at disk scale).
Values stay on disk: memory holds only (offset, length, expiry) per key.

Use in the search engine: the optional cache file for API answers (PAPERSCOPE_CACHE_FILE
in .env, off unless set). A question asked yesterday is answered from the file, so it
costs nothing from the daily OpenAlex budget, even after the app was restarted.
"""
import json
import os
import threading
import time

_MISS = object()


class LogStore:
    def __init__(self, path, max_bytes=200 * 2 ** 20, clock=time.time):
        self.path = path
        self.max_bytes = max_bytes
        self.clock = clock                  # wall-clock time: expiry dates must survive a restart
        self.index = {}                     # key -> (offset, length, expires or None)
        self.lines = 0                      # records in the file, live or not
        self.size = 0
        self.lock = threading.Lock()
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self._load()
        self._open()

    # ------------------------------------------------------------- file handling
    def _open(self):
        self._w = open(self.path, "ab")
        self._r = open(self.path, "rb")

    def _close(self):
        self._w.close()
        self._r.close()

    def _load(self):
        """Rebuild the hash map by reading the log once, from the oldest record to the newest."""
        if not os.path.exists(self.path):
            return
        now, good = self.clock(), 0
        with open(self.path, "rb") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    break                   # a torn last line from a crash: cut off below
                try:
                    key, expires, value = json.loads(line)
                except ValueError:
                    break                   # a damaged record: it and everything after it are dropped
                self.lines += 1
                if value is None or (expires is not None and expires <= now):
                    self.index.pop(key, None)
                else:
                    self.index[key] = (good, len(line), expires)
                good += len(line)
        if good < os.path.getsize(self.path):
            with open(self.path, "r+b") as f:
                f.truncate(good)
        self.size = good

    def _append(self, key, expires, value):
        line = (json.dumps([key, expires, value], separators=(",", ":")) + "\n").encode("utf-8")
        offset = self.size
        self._w.write(line)
        self._w.flush()
        self.size += len(line)
        self.lines += 1
        return offset, len(line)

    # ------------------------------------------------------------- API
    def __len__(self):
        return len(self.index)

    def __contains__(self, key):
        return self.get(key, _MISS) is not _MISS

    def get(self, key, default=None):
        with self.lock:
            entry = self.index.get(key)
            if entry is None:
                return default
            offset, length, expires = entry
            if expires is not None and expires <= self.clock():
                del self.index[key]
                return default
            self._r.seek(offset)
            return json.loads(self._r.read(length))[2]

    def put(self, key, value, ttl=None):
        if value is None:
            raise ValueError("None marks a deleted record; store something else")
        expires = self.clock() + ttl if ttl else None
        with self.lock:
            offset, length = self._append(key, expires, value)
            self.index[key] = (offset, length, expires)
            if self.size > self.max_bytes or self.lines > 2 * len(self.index) + 1000:
                self._compact()

    def delete(self, key):
        with self.lock:
            if self.index.pop(key, None) is not None:
                self._append(key, None, None)

    def compact(self):
        with self.lock:
            self._compact()

    def _compact(self):
        now = self.clock()
        live = sorted((off, length, key, exp) for key, (off, length, exp) in self.index.items()
                      if exp is None or exp > now)                    # oldest record first
        total = sum(length for _, length, _, _ in live)
        while live and total > self.max_bytes // 2:                   # still too big: drop the oldest
            total -= live.pop(0)[1]
        tmp = self.path + ".tmp"
        index, offset = {}, 0
        with open(tmp, "wb") as out:
            for off, length, key, exp in live:
                self._r.seek(off)
                out.write(self._r.read(length))
                index[key] = (offset, length, exp)
                offset += length
        self._close()                       # Windows cannot replace a file that is still open
        os.replace(tmp, self.path)
        self.index, self.size, self.lines = index, offset, len(index)
        self._open()

    def close(self):
        with self.lock:
            self._close()
