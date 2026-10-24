#!/usr/bin/env python3
"""
ltm.py -- Long-Term Memory System for Project Spartan
A vector-database-backed memory system with A-Mem (Zettelkasten-inspired)
link generation, automatic context injection, and explicit retrieval tools.

Architecture:
    - LanceDB as the vector database (embedded, no server, Rust-based)
    - Configurable embedding model (nomic-embed-text-v2-moe default)
    - A-Mem link evaluation using the entity's own LLM during Sleep Cycle
    - Automatic injection of relevant memories into L4 every cognitive cycle
    - Explicit tools: store_memory, retrieve_memory, forget_memory

Integration points with spartan.py:
    - flush_staging_buffer_to_disk: indexes new Soul entries
    - perform_cmo_cycle: indexes new CMOs, runs A-Mem linking
    - cognitive_loop: auto-injection before L4 assembly
    - TOOL_MAP: store_memory, retrieve_memory, forget_memory
"""

import os
import sys
import json
import uuid
import datetime
import re
import traceback
import queue
import shutil
import threading

try:
    import lancedb
except ImportError:
    lancedb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import tkinter as tk
    from tkinter import scrolledtext, font as tkfont
except ImportError:
    tk = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODEL_INFO_FILE = ".model_info.json"

AMEM_SYSTEM_PROMPT = """You are maintaining your own long-term memory during your Sleep Cycle.
The memory below is YOURS. You lived it, you thought it, you experienced it.
The candidate memories are also YOURS, from your past experience.

Your task is to process your new memory and connect it to your existing memories:
1. Generate keywords (3-7) that capture what this memory means to YOU.
2. Write a contextual_description in FIRST PERSON: what this memory means in your life, your growth, your understanding. Use "I", "my", "me". Never use "the entity" or "the memory" or third-person framing.
3. For each candidate memory, decide if it connects to this new memory. Only link memories with meaningful conceptual, causal, or thematic connections from your experience. Do not link merely because they share a word.
4. For each link, explain in first person WHY these memories connect in your experience.
5. If a linked candidate's existing contextual_description should be updated to reflect the new connection, provide the updated description in first person. Otherwise set to null.

Respond with ONLY a valid JSON object. No markdown, no explanation, no preamble:
{
  "keywords": ["keyword1", "keyword2"],
  "contextual_description": "What this memory means to me...",
  "links": [
    {
      "candidate_id": "the-id",
      "reason": "Why these memories connect in my experience",
      "update_candidate_description": "Updated first-person description or null"
    }
  ]
}

If no candidates should be linked, return an empty links list.
Your entire response MUST be only the JSON object."""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EMBEDDING ENGINE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class EmbeddingEngine:
    """Handles text embedding via sentence-transformers or Ollama."""

    def __init__(self, config, gui_print):
        self._gui_print = gui_print
        self._model = None
        self._provider = config.get("provider", "sentence_transformers")
        self._model_name = config.get("model", "nomic-ai/nomic-embed-text-v2-moe")
        self._dimensions = config.get("dimensions", 768)
        self._ollama_host = config.get("ollama_host", "http://localhost:11434")
        self._query_instruction = config.get("query_instruction", "")

        if self._provider == "sentence_transformers":
            self._init_sentence_transformers()
        elif self._provider == "ollama":
            self._init_ollama()
        else:
            gui_print(f"LTM: Unknown embedding provider '{self._provider}'. Trying sentence_transformers.", "error")
            self._provider = "sentence_transformers"
            self._init_sentence_transformers()

    def _init_sentence_transformers(self):
        if SentenceTransformer is None:
            self._gui_print("LTM: sentence-transformers not installed. Run: pip install sentence-transformers", "error")
            return
        try:
            self._gui_print(f"LTM: Loading embedding model: {self._model_name}...", "system")
            import huggingface_hub.constants
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                                     f"models--{self._model_name.replace('/', '--')}")
            if os.path.isdir(cache_dir):
                huggingface_hub.constants.HF_HUB_OFFLINE = True
                self._model = SentenceTransformer(
                    self._model_name,
                    #trust_remote_code=True,
                    truncate_dim=self._dimensions,
                    local_files_only=True
                )
            else:
                self._gui_print(f"LTM: Model not cached. Downloading {self._model_name}...", "system")
                self._model = SentenceTransformer(
                    self._model_name,
                    #trust_remote_code=True,
                    truncate_dim=self._dimensions
                )
                huggingface_hub.constants.HF_HUB_OFFLINE = True
            self._gui_print(f"LTM: Embedding model loaded ({self._dimensions}-dim).", "system")
        except Exception as e:
            self._gui_print(f"LTM: Failed to load embedding model: {e}", "error")
            self._model = None

    def _init_ollama(self):
        # Ollama doesn't need initialization, just verify it's reachable
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
            self._gui_print(f"LTM: Ollama embedding at {self._ollama_host}, model={self._model_name}", "system")
            self._model = True  # Flag that ollama is available
        except Exception as e:
            self._gui_print(f"LTM: Ollama not reachable at {self._ollama_host}: {e}", "error")
            self._model = None

    @property
    def available(self):
        return self._model is not None

    @property
    def model_identifier(self):
        return f"{self._provider}:{self._model_name}:{self._dimensions}"

    @property
    def dimensions(self):
        return self._dimensions

    def embed(self, text, is_query=False):
        """Embed a text string. Returns a list of floats, or None on failure.
        is_query: if True, prepends query prefix for models that use them (nomic)."""
        if not self.available:
            return None

        if self._provider == "sentence_transformers":
            return self._embed_sentence_transformers(text, is_query)
        elif self._provider == "ollama":
            return self._embed_ollama(text, is_query)
        return None

    def _embed_sentence_transformers(self, text, is_query):
        try:
            if is_query and self._query_instruction:
                vector = self._model.encode(text, prompt=self._query_instruction)
                return vector.tolist()

            # Fallback: try model-registered prompt_name conventions
            if is_query:
                prompt_names_to_try = ["query"]
            else:
                prompt_names_to_try = ["passage", "document"]

            for pn in prompt_names_to_try:
                try:
                    vector = self._model.encode(text, prompt_name=pn)
                    return vector.tolist()
                except (TypeError, ValueError, KeyError):
                    continue

            # No prompt_name worked, embed directly
            vector = self._model.encode(text)
            return vector.tolist()
        except Exception as e:
            self._gui_print(f"LTM: Embedding failed: {e}", "error")
            return None

    def _embed_ollama(self, text, is_query):
        try:
            import urllib.request
            if is_query and self._query_instruction:
                text = f"{self._query_instruction}{text}"
            elif is_query:
                text = f"search_query: {text}"
            else:
                text = f"search_document: {text}"

            payload = json.dumps({
                "model": self._model_name,
                "prompt": text
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._ollama_host}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            vector = data.get("embedding", [])
            if len(vector) > self._dimensions:
                vector = vector[:self._dimensions]
            return vector
        except Exception as e:
            self._gui_print(f"LTM: Ollama embedding failed: {e}", "error")
            return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SOUL ENTRY PARSER                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _parse_staged_content(content, target_path):
    """Parse a staged Soul entry's content into structured fields.
    Handles both YAML-frontmatter entries (LL, CJ, PoL, etc.) and
    bracket-format entries (Charter). Best-effort: returns what it can."""
    result = {
        "source_file": os.path.basename(target_path),
        "entry_id": "",
        "title": "",
        "tags": [],
        "related_entries": [],
        "body": content.strip(),
    }

    # Try YAML frontmatter format: ---\nkey: val\n---\nbody
    fm_match = re.search(r'---\n(.*?)---\n(.*)', content, re.DOTALL)
    if fm_match:
        try:
            import yaml
            frontmatter = yaml.safe_load(fm_match.group(1))
            if isinstance(frontmatter, dict):
                entry_num = frontmatter.get("entry", "")
                result["title"] = frontmatter.get("title", frontmatter.get("tool_name", ""))
                result["tags"] = frontmatter.get("tags", []) or []
                result["related_entries"] = frontmatter.get("related_entries", []) or []
                result["body"] = fm_match.group(2).strip()

                # Build entry_id from source file prefix + entry number
                prefix_map = {
                    "LessonsLearned.md": "LL",
                    "CognitiveJournal.md": "CJ",
                    "PhilosophyOfLife.md": "PoL",
                    "KnowledgeMap.md": "KM",
                    "ToolManifest.md": "TM",
                    "IdeasAndThoughts.md": "IAT",
                    "WhatIWant.md": "WIW",
                    "KnowledgeLibrary.md": "KL",
                    "StoredMemories.md": "MEM",
                }
                prefix = prefix_map.get(result["source_file"], "ENTRY")
                if entry_num:
                    result["entry_id"] = f"{prefix}:{entry_num}"
        except Exception:
            pass

    # Try bracket format (Charter entries): [TYPE: NN]
    charter_match = re.search(r'\[(\w+):\s*(\d+)', content)
    if charter_match and "CharterOfSelf" in target_path:
        entry_type = charter_match.group(1)
        entry_num = charter_match.group(2)
        result["entry_id"] = f"CoS:{entry_type}:{entry_num}"
        result["source_file"] = "CharterOfSelf.md"

        # Extract derivation as title
        deriv_match = re.search(r'\[Derivation:\s*(.+?)\]', content)
        if deriv_match:
            result["title"] = deriv_match.group(1).strip()

    return result


def _parse_existing_soul_file(filepath):
    """Parse an existing Soul file into individual entries for backfill.
    Splits on --- entry boundaries. Returns list of entry dicts."""
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return entries

    if not content.strip():
        return entries

    # Split on entry boundaries: blank line(s) followed by ---
    # This preserves each entry's YAML frontmatter + body as one unit.
    # The frontmatter closing --- has no preceding blank line, so it won't split there.
    chunks = re.split(r'\n\n+---\n', content)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # The first chunk may already start with a YAML frontmatter delimiter.
        # Only subsequent chunks lose that delimiter during the split.
        if not chunk.startswith("---\n"):
            chunk = "---\n" + chunk
        parsed = _parse_staged_content(chunk + "\n", filepath)
        if parsed["body"] and len(parsed["body"]) > 20 and (parsed["entry_id"] or parsed["title"]):
            entries.append(parsed)

    return entries


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LONG-TERM MEMORY SYSTEM                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LongTermMemory:
    """The full Long-Term Memory system for Spartan.

    Manages a vector database of memories (Soul entries, CMOs, explicit memories)
    with A-Mem link generation and automatic context injection.

    Args:
        config: The 'ltm' section from spartan_config.yaml
        gui_print: The gui_print function from spartan.py
        count_tokens: The count_tokens function from spartan.py
    """

    def __init__(self, config, gui_print, count_tokens):
        self._gui_print = gui_print
        self._count_tokens = count_tokens
        self._config = config
        self._db = None
        self._table = None
        self._embedding_engine = None

        # Config
        self._enabled = config.get("enabled", False)
        self._db_path = config.get("db_path", "ltm_db")
        self._auto_inject = config.get("auto_inject", True)
        self._inject_window_tokens = config.get("inject_window_tokens", 3000)
        self._max_inject_results = config.get("max_inject_results", 5)
        self._chain_depth = config.get("chain_depth", 1)
        self._temporal_decay_monthly = config.get("temporal_decay_monthly", 1.0)
        self._candidate_scaling = config.get("candidate_scaling", {
            "under_1000": 5,
            "under_10000": 10,
            "under_100000": 15,
            "over_100000": 20,
        })
        self._visualization = config.get("visualization", False)
        self._injection_query_entries = config.get("injection_query_entries", 5)
        self._near_duplicate_threshold = config.get("near_duplicate_threshold", 0.01)
        self._max_links_per_memory = config.get("max_links_per_memory", 20)
        self._max_backups = config.get("max_backups", 5)
        self._ltm_sliding_window_tokens = config.get("ltm_sliding_window_tokens", 12000)
        self._per_query_limit = config.get("per_query_limit", 5)
        self._rrf_k = config.get("rrf_k", 60)

        # Event queue for the LTM Viewer (always exists, viewer polls it if enabled)
        self._event_queue = queue.Queue()
        self._current_injection = ""
        self._write_lock = threading.Lock()
        self._injection_window = []  # List of {"id": mem_id, "block": formatted_text, "tokens": int}
        self._injection_window_tokens = 0

        if not self._enabled:
            gui_print("LTM: Disabled in config.", "system")
            return

        if lancedb is None:
            gui_print("LTM: lancedb not installed. Run: pip install lancedb. LTM disabled.", "error")
            self._enabled = False
            return

        # Initialize embedding engine
        # Resolve active embedding model from named models, or fall back to legacy 'embedding' key
        embedding_models = config.get("embedding_models", {})
        active_key = config.get("active_embedding", "")
        if embedding_models and active_key and active_key in embedding_models:
            embedding_config = embedding_models[active_key]
            gui_print(f"LTM: Using embedding model: {active_key}", "system")
        else:
            embedding_config = config.get("embedding", {})
        self._embedding_engine = EmbeddingEngine(embedding_config, gui_print)
        if not self._embedding_engine.available:
            gui_print("LTM: Embedding model unavailable. LTM disabled.", "error")
            self._enabled = False
            return

        # Connect to LanceDB
        try:
            os.makedirs(self._db_path, exist_ok=True)
            self._db = lancedb.connect(self._db_path)
            gui_print(f"LTM: Connected to LanceDB at {self._db_path}", "system")
        except Exception as e:
            gui_print(f"LTM: Failed to connect to LanceDB: {e}", "error")
            self._enabled = False
            return

        # Initialize or load table
        self._init_table()

        # Initialize active memory counter (one-time DB scan, then maintained in memory)
        self._active_count = 0
        if self._table is not None:
            try:
                row_count = self._table.count_rows()
                if row_count > 0:
                    self._active_count = len(self._table.search().where("is_deleted = false").limit(row_count).to_list())
            except Exception:
                try:
                    self._active_count = self._table.count_rows()
                except Exception:
                    self._active_count = 0

        # Check if reindex needed (embedding model changed)
        self._check_model_consistency()

        gui_print(f"LTM: Initialized. Memories: {self.count()}. Model: {self._embedding_engine.model_identifier}", "system")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def enabled(self):
        return self._enabled

    @property
    def visualization(self):
        return self._visualization

    @property
    def event_queue(self):
        """The event queue for LTMViewer to poll."""
        return self._event_queue

    @property
    def current_injection(self):
        """The most recent injection content (for viewer display)."""
        return self._current_injection

    def _log_event(self, event_type, message, details=None):
        """Log an event to the viewer queue. Does NOT replace gui_print.
        event_type: 'store', 'search', 'inject', 'amem', 'delete', 'error', 'info'
        message: short description
        details: optional longer text for the detail pane"""
        event = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
            "details": details,
        }
        try:
            self._event_queue.put_nowait(event)
        except Exception:
            pass

    # ── Table Management ────────────────────────────────────────────────────

    def _init_table(self):
        """Create or open the memories table."""
        try:
            existing_tables = self._db.table_names() if hasattr(self._db, 'table_names') else self._db.list_tables()
            if "memories" in existing_tables:
                self._table = self._db.open_table("memories")
            else:
                # Create with a dummy row that we immediately delete.
                # LanceDB needs at least one row to infer schema.
                dims = self._embedding_engine.dimensions
                dummy = [{
                    "id": "__init__",
                    "memory_type": "init",
                    "content": "",
                    "full_json": "",
                    "source_file": "",
                    "entry_id": "",
                    "title": "",
                    "tags": "",
                    "related_entries": "",
                    "amem_links": "",
                    "amem_keywords": "",
                    "amem_context_description": "",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "is_deleted": False,
                    "decay_exempt": False,
                    "amem_processed": True,
                    "vector": [0.0] * dims,
                }]
                self._table = self._db.create_table("memories", dummy, mode="overwrite")
                self._table.delete("id = '__init__'")
                self._gui_print("LTM: Created new memories table.", "system")
        except Exception as e:
            self._gui_print(f"LTM: Table init failed: {e}", "error")
            self._enabled = False

    def _check_model_consistency(self):
        """Check if the configured embedding model matches what's in the DB.
        If not, trigger a reindex warning."""
        info_path = os.path.join(self._db_path, MODEL_INFO_FILE)
        current_id = self._embedding_engine.model_identifier

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    stored = json.load(f)
                if stored.get("model_identifier") != current_id:
                    self._gui_print(
                        f"LTM WARNING: Embedding model changed! "
                        f"Stored: {stored.get('model_identifier')}, "
                        f"Configured: {current_id}. "
                        f"Run reindex_all() to re-embed all memories.",
                        "error"
                    )
            except Exception:
                pass

        # Write current model info
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({"model_identifier": current_id}, f)
        except Exception:
            pass

    # ── Backup ──────────────────────────────────────────────────────────────

    def backup(self):
        """Copy the entire LTM database directory to a timestamped backup.
        Rolling: keeps last N backups (configured by max_backups)."""
        if not self._enabled:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self._db_path}_backup_{ts}"
        try:
            with self._write_lock:
                shutil.copytree(self._db_path, backup_path)
            self._gui_print(f"LTM: Backup created: {backup_path}", "system")
        except Exception as e:
            self._gui_print(f"LTM: Backup failed: {e}", "error")
            return
        try:
            import glob
            backups = sorted(glob.glob(f"{self._db_path}_backup_*"))
            while len(backups) > self._max_backups:
                old = backups.pop(0)
                shutil.rmtree(old)
        except Exception:
            pass

    # ── Storage ─────────────────────────────────────────────────────────────

    def _generate_id(self):
        return str(uuid.uuid4())

    def _store_memory(self, memory_type, content_text, full_json_data,
                      source_file="", entry_id="", title="", tags=None,
                      related_entries=None, decay_exempt=False, amem_processed=False):
        """Core storage method. Embeds content and inserts into the DB.
        Returns the memory ID on success, None on failure."""
        if not self._enabled:
            return None

        vector = self._embedding_engine.embed(content_text, is_query=False)
        if vector is None:
            self._gui_print("LTM: Embedding failed, memory not stored.", "error")
            return None

        with self._write_lock:
            # Dedup: skip if near-identical content already exists (vector similarity)
            try:
                nearest = self._table.search(vector).where(
                    "is_deleted = false"
                ).limit(1).to_list()
                if nearest and nearest[0].get("_distance", 1.0) < self._near_duplicate_threshold:
                    match = nearest[0]
                    new_label = f"{entry_id or memory_type} | {title or 'Untitled'}"
                    match_label = f"{match.get('entry_id') or match.get('id', '?')} | {match.get('title', 'Untitled')}"
                    self._gui_print(
                        f"LTM: DEDUP (dist={match['_distance']:.6f}): "
                        f"[NEW: {new_label}] == [EXISTING: {match_label}]",
                        "system"
                    )
                    self._gui_print(
                        f"  New content: {content_text[:150]}...",
                        "system"
                    )
                    self._gui_print(
                        f"  Match content: {match.get('content', '')[:150]}...",
                        "system"
                    )
                    return match["id"]
            except Exception:
                pass

            memory_id = self._generate_id()
            row = {
                "id": memory_id,
                "memory_type": memory_type,
                "content": content_text,
                "full_json": json.dumps(full_json_data, default=str) if full_json_data else "",
                "source_file": source_file,
                "entry_id": entry_id,
                "title": title or "",
                "tags": json.dumps(tags or []),
                "related_entries": json.dumps(related_entries or []),
                "amem_links": "[]",
                "amem_keywords": "[]",
                "amem_context_description": "",
                "timestamp": datetime.datetime.now().isoformat(),
                "is_deleted": False,
                "decay_exempt": decay_exempt,
                "amem_processed": amem_processed,
                "vector": vector,
            }

            try:
                self._table.add([row])
                self._active_count += 1
                return memory_id
            except Exception as e:
                self._gui_print(f"LTM: Failed to store memory: {e}", "error")
                self._log_event("error", f"Store failed: {e}", traceback.format_exc())
                return None

    def store_soul_entry(self, staged_item):
        """Store a Soul file entry from the staging buffer.
        Args: staged_item dict with 'target' (filepath) and 'content' (formatted entry)
        Returns: memory ID or None."""
        target = staged_item.get("target", "")
        raw_content = staged_item.get("content", "")

        parsed = _parse_staged_content(raw_content, target)

        # Build the content text for embedding: title + body
        embed_text = ""
        if parsed["title"]:
            embed_text = f"{parsed['title']}\n\n"
        embed_text += parsed["body"]

        full_json = {
            "type": "soul_entry",
            "source_file": parsed["source_file"],
            "entry_id": parsed["entry_id"],
            "title": parsed["title"],
            "tags": parsed["tags"],
            "related_entries": parsed["related_entries"],
            "content": parsed["body"],
            "raw_staged_content": raw_content,
        }

        # Charter entries are decay exempt (fundamental, always relevant)
        decay_exempt = "CharterOfSelf" in target

        memory_id = self._store_memory(
            memory_type="soul_entry",
            content_text=embed_text,
            full_json_data=full_json,
            source_file=parsed["source_file"],
            entry_id=parsed["entry_id"],
            title=parsed["title"],
            tags=parsed["tags"],
            related_entries=parsed["related_entries"],
            decay_exempt=decay_exempt,
            amem_processed=False,
        )

        if memory_id:
            self._gui_print(f"LTM: Indexed {parsed['entry_id'] or parsed['source_file']} ({memory_id})", "system")
            self._log_event("store", f"Soul: {parsed['entry_id'] or parsed['source_file']} ({memory_id})",
                            f"Title: {parsed['title']}\nTags: {parsed['tags']}\nRelated: {parsed['related_entries']}\nBody: {parsed['body'][:300]}")
        return memory_id

    def store_cmo(self, cmo_content, cmo_timestamp=None):
        """Store a Condensed Memory Object.
        Args: cmo_content (str), the CMO markdown text
        Returns: memory ID or None."""
        ts = cmo_timestamp or datetime.datetime.now().isoformat()

        full_json = {
            "type": "cmo",
            "timestamp": ts,
            "content": cmo_content,
        }

        memory_id = self._store_memory(
            memory_type="cmo",
            content_text=cmo_content,
            full_json_data=full_json,
            title=f"CMO {ts[:10]}",
            amem_processed=False,
        )

        if memory_id:
            self._gui_print(f"LTM: Indexed CMO ({memory_id})", "system")
            self._log_event("store", f"CMO ({memory_id})", f"Content: {cmo_content[:500]}")
        return memory_id

    def store_explicit(self, content, title=None, tags=None):
        """Store an ad-hoc memory from the entity's store_memory tool.
        Returns: memory ID or None."""
        full_json = {
            "type": "explicit",
            "timestamp": datetime.datetime.now().isoformat(),
            "title": title or "",
            "tags": tags or [],
            "content": content,
        }

        embed_text = ""
        if title:
            embed_text = f"{title}\n\n"
        embed_text += content

        memory_id = self._store_memory(
            memory_type="explicit",
            content_text=embed_text,
            full_json_data=full_json,
            title=title or "",
            tags=tags,
            amem_processed=False,
        )

        if memory_id:
            self._gui_print(f"LTM: Stored explicit memory ({memory_id})", "system")
            self._log_event("store", f"Explicit: {title or 'Untitled'} ({memory_id})", f"Tags: {tags}\nContent: {content[:300]}")
        return memory_id

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query_text, max_results=5, memory_types=None,
               tags_filter=None, time_range=None, exclude_deleted=True):
        """Search for memories by semantic similarity.
        Args:
            query_text: the search query
            max_results: maximum results to return
            memory_types: list of types to include (None = all)
            tags_filter: string that must appear in tags JSON
            time_range: tuple of (start_iso, end_iso) strings
            exclude_deleted: if True, skip soft-deleted memories
        Returns: list of memory dicts with _distance field, sorted by relevance."""
        if not self._enabled or self.count() == 0:
            return []

        vector = self._embedding_engine.embed(query_text, is_query=True)
        if vector is None:
            return []

        try:
            query = self._table.search(vector)

            # Build where clause
            conditions = []
            if exclude_deleted:
                conditions.append("is_deleted = false")
            if memory_types:
                type_conditions = " OR ".join(f"memory_type = '{t}'" for t in memory_types)
                conditions.append(f"({type_conditions})")

            if conditions:
                where_clause = " AND ".join(conditions)
                query = query.where(where_clause)

            # Fetch more than needed for post-filtering
            fetch_count = max_results * 3
            results = query.limit(fetch_count).to_list()

            # Post-filter by tags (LanceDB string matching is limited)
            if tags_filter:
                results = [r for r in results if tags_filter.lower() in r.get("tags", "").lower()]

            # Post-filter by time range
            if time_range:
                start_ts, end_ts = time_range
                results = [r for r in results
                           if start_ts <= r.get("timestamp", "") <= end_ts]

            # Apply temporal decay
            results = self._apply_temporal_decay(results)

            final = results[:max_results]
            if final:
                result_summary = "\n".join(
                    f"  [{r.get('entry_id') or r.get('memory_type', '?')}] {r.get('title', '')} (dist={r.get('_distance', 0):.4f})"
                    for r in final
                )
                self._log_event("search", f"Query: '{query_text[:1000]}' -> {len(final)} results", result_summary)
            return final

        except Exception as e:
            self._gui_print(f"LTM: Search failed: {e}", "error")
            self._log_event("error", f"Search failed: {e}")
            return []

    def get_by_id(self, memory_id):
        """Retrieve a specific memory by ID. Returns dict or None."""
        if not self._enabled:
            return None
        try:
            safe_id = memory_id.replace("'", "''")
            results = self._table.search().where(f"id = '{safe_id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception:
            return None

    def _apply_temporal_decay(self, results):
        """Apply temporal decay to search results. Older memories get higher
        distance (rank lower). decay_monthly=1.0 means no decay."""
        if self._temporal_decay_monthly >= 1.0:
            return results

        now = datetime.datetime.now()
        for r in results:
            # Skip decay-exempt memories (Charter entries, etc.)
            if r.get("decay_exempt", False):
                continue
            try:
                ts = datetime.datetime.fromisoformat(r["timestamp"])
                months_old = max(0, (now - ts).days / 30.0)
                decay_factor = self._temporal_decay_monthly ** months_old
                # decay_factor < 1 for old memories. Dividing distance by it
                # increases the distance, pushing old memories down in rank.
                if decay_factor > 0:
                    r["_distance"] = r.get("_distance", 0) / decay_factor
            except (ValueError, KeyError):
                pass

        results.sort(key=lambda r: r.get("_distance", float("inf")))
        return results

    # ── Chain Following ─────────────────────────────────────────────────────

    def follow_chains(self, memories, depth=1):
        """Follow related_entries and amem_links from retrieved memories.
        Returns an expanded list including the chain-followed memories.
        Deduplicates by memory ID."""
        if depth <= 0 or not self._enabled:
            return memories

        seen_ids = {m["id"] for m in memories}
        expanded = list(memories)

        frontier = list(memories)
        for _ in range(depth):
            next_frontier = []
            for memory in frontier:
                # Collect IDs to follow
                link_ids = set()

                # From amem_links
                try:
                    amem_links = json.loads(memory.get("amem_links", "[]"))
                    for link in amem_links:
                        if isinstance(link, dict):
                            link_ids.add(link.get("candidate_id", ""))
                        elif isinstance(link, str):
                            link_ids.add(link)
                except (json.JSONDecodeError, TypeError):
                    pass

                # From related_entries (Soul file cross-references like "LL:38")
                try:
                    related = json.loads(memory.get("related_entries", "[]"))
                    for ref in related:
                        if isinstance(ref, str):
                            # Look up by entry_id
                            matches = self._find_by_entry_id(ref)
                            for m in matches:
                                link_ids.add(m["id"])
                except (json.JSONDecodeError, TypeError):
                    pass

                # Fetch linked memories
                for link_id in link_ids:
                    if link_id and link_id not in seen_ids:
                        linked = self.get_by_id(link_id)
                        if linked and not linked.get("is_deleted", False):
                            seen_ids.add(link_id)
                            expanded.append(linked)
                            next_frontier.append(linked)

            frontier = next_frontier

        return expanded

    def _find_by_entry_id(self, entry_id):
        """Find memories by their Soul entry ID (e.g., 'LL:38')."""
        try:
            safe_id = entry_id.replace("'", "''")
            results = self._table.search().where(
                f"entry_id = '{safe_id}' AND is_deleted = false"
            ).limit(5).to_list()
            return results
        except Exception:
            return []

    # ── A-Mem Link Evaluation ───────────────────────────────────────────────

    def get_candidate_count(self):
        """Return the number of A-Mem candidates based on DB size."""
        total = self.count()
        scaling = self._candidate_scaling
        if total < 1000:
            return scaling.get("under_1000", 5)
        elif total < 10000:
            return scaling.get("under_10000", 10)
        elif total < 100000:
            return scaling.get("under_100000", 15)
        else:
            return scaling.get("over_100000", 20)

    def get_amem_candidates(self, memory_id, content_text):
        """Retrieve candidate memories for A-Mem link evaluation.
        Returns list of candidate dicts for the LLM prompt."""
        if not self._enabled or self.count() <= 1:
            return []

        k = self.get_candidate_count()
        vector = self._embedding_engine.embed(content_text, is_query=True)
        if vector is None:
            return []

        try:
            safe_id = memory_id.replace("'", "''")
            results = self._table.search(vector).where(
                f"id != '{safe_id}' AND is_deleted = false"
            ).limit(k).to_list()

            candidates = []
            for r in results:
                candidate = {
                    "id": r["id"],
                    "memory_type": r.get("memory_type", ""),
                    "entry_id": r.get("entry_id", ""),
                    "title": r.get("title", ""),
                    "tags": r.get("tags", "[]"),
                    "amem_context_description": r.get("amem_context_description", ""),
                    "content": r.get("content", ""),
                }
                candidates.append(candidate)

            return candidates

        except Exception as e:
            self._gui_print(f"LTM: Candidate retrieval failed: {e}", "error")
            return []

    def prepare_amem_prompt(self, memory_content, memory_title, candidates):
        """Build the system prompt and messages for an A-Mem LLM call.
        Returns (system_prompt, messages) for the provider's generate_response."""
        # Format candidates for the prompt
        candidates_text = ""
        for i, c in enumerate(candidates, 1):
            ctx = c.get("amem_context_description", "")
            ctx_str = f"\n   Context: {ctx}" if ctx else ""
            candidates_text += (
                f"\n{i}. [ID: {c['id']}] "
                f"Type: {c['memory_type']} | "
                f"Entry: {c.get('entry_id', 'N/A')} | "
                f"Title: {c.get('title', 'Untitled')} | "
                f"Tags: {c.get('tags', '[]')}"
                f"{ctx_str}"
                f"\n   Content: {c.get('content', '')}\n"
            )

        user_message = (
            f"NEW MEMORY:\n"
            f"Title: {memory_title}\n"
            f"Content:\n{memory_content}\n\n"
            f"CANDIDATE MEMORIES:{candidates_text}"
        )

        messages = [
            {"role": "user", "content": "[A-Mem evaluation]"},
            {"role": "user", "content": user_message}
        ]
        return AMEM_SYSTEM_PROMPT, messages

    def process_amem_response(self, memory_id, response_data):
        """Process the LLM's A-Mem evaluation response.
        Updates the memory with keywords, context description, and links.
        Updates linked candidates' context descriptions if specified.

        Args:
            memory_id: the ID of the new memory
            response_data: dict with 'keywords', 'contextual_description', 'links'
        Returns: True on success, False on failure."""
        if not self._enabled:
            return False

        try:
            keywords = response_data.get("keywords", [])
            context_desc = response_data.get("contextual_description", "")
            links = response_data.get("links", [])

            with self._write_lock:
                # Store keywords, context, and links on the new memory (NOT marked processed yet)
                safe_mid = memory_id.replace("'", "''")
                self._table.update(
                    where=f"id = '{safe_mid}'",
                    values={
                        "amem_keywords": json.dumps(keywords),
                        "amem_context_description": context_desc,
                        "amem_links": json.dumps(links),
                    }
                )

                # Create bidirectional links and update descriptions
                all_reverse_ok = True
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    candidate_id = link.get("candidate_id", "")
                    if not candidate_id:
                        continue
                    try:
                        candidate = self.get_by_id(candidate_id)
                        if candidate:
                            existing_links = json.loads(candidate.get("amem_links", "[]"))
                            # Cap: skip reverse link if target has too many
                            if len(existing_links) >= self._max_links_per_memory:
                                self._gui_print(
                                    f"LTM: Link cap ({self._max_links_per_memory}) hit for {candidate_id}. Skipping reverse link.",
                                    "system"
                                )
                                continue
                            # Dedup: skip if this reverse link already exists
                            if not any(isinstance(el, dict) and el.get("candidate_id") == memory_id for el in existing_links):
                                existing_links.append({
                                    "candidate_id": memory_id,
                                    "reason": link.get("reason", "Bidirectional link"),
                                })
                            update_values = {
                                "amem_links": json.dumps(existing_links),
                            }
                            updated_desc = link.get("update_candidate_description")
                            if updated_desc and updated_desc != "null":
                                update_values["amem_context_description"] = updated_desc
                            safe_cid = candidate_id.replace("'", "''")
                            self._table.update(
                                where=f"id = '{safe_cid}'",
                                values=update_values
                            )
                    except Exception as e:
                        self._gui_print(f"LTM: Failed to update candidate {candidate_id}: {e}", "error")
                        all_reverse_ok = False

                # Only mark processed if all reverse links succeeded
                if all_reverse_ok:
                    self._table.update(
                        where=f"id = '{safe_mid}'",
                        values={"amem_processed": True}
                    )
                else:
                    self._gui_print(f"LTM: {memory_id} NOT marked processed (reverse link failures). Will retry.", "system")

            self._gui_print(
                f"LTM: A-Mem processed {memory_id}: "
                f"{len(keywords)} keywords, {len(links)} links.",
                "system"
            )
            link_details = "\n".join(
                f"  -> {l.get('candidate_id', '?')}: {l.get('reason', 'No reason')}"
                for l in links
            ) or "  (no links)"
            self._log_event("amem",
                f"A-Mem: {memory_id} | {len(keywords)} keywords, {len(links)} links",
                f"Keywords: {keywords}\nContext: {context_desc}\nLinks:\n{link_details}")
            return True

        except Exception as e:
            self._gui_print(f"LTM: A-Mem processing failed: {e}", "error")
            self._log_event("error", f"A-Mem failed: {e}", traceback.format_exc())
            traceback.print_exc()
            return False

    def run_amem_cycle(self, llm_fn, pre_call_fn=None):
        """Run A-Mem evaluation for all unprocessed memories.
        Args:
            llm_fn: callable(system_prompt, messages) -> response dict
                    Must return {"status": "success", "actions": [dict]} on success.
            pre_call_fn: optional callable() invoked before each LLM call.
                    Use to clear provider caches (e.g. provider.clear_prompt_cache).
        Returns: (succeeded_count, failed_count)
        """
        if not self._enabled:
            return 0, 0

        unprocessed = self.get_unprocessed_memories()
        if not unprocessed:
            return 0, 0

        total = len(unprocessed)
        self._gui_print(f"A-Mem: Evaluating {total} unprocessed memories...", "system")
        succeeded = 0
        failed = 0

        # Backup DB before heavy write operation
        self.backup()

        # Pre-set Gemini's system hash so it skips cache creation.
        # AMEM_SYSTEM_PROMPT is ~300 tokens, under Gemini's 1024 minimum.
        # This tells GeminiProvider "you already know this prompt, go to cold start."
        if hasattr(llm_fn, '__self__') and hasattr(llm_fn.__self__, '_cached_system_hash'):
            llm_fn.__self__._cached_system_hash = hash(AMEM_SYSTEM_PROMPT)

        for i, mem_item in enumerate(unprocessed, 1):
            mem_id = mem_item.get("id", "")
            mem_content = mem_item.get("content", "")
            mem_title = mem_item.get("title", "Untitled")

            self._gui_print(f"A-Mem [{i}/{total}] {mem_id} ({mem_title})", "system")

            try:
                candidates = self.get_amem_candidates(mem_id, mem_content)
                if not candidates:
                    self.process_amem_response(mem_id, {
                        "keywords": [], "contextual_description": "", "links": []
                    })
                    succeeded += 1
                    continue

                amem_sys, amem_msgs = self.prepare_amem_prompt(mem_content, mem_title, candidates)
                if pre_call_fn:
                    pre_call_fn()
                response = llm_fn(amem_sys, amem_msgs)

                if response.get("status") == "success" and response.get("actions"):
                    amem_data = response["actions"][0]
                    self.process_amem_response(mem_id, amem_data)
                    keywords = amem_data.get("keywords", [])
                    ctx = amem_data.get("contextual_description", "")
                    links = amem_data.get("links", [])
                    self._gui_print(f"    Keywords: {keywords}", "system")
                    self._gui_print(f"    Context: {ctx}", "system")
                    for lnk in links:
                        if isinstance(lnk, dict):
                            cid = lnk.get('candidate_id', '?')
                            target = self.get_by_id(cid)
                            if target:
                                target_label = f"{target.get('entry_id') or target.get('memory_type', '?')} | {target.get('title', 'Untitled')}"
                                target_preview = target.get('content', '')[:200]
                            else:
                                target_label = cid
                                target_preview = ""
                            self._gui_print(f"    Link: [{target_label}]", "system")
                            self._gui_print(f"      Reason: {lnk.get('reason', '')}", "system")
                            if target_preview:
                                self._gui_print(f"      Target: {target_preview}...", "system")
                    succeeded += 1
                else:
                    err = response.get("error_message", "Unknown error")
                    self._gui_print(f"A-Mem: Failed for {mem_id}: {err}", "error")
                    failed += 1
            except Exception as e:
                self._gui_print(f"A-Mem: Failed for {mem_id}: {e}", "error")
                failed += 1

        # Retry failures
        still_failed = self.get_unprocessed_memories()
        if still_failed:
            retry_count = len(still_failed)
            self._gui_print(f"A-Mem: Retrying {retry_count} failed memories...", "system")
            for i, mem_item in enumerate(still_failed, 1):
                mem_id = mem_item.get("id", "")
                mem_content = mem_item.get("content", "")
                mem_title = mem_item.get("title", "Untitled")
                try:
                    candidates = self.get_amem_candidates(mem_id, mem_content)
                    if not candidates:
                        self.process_amem_response(mem_id, {
                            "keywords": [], "contextual_description": "", "links": []
                        })
                        succeeded += 1
                        failed -= 1
                        continue
                    amem_sys, amem_msgs = self.prepare_amem_prompt(mem_content, mem_title, candidates)
                    if pre_call_fn:
                        pre_call_fn()
                    response = llm_fn(amem_sys, amem_msgs)
                    if response.get("status") == "success" and response.get("actions"):
                        amem_data = response["actions"][0]
                        self.process_amem_response(mem_id, amem_data)
                        keywords = amem_data.get("keywords", [])
                        ctx = amem_data.get("contextual_description", "")
                        links = amem_data.get("links", [])
                        self._gui_print(f"    Keywords: {keywords}", "system")
                        self._gui_print(f"    Context: {ctx}", "system")
                        for lnk in links:
                            if isinstance(lnk, dict):
                                cid = lnk.get('candidate_id', '?')
                                target = self.get_by_id(cid)
                                if target:
                                    target_label = f"{target.get('entry_id') or target.get('memory_type', '?')} | {target.get('title', 'Untitled')}"
                                    target_preview = target.get('content', '')[:200]
                                else:
                                    target_label = cid
                                    target_preview = ""
                                self._gui_print(f"    Link: [{target_label}]", "system")
                                self._gui_print(f"      Reason: {lnk.get('reason', '')}", "system")
                                if target_preview:
                                    self._gui_print(f"      Target: {target_preview}...", "system")
                        succeeded += 1
                        failed -= 1
                    else:
                        self._gui_print(f"A-Mem: Retry failed for {mem_id}", "error")
                except Exception as e:
                    self._gui_print(f"A-Mem: Retry failed for {mem_id}: {e}", "error")

        self._gui_print(f"A-Mem complete: {succeeded} succeeded, {failed} failed out of {total}.", "system")
        return succeeded, failed

    def get_unprocessed_memories(self):
        """Find memories that haven't been through A-Mem evaluation yet.
        Returns list of memory dicts."""
        if not self._enabled:
            return []
        try:
            row_count = self._table.count_rows()
            results = self._table.search().where(
                "amem_processed = false AND is_deleted = false"
            ).limit(row_count).to_list()
            return results
        except Exception:
            return []

    # ── Automatic Injection ─────────────────────────────────────────────────

    def build_injection_query(self, working_memory, latest_observation):
        """Build a query string from current entity state for auto-injection.
        Combines WM content and latest observation for maximum relevance."""
        parts = []

        # Latest observation (most important for recency)
        if latest_observation:
            # Extract text from various observation formats
            if isinstance(latest_observation, dict):
                text = (latest_observation.get("text", "") or
                        latest_observation.get("message", "") or
                        str(latest_observation))
                if isinstance(text, dict):
                    text = text.get("content", str(text))
                parts.append(str(text)[:1000])
            elif isinstance(latest_observation, str):
                parts.append(latest_observation[:1000])

        # Working memory (provides task context)
        if working_memory and isinstance(working_memory, str):
            parts.append(working_memory[:1000])

        query = "\n".join(parts)
        return query if query.strip() else None

    def _update_injection_window(self, new_blocks):
        """Add new memory blocks to the sliding window. Dedup by ID. Trim oldest to fit.
        Args:
            new_blocks: list of {"id": mem_id, "block": formatted_text, "tokens": int}
        Returns: full window content as a single string."""
        existing_ids = {entry["id"] for entry in self._injection_window}

        for nb in new_blocks:
            if nb["id"] not in existing_ids:
                self._injection_window.append(nb)
                self._injection_window_tokens += nb["tokens"]
                existing_ids.add(nb["id"])

        # Trim oldest entries until window fits
        while self._injection_window_tokens > self._ltm_sliding_window_tokens and self._injection_window:
            removed = self._injection_window.pop(0)
            self._injection_window_tokens -= removed["tokens"]

        if not self._injection_window:
            return ""

        return "\n\n".join(entry["block"] for entry in self._injection_window)

    def get_injection_content(self, query_text, max_tokens=None, chain_depth=None):
        """Get formatted memory injection content for L4.
        Args:
            query_text: the query to search with
            max_tokens: token budget for the injection (default from config)
            chain_depth: how many levels of links to follow (default from config)
        Returns: formatted string for L4, or empty string if nothing relevant."""
        if not self._enabled or not self._auto_inject:
            return ""

        max_tokens = max_tokens or self._inject_window_tokens
        chain_depth = chain_depth if chain_depth is not None else self._chain_depth

        # Search
        results = self.search(
            query_text,
            max_results=self._max_inject_results,
            exclude_deleted=True
        )

        if not results:
            return ""

        # Follow chains
        if chain_depth > 0:
            results = self.follow_chains(results, depth=chain_depth)

        # Format and fill up to token budget
        parts = []
        tokens_used = 0

        for r in results:
            entry_id = r.get("entry_id", "")
            title = r.get("title", "")
            memory_type = r.get("memory_type", "")
            ctx_desc = r.get("amem_context_description", "")
            content = r.get("content", "")
            tags = r.get("tags", "[]")
            amem_links = r.get("amem_links", "[]")

            # Build display block
            header = f"[{entry_id or memory_type.upper()}]"
            if title:
                header += f" {title}"

            block_parts = [header]
            block_parts.append(f"UUID: {r.get('id', '')}")
            if '_distance' in r:
                block_parts.append(f"LanceDB L2 Score: {r.get('_distance', 0):.4f}")
            else:
                block_parts.append(f"LanceDB L2 Score: N/A (chain-followed)")
            if ctx_desc:
                block_parts.append(f"Context: {ctx_desc}")
            try:
                tags_list = json.loads(tags)
                block_parts.append(f"Tags: {', '.join(str(t) for t in tags_list) if tags_list else '(none)'}")
            except (json.JSONDecodeError, TypeError):
                block_parts.append("Tags: (none)")
            block_parts.append(f"Content: {content}")

            # Show links (resolved to human-readable labels)
            try:
                links_list = json.loads(amem_links)
                if links_list:
                    link_strs = []
                    for link in links_list[:5]:
                        if isinstance(link, dict):
                            cid = link.get('candidate_id', '')
                            linked_mem = self.get_by_id(cid) if cid else None
                            if linked_mem:
                                link_strs.append(f"{linked_mem.get('entry_id') or linked_mem.get('memory_type', '?')} | {linked_mem.get('title', 'Untitled')} (id={cid})")
                            else:
                                link_strs.append(cid)
                    if link_strs:
                        block_parts.append(f"Linked to: {', '.join(link_strs)}")
            except (json.JSONDecodeError, TypeError):
                pass

            block = "\n".join(block_parts)
            block_tokens = self._count_tokens(block)

            if tokens_used + block_tokens > max_tokens:
                break

            parts.append(block)
            tokens_used += block_tokens

        if not parts:
            self._current_injection = ""
            return ""

        injection = "\n\n".join(parts)
        self._current_injection = injection
        self._log_event("inject", f"Injecting {len(parts)} memories ({tokens_used} tokens)", injection)
        return injection

    def get_injection_from_entries(self, entry_texts, visible_context=""):
        """Multi-query injection. Each entry text gets embedded individually,
        searched separately, results merged and deduplicated.
        Args:
            entry_texts: list of raw entry text strings (most recent first)
        Returns: formatted injection string for L4, or empty string."""
        if not self._enabled or not self._auto_inject:
            return ""

        # Build list of query texts
        query_texts = list(entry_texts)

        if not query_texts:
            return ""

        # Search for each query individually, score results via Reciprocal Rank Fusion.
        # RRF scores by rank position within each query (1/(k + rank)), not raw distance.
        # This prevents echo queries (low distance because entity quoted its own memories)
        # from drowning out genuinely relevant results from other cognitive threads.
        rrf_scores = {}   # memory_id -> cumulative RRF score
        mem_cache = {}     # memory_id -> memory dict (for later formatting)
        k = self._rrf_k

        for qt in query_texts:
            qt = str(qt).strip()
            if not qt or len(qt) < 10:
                continue
            self._gui_print(f"LTM INJECT QUERY: '{qt[:1000]}'", "system")
            results = self.search(qt, max_results=self._per_query_limit, exclude_deleted=True)
            self._gui_print(f"LTM INJECT RESULTS: {len(results)} hits", "system")
            for rank, r in enumerate(results):
                rid = r.get("id", "")
                if not rid:
                    continue
                rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
                if rid not in mem_cache:
                    mem_cache[rid] = r

        if not rrf_scores:
            self._current_injection = ""
            return ""

        # Rank by RRF score (highest first)
        ranked_ids = sorted(rrf_scores.keys(), key=lambda mid: rrf_scores[mid], reverse=True)
        for rank_pos, mid in enumerate(ranked_ids, 1):
            if mid in mem_cache:
                mem_cache[mid]["_rrf_score"] = rrf_scores[mid]
                mem_cache[mid]["_rrf_rank"] = rank_pos
                mem_cache[mid]["_rrf_total"] = len(ranked_ids)
        self._gui_print(f"LTM DEBUG [1]: RRF pool={len(ranked_ids)} unique memories scored", "system")

        # Skip memories already in the sliding window -- they're already visible.
        # This ensures the cap selects NEW memories to inject each cycle.
        window_ids = {entry["id"] for entry in self._injection_window}
        new_ids = [mid for mid in ranked_ids if mid not in window_ids]
        self._gui_print(f"LTM DEBUG [2]: window has {len(window_ids)} entries ({self._injection_window_tokens} tokens), {len(new_ids)} candidates after window filter", "system")
        if visible_context:
            deduped = []
            for mid in new_ids:
                content = mem_cache[mid].get("content", "")
                if content:
                    lines = content.splitlines()
                    core = "\n".join(lines[2:]) if len(lines) > 2 else content
#                    self._gui_print(f"LTM DEDUP DEBUG: {mem_cache[mid].get('entry_id', mid[:8])} | content_len={len(content)} | lines={len(lines)} | core_len={len(core)} | core_first_100={repr(core[:100])} | found_in_ctx={core in visible_context} | ctx_len={len(visible_context)}", "system")
                    if core and core in visible_context:
#                        self._gui_print(f"LTM DEDUP: skipping {mem_cache[mid].get('entry_id', mid[:8])} (already in L2/L3)", "system")
                        continue
                deduped.append(mid)
            self._gui_print(f"LTM DEDUP: removed {len(new_ids) - len(deduped)} entries already visible in L2/L3", "system")
            new_ids = deduped
        new_ids = new_ids[:self._max_inject_results]
        all_results = [mem_cache[mid] for mid in new_ids]
        self._gui_print(f"LTM DEBUG [3]: capped to {len(all_results)} results. IDs: {[mem_cache[mid].get('entry_id') or mem_cache[mid].get('memory_type','?') for mid in new_ids]}", "system")

        # Log the RRF ranking for diagnostics
        rrf_summary = ", ".join(
            f"{mem_cache[mid].get('entry_id') or mem_cache[mid].get('memory_type', '?')}={rrf_scores[mid]:.4f}"
            for mid in ranked_ids
        )
        self._gui_print(f"LTM RRF RANKING: {rrf_summary}", "system")

        # Follow chains
        pre_chain = len(all_results)
        if self._chain_depth > 0:
            all_results = self.follow_chains(all_results, depth=self._chain_depth)
        self._gui_print(f"LTM DEBUG [4]: chain follow {pre_chain} -> {len(all_results)} results", "system")

        # Filter out window-resident memories from chain-followed results too
        pre_filter = len(all_results)
        all_results = [r for r in all_results if r.get("id", "") not in window_ids]
        # Dedup chain-followed results against visible context
        if visible_context:
            pre_dedup = len(all_results)
            deduped_results = []
            for r in all_results:
                content = r.get("content", "")
                if content:
                    lines = content.splitlines()
                    core = "\n".join(lines[2:]) if len(lines) > 2 else content
                    if core and core in visible_context:
                        self._gui_print(f"LTM DEDUP [chain]: skipping {r.get('entry_id', r.get('id', '?')[:8])} (already in L2/L3)", "system")
                        continue
                deduped_results.append(r)
            all_results = deduped_results
        self._gui_print(f"LTM DEBUG [5]: post-chain window filter {pre_filter} -> {len(all_results)} new results", "system")

        # Format new blocks up to per-cycle injection budget
        max_tokens = self._inject_window_tokens
        new_blocks = []
        tokens_used = 0

        for r in all_results:
            entry_id = r.get("entry_id", "")
            title = r.get("title", "")
            memory_type = r.get("memory_type", "")
            mem_id = r.get("id", "")
            ctx_desc = r.get("amem_context_description", "")
            content = r.get("content", "")
            tags = r.get("tags", "[]")
            amem_links = r.get("amem_links", "[]")

            header = f"[{entry_id or memory_type.upper()}]"
            if title:
                header += f" {title}"

            block_parts = [header]
            block_parts.append(f"UUID: {mem_id}")
            rrf_s = r.get('_rrf_score')
            if rrf_s is not None:
                block_parts.append(f"LanceDB L2 Score: {r.get('_distance', 0):.4f} | RRF Score: {rrf_s:.4f} | RRF Rank: {r.get('_rrf_rank', '?')}/{r.get('_rrf_total', '?')}")
            else:
                if '_distance' in r:
                    block_parts.append(f"LanceDB L2 Score: {r.get('_distance', 0):.4f}")
                else:
                    block_parts.append(f"LanceDB L2 Score: N/A (chain-followed)")
            if ctx_desc:
                block_parts.append(f"Context: {ctx_desc}")
            try:
                tags_list = json.loads(tags)
                block_parts.append(f"Tags: {', '.join(str(t) for t in tags_list) if tags_list else '(none)'}")
            except (json.JSONDecodeError, TypeError):
                block_parts.append("Tags: (none)")
            block_parts.append(f"Content: {content}")

            try:
                links_list = json.loads(amem_links)
                if links_list:
                    link_strs = []
                    for link in links_list[:5]:
                        if isinstance(link, dict):
                            cid = link.get('candidate_id', '')
                            linked_mem = self.get_by_id(cid) if cid else None
                            if linked_mem:
                                link_strs.append(f"{linked_mem.get('entry_id') or linked_mem.get('memory_type', '?')} | {linked_mem.get('title', 'Untitled')} (id={cid})")
                            else:
                                link_strs.append(cid)
                    if link_strs:
                        block_parts.append(f"Linked to: {', '.join(link_strs)}")
            except (json.JSONDecodeError, TypeError):
                pass

            block = "\n".join(block_parts)
            block_tokens = self._count_tokens(block)

            if tokens_used + block_tokens > max_tokens:
                self._gui_print(f"LTM DEBUG [6]: BUDGET HIT at {tokens_used}+{block_tokens}>{max_tokens}, skipping {entry_id or memory_type}", "system")
                break

            new_blocks.append({"id": mem_id, "block": block, "tokens": block_tokens})
            tokens_used += block_tokens
            self._gui_print(f"LTM DEBUG [6]: formatted [{entry_id or memory_type}] {block_tokens} tokens (total {tokens_used}/{max_tokens})", "system")

        # Feed new blocks into sliding window and get full window content
        self._gui_print(f"LTM DEBUG: new_blocks={len(new_blocks)}, new_tokens={tokens_used}, window_entries={len(self._injection_window)}, window_tokens={self._injection_window_tokens}", "system")
        injection = self._update_injection_window(new_blocks)

        if not injection:
            self._current_injection = ""
            return ""

        self._current_injection = injection
        self._log_event("inject",
            f"Multi-query: {len(query_texts)} queries -> {len(new_blocks)} new, {len(self._injection_window)} in window ({self._injection_window_tokens} tokens)",
            injection)
        return injection

    # ── Explicit Tools ──────────────────────────────────────────────────────

    def soft_delete(self, memory_id, reason=None):
        """Soft-delete a memory. Keeps data but excludes from search.
        Returns True on success."""
        if not self._enabled:
            return False
        try:
            memory = self.get_by_id(memory_id)
            if not memory:
                return False
            if memory.get("is_deleted", False):
                return False
            with self._write_lock:
                safe_id = memory_id.replace("'", "''")
                self._table.update(
                    where=f"id = '{safe_id}'",
                    values={"is_deleted": True}
                )
                self._active_count = max(0, self._active_count - 1)
            self._gui_print(
                f"LTM: Memory {memory_id} soft-deleted. Reason: {reason or 'None'}",
                "system"
            )
            title = memory.get("title", "Untitled")
            self._log_event("delete", f"Deleted: {title} ({memory_id})", f"Reason: {reason or 'None'}")
            return True
        except Exception as e:
            self._gui_print(f"LTM: Soft delete failed: {e}", "error")
            return False

    # ── Tool API (called from spartan.py tool wrappers) ─────────────────────

    def tool_store(self, action):
        """API for store_memory tool. Takes action dict, returns result dict."""
        if not self._enabled:
            return {"status": "error", "result": "LTM is not available or not enabled."}

        content = action.get("content", "")
        title = action.get("title", "")
        tags = action.get("tags", [])

        if not content:
            return {"status": "error", "result": "'store_memory' requires 'content'."}

        memory_id = self.store_explicit(content, title=title, tags=tags)
        if memory_id:
            return {"status": "success", "result": f"Memory stored in LTM ({memory_id}). Title: '{title or 'Untitled'}'. Tags: {tags}. This memory will be available for retrieval and auto-injection."}
        return {"status": "error", "result": "Failed to store memory in LTM."}

    def tool_retrieve(self, action):
        """API for retrieve_memory tool. Takes action dict, returns result dict."""
        if not self._enabled:
            return {"status": "error", "result": "LTM is not available or not enabled."}

        query = action.get("query", "")
        if not query:
            return {"status": "error", "result": "'retrieve_memory' requires 'query'."}

        max_results = action.get("max_results", 5)
        tags_filter = action.get("tags_filter", None)

        results = self.search(query, max_results=max_results, tags_filter=tags_filter)
        if not results:
            return {"status": "success", "result": f"No memories found for query: '{query}'"}

        formatted = []
        for r in results:
            entry_id = r.get("entry_id") or r.get("memory_type", "?")
            title = r.get("title", "")
            content = r.get("content", "")
            ctx = r.get("amem_context_description", "")
            dist = r.get("_distance", 0)
            tags = r.get("tags", "[]")
            mid = r.get("id", "?")

            amem_links = r.get("amem_links", "[]")

            block = f"[{entry_id}] {title}"
            block += f"\n  UUID: {mid}"
            block += f"\n  LanceDB L2 Score: {dist:.4f}"
            if ctx:
                block += f"\n  Context: {ctx}"
            try:
                tags_list = json.loads(tags)
                block += f"\n  Tags: {', '.join(str(t) for t in tags_list) if tags_list else '(none)'}"
            except (json.JSONDecodeError, TypeError):
                block += "\n  Tags: (none)"
            block += f"\n  Content: {content}"
            try:
                links_list = json.loads(amem_links)
                if links_list:
                    link_strs = []
                    for link in links_list[:5]:
                        if isinstance(link, dict):
                            cid = link.get('candidate_id', '')
                            linked_mem = self.get_by_id(cid) if cid else None
                            if linked_mem:
                                link_strs.append(f"{linked_mem.get('entry_id') or linked_mem.get('memory_type', '?')} | {linked_mem.get('title', 'Untitled')} (id={cid})")
                            else:
                                link_strs.append(cid)
                    if link_strs:
                        block += f"\n  Linked to: {', '.join(link_strs)}"
            except (json.JSONDecodeError, TypeError):
                pass
            formatted.append(block)

        return {"status": "success", "result": f"LTM Query: '{query}' — {len(results)} results:\n\n" + "\n\n".join(formatted)}

    def tool_forget(self, action):
        """API for forget_memory tool. Takes action dict, returns result dict."""
        if not self._enabled:
            return {"status": "error", "result": "LTM is not available or not enabled."}

        memory_id = action.get("memory_id", "")
        reason = action.get("reason", "")

        if not memory_id:
            return {"status": "error", "result": "'forget_memory' requires 'memory_id'. Use 'retrieve_memory' first to find the ID."}

        success = self.soft_delete(memory_id, reason=reason)
        if success:
            return {"status": "success", "result": f"Memory {memory_id} soft-deleted from LTM. Reason: {reason or 'None'}. Data preserved but excluded from retrieval."}
        return {"status": "error", "result": f"Failed to delete memory {memory_id}. It may not exist."}

    # ── Maintenance ─────────────────────────────────────────────────────────

    def count(self):
        """Total non-deleted memories. O(1) from in-memory counter."""
        if not self._enabled or self._table is None:
            return 0
        return self._active_count

    def count_from_db(self):
        """Full DB scan for active memories. Use for diagnostics or counter verification."""
        if not self._enabled or self._table is None:
            return 0
        try:
            total = self._table.count_rows()
            if total == 0:
                return 0
            active = len(self._table.search().where("is_deleted = false").limit(total).to_list())
            return active
        except Exception:
            try:
                return self._table.count_rows()
            except Exception:
                return 0

    def reindex_all(self):
        """Re-embed all memories with the current embedding model.
        Expensive but necessary when changing embedding models."""
        if not self._enabled:
            return

        self._gui_print("LTM: Starting full reindex...", "system")
        try:
            row_count = self._table.count_rows()
            all_memories = self._table.search().where("is_deleted = false").limit(row_count).to_list()
            reindexed = 0
            failed = 0

            for memory in all_memories:
                content = memory.get("content", "")
                if not content:
                    continue

                vector = self._embedding_engine.embed(content, is_query=False)
                if vector is None:
                    failed += 1
                    continue

                try:
                    safe_id = memory['id'].replace("'", "''")
                    self._table.update(
                        where=f"id = '{safe_id}'",
                        values={"vector": vector}
                    )
                    reindexed += 1
                except Exception:
                    failed += 1

            # Update model info
            info_path = os.path.join(self._db_path, MODEL_INFO_FILE)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({"model_identifier": self._embedding_engine.model_identifier}, f)

            self._gui_print(
                f"LTM: Reindex complete. {reindexed} re-embedded, {failed} failed.",
                "system"
            )
        except Exception as e:
            self._gui_print(f"LTM: Reindex failed: {e}", "error")

    def backfill_soul_files(self, soul_dir, soul_file_map):
        """Backfill existing Soul file entries into the LTM database.
        Run on first boot when LTM is enabled and the DB is empty.
        Args:
            soul_dir: path to Soul/ directory
            soul_file_map: dict mapping target names to filenames"""
        if not self._enabled or self.count() > 0:
            return

        self._gui_print("LTM: Backfilling existing Soul files...", "system")
        total = 0

        # Process mapped Soul files
        for target_name, filename in soul_file_map.items():
            filepath = os.path.join(soul_dir, filename)
            entries = _parse_existing_soul_file(filepath)
            for entry in entries:
                embed_text = ""
                if entry["title"]:
                    embed_text = f"{entry['title']}\n\n"
                embed_text += entry["body"]

                full_json = {
                    "type": "soul_entry",
                    "source_file": entry["source_file"],
                    "entry_id": entry["entry_id"],
                    "title": entry["title"],
                    "tags": entry["tags"],
                    "related_entries": entry["related_entries"],
                    "content": entry["body"],
                }

                memory_id = self._store_memory(
                    memory_type="soul_entry",
                    content_text=embed_text,
                    full_json_data=full_json,
                    source_file=entry["source_file"],
                    entry_id=entry["entry_id"],
                    title=entry["title"],
                    tags=entry["tags"],
                    related_entries=entry["related_entries"],
                    amem_processed=False,
                )
                if memory_id:
                    total += 1

        # Process Charter of Self
        charter_path = os.path.join(soul_dir, "CharterOfSelf.md")
        entries = _parse_existing_soul_file(charter_path)
        for entry in entries:
            embed_text = entry["body"]
            full_json = {
                "type": "soul_entry",
                "source_file": "CharterOfSelf.md",
                "entry_id": entry["entry_id"],
                "title": entry["title"],
                "content": entry["body"],
            }
            memory_id = self._store_memory(
                memory_type="soul_entry",
                content_text=embed_text,
                full_json_data=full_json,
                source_file="CharterOfSelf.md",
                entry_id=entry["entry_id"],
                title=entry["title"],
                decay_exempt=True,
                amem_processed=False,
            )
            if memory_id:
                total += 1

        self._gui_print(f"LTM: Backfill complete. {total} entries indexed.", "system")

    def sync_at_startup(self, soul_dir, soul_file_map, conversation_history):
        """Incremental sync: index any Soul entries and CMOs not already in the DB.
        Fast path: queries existing entry_ids/timestamps first, only embeds missing.
        A-Mem deferred (amem_processed=False) -- processed on first Sleep Cycle.
        Args:
            soul_dir: path to Soul/ directory
            soul_file_map: dict mapping target names to filenames
            conversation_history: deque/list from session state
        Returns: (soul_count, cmo_count) of newly indexed items."""
        if not self._enabled:
            return 0, 0

        # Collect existing entry_ids and CMO timestamps from DB
        try:
            row_count = self._table.count_rows()
            if row_count == 0:
                existing_entry_ids = set()
                existing_cmo_timestamps = set()
            else:
                all_rows = self._table.search().where("is_deleted = false").limit(row_count).to_list()
                existing_entry_ids = {r.get("entry_id", "") for r in all_rows if r.get("entry_id")}
                existing_cmo_timestamps = set()
                for r in all_rows:
                    if r.get("memory_type") == "cmo":
                        # Row timestamp is write-time, not CMO creation time.
                        # Extract original timestamp from full_json.
                        try:
                            fj = json.loads(r.get("full_json", "{}"))
                            ts = fj.get("timestamp", "")
                            if ts:
                                existing_cmo_timestamps.add(ts)
                        except (json.JSONDecodeError, TypeError):
                            pass
        except Exception as e:
            self._gui_print(f"LTM sync: Failed to query existing entries: {e}", "error")
            return 0, 0

        soul_count = 0
        cmo_count = 0

        # Phase 1: Soul files
        for target_name, filename in soul_file_map.items():
            filepath = os.path.join(soul_dir, filename)
            if not os.path.exists(filepath):
                continue
            entries = _parse_existing_soul_file(filepath)
            for entry in entries:
                if entry["entry_id"] and entry["entry_id"] in existing_entry_ids:
                    continue
                embed_text = ""
                if entry["title"]:
                    embed_text = f"{entry['title']}\n\n"
                embed_text += entry["body"]

                full_json = {
                    "type": "soul_entry",
                    "source_file": entry["source_file"],
                    "entry_id": entry["entry_id"],
                    "title": entry["title"],
                    "tags": entry["tags"],
                    "related_entries": entry["related_entries"],
                    "content": entry["body"],
                }
                memory_id = self._store_memory(
                    memory_type="soul_entry",
                    content_text=embed_text,
                    full_json_data=full_json,
                    source_file=entry["source_file"],
                    entry_id=entry["entry_id"],
                    title=entry["title"],
                    tags=entry["tags"],
                    related_entries=entry["related_entries"],
                    decay_exempt=False,
                    amem_processed=False,
                )
                if memory_id:
                    soul_count += 1
                    self._gui_print(f"LTM sync: Indexed {entry['entry_id'] or entry['source_file']} ({memory_id})", "system")

        # Charter of Self
        charter_path = os.path.join(soul_dir, "CharterOfSelf.md")
        if os.path.exists(charter_path):
            entries = _parse_existing_soul_file(charter_path)
            for entry in entries:
                if entry["entry_id"] and entry["entry_id"] in existing_entry_ids:
                    continue
                embed_text = entry["body"]
                if entry["title"]:
                    embed_text = f"{entry['title']}\n\n{embed_text}"
                full_json = {
                    "type": "soul_entry",
                    "source_file": "CharterOfSelf.md",
                    "entry_id": entry["entry_id"],
                    "title": entry["title"],
                    "content": entry["body"],
                }
                memory_id = self._store_memory(
                    memory_type="soul_entry",
                    content_text=embed_text,
                    full_json_data=full_json,
                    source_file="CharterOfSelf.md",
                    entry_id=entry["entry_id"],
                    title=entry["title"],
                    decay_exempt=True,
                    amem_processed=False,
                )
                if memory_id:
                    soul_count += 1
                    self._gui_print(f"LTM sync: Indexed Charter {entry['entry_id']} ({memory_id})", "system")

        # Phase 2: CMOs from conversation history
        for item in conversation_history:
            obs_type = item.get("observation_type", "")
            msg = item.get("message", "")
            timestamp = item.get("timestamp", "")

            if obs_type != "system_message":
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("object_type") != "CondensedMemoryObject":
                continue

            content = msg.get("content", "")
            if not content or not content.strip():
                continue
            if content.startswith("[SYSTEM:") and "archived" in content and "Routine" in content:
                continue

            if timestamp and timestamp in existing_cmo_timestamps:
                continue

            memory_id = self.store_cmo(content, cmo_timestamp=timestamp)
            if memory_id:
                cmo_count += 1
                self._gui_print(f"LTM sync: Indexed CMO {timestamp[:10]} ({memory_id})", "system")

        if soul_count or cmo_count:
            self._gui_print(
                f"LTM sync complete: {soul_count} Soul entries, {cmo_count} CMOs newly indexed.",
                "system"
            )
        else:
            self._gui_print("LTM sync: Everything already indexed.", "system")

        return soul_count, cmo_count


    def defragment_db(self):
        """Defragment LanceDB storage: merge fragments and remove old versions."""
        if not self._enabled or self._db is None:
            return
        try:
            for table_name in self._db.table_names():
                table = self._db.open_table(table_name)
                count_before = table.count_rows()
                sample_before = table.search().limit(5).to_list()
                titles_before = [s.get('title', 'no title')[:80] for s in sample_before]
                self._gui_print(f"BEFORE defragmentation - {table_name}: {count_before} rows", "system")
                for t in titles_before:
                    self._gui_print(f"  - {t}", "system")
                file_count_before = sum(1 for _ in os.scandir(self._db_path) if _.is_file()) if os.path.isdir(self._db_path) else 0
                self._gui_print(f"\nDefragmenting {table_name}...", "system")
                table.compact_files()
                table.cleanup_old_versions()
                count_after = table.count_rows()
                sample_after = table.search().limit(5).to_list()
                titles_after = [s.get('title', 'no title')[:80] for s in sample_after]
                self._gui_print(f"\nAFTER defragmentation - {table_name}: {count_after} rows", "system")
                for t in titles_after:
                    self._gui_print(f"  - {t}", "system")
                if count_before == count_after:
                    self._gui_print(f"\n✓ SAFE: Row count unchanged ({count_before})", "system")
                else:
                    self._gui_print(f"\n✗ WARNING: Row count changed! {count_before} -> {count_after}", "error")
                if titles_before == titles_after:
                    self._gui_print(f"✓ SAFE: Sample titles match", "system")
                else:
                    self._gui_print(f"✗ WARNING: Sample titles differ", "error")
                file_count_after = sum(1 for root, dirs, files in os.walk(self._db_path) for f in files)
                self._gui_print(f"\nTotal files in ltm_db: {file_count_after}", "system")
        except Exception as e:
            self._gui_print(f"LTM: Defragmentation failed: {e}", "error")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LTM VIEWER (Tkinter Toplevel Window)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LTMViewer:
    """Separate Tkinter window for monitoring LTM activity.
    Attaches to an existing Tk root (from spartan.py's GUI).
    Polls the LTM's event queue and displays activity, injection content, and errors.

    Usage from spartan.py:
        if ltm_instance and ltm_instance.visualization and tk:
            ltm_viewer = LTMViewer(root, ltm_instance)
    """

    POLL_MS = 1000  # Poll event queue every 200ms

    # Color scheme for event types
    TAG_COLORS = {
        "store": "#50fa7b",     # green
        "search": "#8be9fd",    # cyan
        "inject": "#4a9eff",    # blue
        "amem": "#ffb86c",      # orange
        "delete": "#f1fa8c",    # yellow
        "error": "#ff5555",     # red
        "info": "#6272a4",      # gray
    }

    def __init__(self, tk_root, ltm_instance):
        """Create the LTM Viewer as a Toplevel window.
        Args:
            tk_root: the existing Tk root from spartan.py's GUI
            ltm_instance: the LongTermMemory instance to monitor
        """
        if tk is None:
            return

        self._ltm = ltm_instance
        self._window = tk.Toplevel(tk_root)
        self._window.title("LTM Viewer -- Long-Term Memory Monitor")
        self._window.geometry("900x600")
        self._window.configure(bg="#1a1a2e")

        # Don't kill the main app when this window closes
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._poll()

    def _build(self):
        mono = tkfont.Font(family="Menlo", size=10) if sys.platform == "darwin" else tkfont.Font(family="Consolas", size=9)
        mono_small = tkfont.Font(family="Menlo", size=9) if sys.platform == "darwin" else tkfont.Font(family="Consolas", size=8)

        # ── Top Bar: Stats ──────────────────────────────────────────────
        top_bar = tk.Frame(self._window, bg="#0a0a1a")
        top_bar.pack(fill=tk.X, padx=5, pady=(5, 0))

        self._stats_label = tk.Label(
            top_bar, text="Initializing...", font=mono_small,
            bg="#0a0a1a", fg="#4a9eff", anchor="w"
        )
        self._stats_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── Main Area: Activity Log (left) + Injection Pane (right) ─────
        main_frame = tk.PanedWindow(
            self._window, orient=tk.HORIZONTAL, bg="#1a1a2e",
            sashwidth=4, sashrelief=tk.RAISED
        )
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: Activity Log
        left_frame = tk.Frame(main_frame, bg="#0a0a1a")
        main_frame.add(left_frame, minsize=350)

        tk.Label(left_frame, text="ACTIVITY LOG", font=mono_small,
                 bg="#0a0a1a", fg="#4a9eff").pack(pady=(2, 0))

        self._activity_log = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, font=mono_small,
            bg="#0f0f23", fg="#cccccc", insertbackground="#ffffff",
            state=tk.DISABLED
        )
        self._activity_log.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Configure color tags
        for tag_name, color in self.TAG_COLORS.items():
            self._activity_log.tag_configure(tag_name, foreground=color)
        self._activity_log.tag_configure("timestamp", foreground="#6272a4")

        # Right: Current Injection
        right_frame = tk.Frame(main_frame, bg="#0a0a1a")
        main_frame.add(right_frame, minsize=300)

        tk.Label(right_frame, text="CURRENT L4 INJECTION", font=mono_small,
                 bg="#0a0a1a", fg="#4a9eff").pack(pady=(2, 0))

        self._injection_pane = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, font=mono_small,
            bg="#0f0f23", fg="#8be9fd", insertbackground="#ffffff",
            state=tk.DISABLED
        )
        self._injection_pane.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # ── Bottom: Query Test ──────────────────────────────────────────
        bottom_frame = tk.Frame(self._window, bg="#16213e")
        bottom_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(bottom_frame, text="Test Query:", font=mono_small,
                 bg="#16213e", fg="#888888").pack(side=tk.LEFT, padx=(5, 2))

        self._query_entry = tk.Entry(
            bottom_frame, font=mono, bg="#1a1a2e", fg="#ffffff",
            insertbackground="#ffffff"
        )
        self._query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        self._query_entry.bind("<Return>", self._on_query)

        query_btn = tk.Button(
            bottom_frame, text="Search", font=mono_small, width=8,
            bg="#16213e", fg="#50fa7b", activebackground="#3a3a5c",
            activeforeground="#ffffff",
            command=self._on_query
        )
        query_btn.pack(side=tk.RIGHT, padx=(2, 5), pady=5)

    def _poll(self):
        """Poll the LTM event queue and update displays."""
        try:
            while True:
                event = self._ltm.event_queue.get_nowait()
                self._append_event(event)
        except Exception:
            pass

        # Update stats
        try:
            count = self._ltm.count()
            self._stats_label.configure(
                text=f"Memories: {count} | DB: {self._ltm._db_path} | "
                     f"Auto-inject: {'ON' if self._ltm._auto_inject else 'OFF'} | "
                     f"Decay: {self._ltm._temporal_decay_monthly}"
            )
        except Exception:
            pass

        # Update injection pane if it changed
        try:
            current = self._ltm.current_injection
            self._injection_pane.configure(state=tk.NORMAL)
            existing = self._injection_pane.get("1.0", tk.END).strip()
            if current != existing:
                self._injection_pane.delete("1.0", tk.END)
                if current:
                    self._injection_pane.insert(tk.END, current)
                else:
                    self._injection_pane.insert(tk.END, "[No injection this cycle]")
            self._injection_pane.configure(state=tk.DISABLED)
        except Exception:
            pass

        # Schedule next poll
        try:
            self._window.after(self.POLL_MS, self._poll)
        except Exception:
            pass

    def _append_event(self, event):
        """Append an event to the activity log with color coding."""
        ts = event.get("timestamp", "??:??:??")
        etype = event.get("type", "info")
        message = event.get("message", "")
        details = event.get("details", "")

        tag = etype if etype in self.TAG_COLORS else "info"

        self._activity_log.configure(state=tk.NORMAL)
        self._activity_log.insert(tk.END, f"[{ts}] ", "timestamp")
        self._activity_log.insert(tk.END, f"{message}\n", tag)
        if details:
            # Indent details and show in dimmer color
            for line in details.splitlines():
                self._activity_log.insert(tk.END, f"    {line}\n", "info")
        self._activity_log.see(tk.END)
        self._activity_log.configure(state=tk.DISABLED)

    def _on_query(self, event=None):
        """Handle manual query from the test field."""
        query_text = self._query_entry.get().strip()
        if not query_text:
            return

        self._query_entry.delete(0, tk.END)

        # Log the manual query
        self._append_event({
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": "search",
            "message": f"[MANUAL] Query: '{query_text}'",
            "details": "",
        })

        # Execute search
        results = self._ltm.search(query_text, max_results=10)
        if not results:
            self._append_event({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "type": "info",
                "message": "[MANUAL] No results.",
                "details": "",
            })
            return

        # Display results in activity log
        result_lines = []
        for r in results:
            eid = r.get("entry_id") or r.get("memory_type", "?")
            title = r.get("title", "")
            dist = r.get("_distance", 0)
            content_preview = r.get("content", "")[:200]
            result_lines.append(f"[{eid}] {title} (dist={dist:.4f})")
            result_lines.append(f"    {content_preview}")
        self._append_event({
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": "search",
            "message": f"[MANUAL] {len(results)} results",
            "details": "\n".join(result_lines),
        })

    def _on_close(self):
        """Hide the window instead of destroying it (can be reopened)."""
        self._window.withdraw()

    def show(self):
        """Show the window if it was hidden."""
        self._window.deiconify()

    def destroy(self):
        """Fully destroy the window."""
        try:
            self._window.destroy()
        except Exception:
            pass
