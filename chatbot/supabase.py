from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any, Callable


def _import_third_party_supabase_create_client():
    """Import `supabase.create_client` from the pip package safely.

    This repo contains `chatbot/supabase.py`. In environments where the working
    directory (or script directory) is `chatbot/`, `import supabase` would
    resolve to this local file and shadow the third-party package.
    """

    here = os.path.dirname(os.path.abspath(__file__))
    here_norm = os.path.normcase(os.path.abspath(here))
    removed: list[str] = []
    popped_module = None

    try:
        # If this local file was imported as top-level module name `supabase`,
        # it will live in sys.modules['supabase'] and will *always* win over
        # any sys.path manipulation. Temporarily pop it so we can import the
        # third-party pip package.
        try:
            existing = sys.modules.get('supabase')
            existing_file = os.path.normcase(os.path.abspath(getattr(existing, '__file__', '') or '')) if existing is not None else ''
            if existing is not None and existing_file == os.path.normcase(os.path.abspath(__file__)):
                popped_module = sys.modules.pop('supabase', None)
        except Exception:
            popped_module = None

        # Remove the local package dir from sys.path temporarily so that
        # `import_module('supabase')` resolves to the installed pip package.
        for p in list(sys.path):
            try:
                if os.path.normcase(os.path.abspath(p)) == here_norm:
                    sys.path.remove(p)
                    removed.append(p)
            except Exception:
                continue

        supabase_pkg = import_module('supabase')
        create_client = getattr(supabase_pkg, 'create_client', None)
        if create_client is None:
            raise ImportError('supabase.create_client not found (package shadowing or wrong version)')
        return create_client
    finally:
        # Restore removed paths (front) to keep other local imports intact.
        if removed:
            sys.path[:0] = removed

        if popped_module is not None:
            sys.modules['supabase'] = popped_module


def _is_missing_or_placeholder(value: str | None) -> bool:
    text = str(value or '').strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {'none', 'null', 'changeme', 'your_key_here', 'your_url_here', 'your_supabase_key'}:
        return True
    if lowered.startswith('your_'):
        return True
    return False


@lru_cache(maxsize=1)
def _cached_create_supabase_client(url: str, key: str) -> Any:
    create_client = _import_third_party_supabase_create_client()
    return create_client(url, key)


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    disabled: bool = False

    @staticmethod
    def from_env(
        *,
        url_env: str = 'SUPABASE_URL',
        key_env: str = 'SUPABASE_KEY',
        service_role_key_env: str = 'SUPABASE_SERVICE_ROLE_KEY',
        disabled_env: str = 'SUPABASE_DISABLED',
    ) -> 'SupabaseConfig':
        disabled = str(os.getenv(disabled_env, '')).strip().lower() in {'1', 'true', 'yes', 'on'}
        url = str(os.getenv(url_env, '') or '')
        key = str(os.getenv(key_env, '') or os.getenv(service_role_key_env, '') or '')
        return SupabaseConfig(url=url, key=key, disabled=disabled)

    def is_configured(self) -> bool:
        if self.disabled:
            return False
        if _is_missing_or_placeholder(self.url) or _is_missing_or_placeholder(self.key):
            return False
        return True


class SupabaseClientFactory:
    def __init__(self, *, config: SupabaseConfig | None = None) -> None:
        self._config = config

    def config(self) -> SupabaseConfig:
        return self._config or SupabaseConfig.from_env()

    def get_client(self) -> Any | None:
        cfg = self.config()
        if not cfg.is_configured():
            return None
        try:
            return _cached_create_supabase_client(cfg.url, cfg.key)
        except Exception:
            return None


@lru_cache(maxsize=1)
def _cached_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LyricsEmbeddingProvider:
    def __init__(
        self,
        *,
        model_env: str = 'SUPABASE_LYRICS_EMBEDDING_MODEL',
        default_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    ) -> None:
        self.model_env = str(model_env)
        self.default_model = str(default_model)

    def model_name(self) -> str:
        name = str(os.getenv(self.model_env, '') or '').strip()
        return name or self.default_model

    def encode(self, text: str) -> list[float] | None:
        text = str(text or '').strip()
        if not text:
            return None
        try:
            model = _cached_sentence_transformer(self.model_name())
            vec = model.encode(text)
            return vec.tolist()  # type: ignore[no-any-return]
        except Exception:
            return None


def encode_lyrics_embedding_debug(text: str) -> tuple[list[float] | None, str | None]:
    """Encode text into the same embedding space as the lyrics vectors.

    Unlike LyricsEmbeddingProvider.encode(), this returns an error string
    instead of swallowing exceptions, so UI/tests can show the real cause.
    """

    text = str(text or '').strip()
    if not text:
        return None, 'empty-text'
    try:
        provider = LyricsEmbeddingProvider()
        model = _cached_sentence_transformer(provider.model_name())
        vec = model.encode(text)
        out = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        if not out:
            return None, 'empty-vector'
        return [float(x) for x in out], None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


class SupabaseLyricsSearchService:
    def __init__(
        self,
        *,
        client_factory: SupabaseClientFactory | None = None,
        embedding_provider: LyricsEmbeddingProvider | None = None,
        rpc_env: str = 'SUPABASE_LYRICS_RPC',
        default_rpc: str = 'match_lyrics',
    ) -> None:
        self.client_factory = client_factory or SupabaseClientFactory()
        self.embedding_provider = embedding_provider or LyricsEmbeddingProvider()
        self.rpc_env = str(rpc_env)
        self.default_rpc = str(default_rpc)

    def rpc_name(self) -> str:
        return str(os.getenv(self.rpc_env, '') or '').strip() or self.default_rpc

    def query(self, user_input: str, *, match_threshold: float = 0.4, match_count: int = 5) -> list[dict]:
        text = str(user_input or '').strip()
        if not text:
            return []

        client = self.client_factory.get_client()
        if client is None:
            return []

        query_embedding = self.embedding_provider.encode(text)
        if not query_embedding:
            return []

        payload = {
            'query_embedding': query_embedding,
            'match_threshold': float(match_threshold),
            'match_count': int(match_count),
        }

        try:
            response = client.rpc(self.rpc_name(), payload).execute()
            data = getattr(response, 'data', None) or []
            return data if isinstance(data, list) else []
        except Exception:
            return []


class SupabaseVectorSearchService:
    def __init__(
        self,
        *,
        client_factory: SupabaseClientFactory | None = None,
        embedding_provider: LyricsEmbeddingProvider | None = None,
        rpc_env: str = 'SUPABASE_VECTOR_RPC',
        default_rpc: str = 'match_vpop_tracks',
        threshold_env: str = 'SUPABASE_VECTOR_THRESHOLD',
        default_threshold: float = 0.4,
    ) -> None:
        self.client_factory = client_factory or SupabaseClientFactory()
        self.embedding_provider = embedding_provider or LyricsEmbeddingProvider()
        self.rpc_env = str(rpc_env)
        self.default_rpc = str(default_rpc)
        self.threshold_env = str(threshold_env)
        self.default_threshold = float(default_threshold)

    def rpc_name(self) -> str:
        return str(os.getenv(self.rpc_env, '') or '').strip() or self.default_rpc

    def threshold(self) -> float:
        try:
            return float(os.getenv(self.threshold_env, str(self.default_threshold)))
        except Exception:
            return float(self.default_threshold)

    def query(
        self,
        intent_json: dict,
        *,
        build_fallback: Callable[[dict], list[dict]],
        normalize_rows: Callable[[list[dict]], list[dict]],
        enrich_intent: Callable[[dict], dict],
    ) -> dict:
        cfg = self.client_factory.config()
        if not cfg.is_configured():
            return {
                'tracks': build_fallback(intent_json),
                'source': 'fallback-no-env',
                'error': 'Supabase chưa được cấu hình hoặc đang bị tắt (SUPABASE_URL/SUPABASE_KEY/SUPABASE_DISABLED).',
            }

        client = self.client_factory.get_client()
        if client is None:
            return {
                'tracks': build_fallback(intent_json),
                'source': 'fallback-client-init-failed',
                'error': 'Không khởi tạo được Supabase client (thiếu package, import shadowing, hoặc key/url không hợp lệ).',
            }
        try:
            enriched = enrich_intent(intent_json)
        except Exception:
            enriched = dict(intent_json or {})

        # Your RPC `match_vpop_tracks` requires an actual `vector` embedding.
        # Build the embedding from query_text unless caller explicitly provides one.
        q_text = str(enriched.get('query_text', '') or '').strip()
        if not q_text:
            mood = str(enriched.get('mood', '') or '').strip()
            keywords = enriched.get('keywords', []) or []
            kw_text = ' '.join([str(k).strip() for k in keywords if str(k).strip()])
            q_text = ' '.join([mood, kw_text]).strip()

        embedding = None
        if isinstance(intent_json, dict) and intent_json.get('query_embedding') is not None:
            embedding = intent_json.get('query_embedding')
        if not embedding:
            embedding = self.embedding_provider.encode(q_text)

        if not embedding:
            return {
                'tracks': build_fallback(intent_json),
                'source': 'fallback-missing-embedding',
                'error': (
                    'Không tạo được embedding để gọi Supabase RPC. '
                    'Kiểm tra `sentence-transformers`/`torch` trong môi trường đang chạy, '
                    'và đảm bảo query_text không rỗng.'
                ),
            }

        payload: dict[str, Any] = {
            'query_embedding': embedding,
            'match_threshold': self.threshold(),
            'match_count': int(enriched.get('top_k', 5) or 5),
        }

        response = None
        try:
            response = client.rpc(self.rpc_name(), payload).execute()
        except Exception as ex:
            return {
                'tracks': build_fallback(intent_json),
                'source': 'fallback-rpc-error',
                'error': f"RPC {self.rpc_name()} failed ({type(ex).__name__}: {ex})",
            }

        rows = getattr(response, 'data', None) or []
        if not isinstance(rows, list):
            rows = []

        normalized = normalize_rows([r for r in rows if isinstance(r, dict)])
        if not normalized:
            return {
                'tracks': build_fallback(intent_json),
                'source': 'fallback-empty-live',
                'error': f"RPC {self.rpc_name()} trả về rỗng (không có dòng phù hợp).",
            }

        top_k = int((intent_json or {}).get('top_k', 5) or 5)
        return {
            'tracks': normalized[: max(1, top_k)],
            'source': f'live-supabase-rpc:{self.rpc_name()}',
            'error': None,
        }


_DEFAULT_CLIENT_FACTORY = SupabaseClientFactory()
_DEFAULT_LYRICS_SERVICE = SupabaseLyricsSearchService(client_factory=_DEFAULT_CLIENT_FACTORY)
_DEFAULT_VECTOR_SERVICE = SupabaseVectorSearchService(client_factory=_DEFAULT_CLIENT_FACTORY)


def get_supabase_client() -> Any | None:
    return _DEFAULT_CLIENT_FACTORY.get_client()


def query_supabase_lyrics(user_input: str, match_threshold: float = 0.4, match_count: int = 5) -> list[dict]:
    return _DEFAULT_LYRICS_SERVICE.query(user_input, match_threshold=match_threshold, match_count=match_count)


def query_supabase_vector(
    intent_json: dict,
    *,
    build_fallback: Callable[[dict], list[dict]],
    normalize_rows: Callable[[list[dict]], list[dict]],
    enrich_intent: Callable[[dict], dict],
) -> dict:
    return _DEFAULT_VECTOR_SERVICE.query(
        intent_json,
        build_fallback=build_fallback,
        normalize_rows=normalize_rows,
        enrich_intent=enrich_intent,
    )


class SupabaseChatHistoryService:
    def __init__(
        self,
        *,
        client_factory: SupabaseClientFactory | None = None,
        table_env: str = 'SUPABASE_CHAT_HISTORY_TABLE',
        default_table: str = 'chat_history',
        session_id_col: str = 'session_id',
        alt_session_id_col: str = 'sesion_id',
    ) -> None:
        self.client_factory = client_factory or SupabaseClientFactory()
        self.table_env = str(table_env)
        self.default_table = str(default_table)
        self.session_id_col = str(session_id_col)
        self.alt_session_id_col = str(alt_session_id_col)

    def table_name(self) -> str:
        return str(os.getenv(self.table_env, '') or '').strip() or self.default_table

    def _client(self) -> Any | None:
        return self.client_factory.get_client()

    def append(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        module: str,
    ) -> bool:
        session_id = str(session_id or '').strip()
        role = str(role or '').strip() or 'user'
        content = str(content or '').strip()
        module = str(module or '').strip() or 'default'
        if not session_id or not content:
            return False

        client = self._client()
        if client is None:
            return False

        payload = {
            self.session_id_col: session_id,
            'role': role,
            'content': content,
            'module': module,
        }

        try:
            client.table(self.table_name()).insert(payload).execute()
            return True
        except Exception:
            # Tolerate legacy schemas with typo column name.
            try:
                payload2 = dict(payload)
                payload2.pop(self.session_id_col, None)
                payload2[self.alt_session_id_col] = session_id
                client.table(self.table_name()).insert(payload2).execute()
                return True
            except Exception:
                return False

    def fetch(
        self,
        *,
        session_id: str,
        module: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        session_id = str(session_id or '').strip()
        if not session_id:
            return []

        client = self._client()
        if client is None:
            return []

        try:
            q = client.table(self.table_name()).select('*').eq(self.session_id_col, session_id)
            if module is not None:
                q = q.eq('module', str(module))
            q = q.order('created_at', desc=False).limit(max(1, int(limit)))
            resp = q.execute()
            data = getattr(resp, 'data', None) or []
            return data if isinstance(data, list) else []
        except Exception:
            # Try legacy typo column name.
            try:
                q = client.table(self.table_name()).select('*').eq(self.alt_session_id_col, session_id)
                if module is not None:
                    q = q.eq('module', str(module))
                q = q.order('created_at', desc=False).limit(max(1, int(limit)))
                resp = q.execute()
                data = getattr(resp, 'data', None) or []
                return data if isinstance(data, list) else []
            except Exception:
                return []

    def fetch_recent(
        self,
        *,
        session_id: str,
        module: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        rows = self.fetch(session_id=session_id, module=module, limit=max(1, int(limit)))
        # `fetch` is ascending, so take last N.
        return rows[-max(1, int(limit)) :]


_DEFAULT_CHAT_HISTORY_SERVICE = SupabaseChatHistoryService(client_factory=_DEFAULT_CLIENT_FACTORY)


def append_chat_history(*, session_id: str, role: str, content: str, module: str) -> bool:
    return _DEFAULT_CHAT_HISTORY_SERVICE.append(
        session_id=session_id,
        role=role,
        content=content,
        module=module,
    )


def fetch_chat_history(*, session_id: str, module: str | None = None, limit: int = 200) -> list[dict]:
    return _DEFAULT_CHAT_HISTORY_SERVICE.fetch(session_id=session_id, module=module, limit=limit)


def fetch_recent_chat_history(*, session_id: str, module: str | None = None, limit: int = 5) -> list[dict]:
    return _DEFAULT_CHAT_HISTORY_SERVICE.fetch_recent(session_id=session_id, module=module, limit=limit)
