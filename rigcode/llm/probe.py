"""Detect what kind of OpenAI-compatible server sits behind a base URL and what
it supports (context size, slots, template), so the client can tune requests."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import httpx


@dataclass
class ServerInfo:
    kind: str = "unknown"          # llama.cpp | vllm | sglang | ollama | lmstudio | unknown
    reachable: bool = False
    error: str | None = None
    models: list[str] = field(default_factory=list)
    context_window: int | None = None
    total_slots: int | None = None
    chat_template: str | None = None
    build: str | None = None
    vision: bool | None = None
    effort_levels: tuple[str, ...] | None = None    # reasoning_effort values the chat template accepts
    loaded_models: list[str] = field(default_factory=list)       # Ollama / LM Studio: models in memory now
    model_contexts: dict[str, int] = field(default_factory=dict)  # Ollama: runtime context per loaded model
    raw: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.kind]
        if self.build:
            bits.append(self.build)
        if self.context_window:
            bits.append(f"ctx {self.context_window:,}")
        if self.total_slots:
            bits.append(f"{self.total_slots} slot{'s' if self.total_slots != 1 else ''}")
        return " · ".join(bits)


def server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    for suffix in ("/v1", "/api/v1", "/openai/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


async def _get_json(client: httpx.AsyncClient, url: str) -> tuple[int | None, object | None]:
    try:
        r = await client.get(url)
    except httpx.HTTPError:
        return None, None
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


_EFFORT_LIST_RE = re.compile(r"reasoning_effort\s+in\s+\[([^\]]*)\]")


def template_effort_levels(template: str | None) -> tuple[str, ...] | None:
    """Every reasoning_effort value a Jinja chat template checks for, if it lists any."""
    levels: list[str] = []
    for m in _EFFORT_LIST_RE.finditer(template or ""):
        for v in re.findall(r"['\"]([a-z]+)['\"]", m.group(1)):
            if v not in levels:
                levels.append(v)
    return tuple(levels) or None


def _first_int(*vals) -> int | None:
    for v in vals:
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            continue
    return None


async def probe_server(base_url: str, api_key: str | None = None, timeout: float = 8.0) -> ServerInfo:
    info = ServerInfo()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    root = server_root(base_url)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), headers=headers) as client:
        try:
            r = await client.get(f"{base_url.rstrip('/')}/models")
            info.reachable = True
            models_json = r.json() if r.status_code == 200 else None
        except httpx.ConnectTimeout:
            info.error = "connect timeout (host reachable? firewall blocking the port?)"
            return info
        except httpx.ConnectError as e:
            info.error = f"connection refused/failed: {e}"
            return info
        except httpx.HTTPError as e:
            info.error = f"{type(e).__name__}: {e}"
            return info
        except ValueError:
            models_json = None

        if isinstance(models_json, dict):
            for m in models_json.get("data") or []:
                if not isinstance(m, dict):
                    continue
                if m.get("id"):
                    info.models.append(str(m["id"]))
                meta = m.get("meta") or {}
                info.context_window = info.context_window or _first_int(
                    m.get("max_model_len"), m.get("context_length"), m.get("max_context_length"),
                    meta.get("n_ctx"), meta.get("n_ctx_train"))
                owner = str(m.get("owned_by", "")).lower()
                if owner == "vllm":
                    info.kind = "vllm"
                elif owner == "llamacpp":
                    info.kind = "llama.cpp"
                elif owner in ("sglang",):
                    info.kind = "sglang"

        # llama.cpp exposes /props (n_ctx is the *runtime* context, better than n_ctx_train).
        props_status, props, slots_status, slots = None, None, None, None
        (props_status, props), (slots_status, slots) = await asyncio.gather(
            _get_json(client, f"{root}/props"), _get_json(client, f"{root}/slots"))
        if props_status == 200 and isinstance(props, dict) and (
                "default_generation_settings" in props or "total_slots" in props or "build_info" in props):
            info.kind = "llama.cpp"
            info.raw["props"] = {k: v for k, v in props.items() if k != "chat_template"}
            dgs = props.get("default_generation_settings") or {}
            n_ctx = _first_int(dgs.get("n_ctx"), props.get("n_ctx"))
            if n_ctx:
                info.context_window = n_ctx
            info.total_slots = _first_int(props.get("total_slots"))
            info.chat_template = props.get("chat_template")
            info.effort_levels = template_effort_levels(info.chat_template)
            info.build = props.get("build_info")
            mods = props.get("modalities") or {}
            if isinstance(mods, dict):
                info.vision = bool(mods.get("vision"))
            if slots_status == 200 and isinstance(slots, list) and slots:
                info.total_slots = info.total_slots or len(slots)
                if not n_ctx:
                    info.context_window = _first_int(slots[0].get("n_ctx")) or info.context_window
            return info

        if info.kind == "unknown":
            st, data = await _get_json(client, f"{root}/get_model_info")
            if st == 200 and isinstance(data, dict) and "model_path" in data:
                info.kind = "sglang"
        if info.kind in ("unknown", "sglang"):
            st, data = await _get_json(client, f"{root}/get_server_info")
            if st == 200 and isinstance(data, dict):
                info.kind = "sglang"
                info.context_window = info.context_window or _first_int(data.get("context_length"))
        if info.kind == "unknown":
            st, data = await _get_json(client, f"{root}/api/version")
            if st == 200 and isinstance(data, dict) and "version" in data:
                info.kind = "ollama"
                info.build = str(data.get("version"))
                st, ps = await _get_json(client, f"{root}/api/ps")
                for m in (ps.get("models") or []) if isinstance(ps, dict) else []:
                    name = m.get("name") or m.get("model")
                    if name:
                        info.loaded_models.append(str(name))
                        ctx = _first_int(m.get("context_length"))
                        if ctx:
                            info.model_contexts[str(name)] = ctx
        if info.kind == "unknown":
            st, data = await _get_json(client, f"{root}/version")
            if st == 200 and isinstance(data, dict) and "version" in data:
                info.kind = "vllm"
                info.build = str(data.get("version"))
        if info.kind == "unknown":
            st, data = await _get_json(client, f"{root}/api/v0/models")
            if st == 200 and isinstance(data, dict) and "data" in data:
                info.kind = "lmstudio"
                for m in data.get("data") or []:
                    if m.get("state") == "loaded" and m.get("id"):
                        info.loaded_models.append(str(m["id"]))
                        info.model_contexts[str(m["id"])] = _first_int(
                            m.get("loaded_context_length"), m.get("max_context_length")) or 0
                    info.context_window = info.context_window or _first_int(m.get("max_context_length"))
    return info


AUTO = (None, "", "auto")
LOCAL_SERVERS = (
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("vLLM/SGLang", "http://127.0.0.1:8000/v1"),
)


async def discover(provider: dict, timeout: float = 8.0) -> tuple[ServerInfo, str | None]:
    """Probe the configured server, or find a local one when base_url is 'auto', and pick the model
    when model is 'auto'. Fills in provider['base_url'] / provider['model'] in place.

    Returns the server info and a one-line notice about what was chosen (or why nothing could be)."""
    key = provider.get("api_key")
    notice = None
    if provider.get("base_url") in AUTO:
        infos = await asyncio.gather(*(probe_server(url, key, timeout=2.0) for _, url in LOCAL_SERVERS))
        # A server counts only if it lists models: dev web servers often sit on 8000/8080 too.
        found = [(name, url, i) for (name, url), i in zip(LOCAL_SERVERS, infos) if i.reachable and i.models]
        if not found:
            ports = ", ".join(f"{url.split(':')[2].split('/')[0]} ({name})" for name, url in LOCAL_SERVERS)
            return ServerInfo(error=f"no local model server found on ports {ports}. Start one, or point "
                                    "rigcode at yours: rigcode --base-url http://HOST:PORT/v1"), None
        name, provider["base_url"], info = found[0]
        notice = f"Found {name} at {provider['base_url']}"
    else:
        info = await probe_server(str(provider["base_url"]), key, timeout=timeout)
    if not info.reachable:
        return info, notice
    if provider.get("model") in AUTO:
        candidates = info.loaded_models or info.models
        if not candidates:
            info.reachable, info.error = False, "the server lists no models; set one with --model NAME"
            return info, notice
        provider["model"] = candidates[0]
        others = [m for m in info.models if m != candidates[0]]
        picked = f"using model {candidates[0]}"
        if others:
            picked += f" (also available: {', '.join(others[:5])}{' …' if len(others) > 5 else ''}; pick with -m)"
        notice = f"{notice}, {picked}" if notice else picked[0].upper() + picked[1:]
    ctx = info.model_contexts.get(str(provider.get("model")))
    if ctx:
        info.context_window = ctx
    return info, notice
