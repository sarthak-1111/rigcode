"""Web tools: WebFetch (URL -> markdown) and WebSearch."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx

from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.utils.text import one_line

MAX_DOWNLOAD = 5 * 1024 * 1024
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "canvas", "head", "form", "button", "select"}
BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "table", "tr",
              "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "details", "summary", "address", "hr"}


class _HTMLToMarkdown(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.lists: list[list] = []      # stack of [type, counter]
        self.link: str | None = None
        self.link_text: list[str] = []
        self.title = ""
        self._in_title = False
        self.cell = 0

    def _emit(self, text: str) -> None:
        if self.link is not None:
            self.link_text.append(text)
        else:
            self.out.append(text)

    def _nl(self, n: int = 1) -> None:
        if self.link is None:
            self.out.append("\n" * n)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._nl(2)
            self._emit("#" * int(tag[1]) + " ")
        elif tag == "br":
            self._emit("\n")
        elif tag == "pre":
            self.pre += 1
            self._nl(2)
            self._emit("```\n")
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "blockquote":
            self._nl(2)
            self._emit("> ")
        elif tag in ("ul", "ol"):
            self.lists.append([tag, 0])
            self._nl()
        elif tag == "li":
            self._nl()
            depth = max(0, len(self.lists) - 1)
            if self.lists and self.lists[-1][0] == "ol":
                self.lists[-1][1] += 1
                self._emit("  " * depth + f"{self.lists[-1][1]}. ")
            else:
                self._emit("  " * depth + "- ")
        elif tag == "a":
            href = a.get("href") or ""
            if href and not href.startswith(("javascript:", "#", "mailto:")):
                self.link = urljoin(self.base, href)
                self.link_text = []
        elif tag == "img":
            alt = (a.get("alt") or "").strip()
            if alt:
                self._emit(f"[image: {alt}]")
        elif tag in ("td", "th"):
            self._emit(" | " if self.cell else "| ")
            self.cell += 1
        elif tag in BLOCK_TAGS:
            self._nl(2 if tag in ("p", "table") else 1)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._nl(2)
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self._emit("\n```")
            self._nl(2)
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._nl()
        elif tag == "a" and self.link is not None:
            text = " ".join("".join(self.link_text).split())
            link, self.link = self.link, None
            if text:
                self.out.append(f"[{text}]({link})")
        elif tag == "tr":
            if self.cell:
                self.out.append(" |")
            self.cell = 0
            self._nl()
        elif tag in BLOCK_TAGS:
            self._nl(2 if tag in ("p", "table") else 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self.skip:
            return
        if self.pre:
            self._emit(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if text.strip() or (text == " " and self.out and not self.out[-1].endswith((" ", "\n"))):
                self._emit(text)

    def markdown(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _main_html(doc: str) -> str:
    for tag in ("main", "article"):
        m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", doc, re.S | re.I)
        if m and len(re.sub(r"<[^>]+>", "", m.group(1))) > 1500:
            return m.group(1)
    m = re.search(r"<body\b[^>]*>(.*)</body>", doc, re.S | re.I)
    return m.group(1) if m else doc


def html_to_markdown(doc: str, base_url: str = "") -> tuple[str, str]:
    title_m = re.search(r"<title[^>]*>(.*?)</title>", doc, re.S | re.I)
    title = html.unescape(title_m.group(1)).strip() if title_m else ""
    parser = _HTMLToMarkdown(base_url)
    try:
        parser.feed(_main_html(doc))
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML; return what we have
        pass
    return title, parser.markdown()


def select_relevant(text: str, query: str, budget: int) -> str:
    """Pick the sections most relevant to `query` (keyword overlap), in page order."""
    if len(text) <= budget:
        return text
    words = {w for w in re.findall(r"[a-zA-Z0-9_]{3,}", (query or "").lower())}
    sections = re.split(r"\n(?=#{1,6} )|\n\n", text)
    if not words:
        return text[:budget] + "\n\n[... page truncated ...]"
    scored = []
    for i, sec in enumerate(sections):
        low = sec.lower()
        score = sum(low.count(w) for w in words) / (1 + len(sec) / 2000)
        scored.append((score, i, sec))
    keep, used = set(), 0
    for score, i, sec in sorted(scored, key=lambda t: -t[0]):
        if score <= 0 and keep:
            break
        if used + len(sec) > budget:
            continue
        keep.add(i)
        used += len(sec) + 2
    head = sections[0][:600]
    out = [head] if 0 not in keep else []
    out += [sec for i, sec in enumerate(sections) if i in keep]
    return "\n\n".join(out) + f"\n\n[Showing the {len(keep)} sections most relevant to the prompt; page truncated.]"


class WebFetchTool(Tool):
    name = "WebFetch"
    kind = "web"
    read_only = True
    description = (
        "Fetch a URL and return its content as Markdown (HTML is converted; JSON/text returned as-is). Pass "
        "prompt to say what you are looking for; long pages are cut down to the most relevant sections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full http(s) URL"},
            "prompt": {"type": "string", "description": "What information you need from the page"},
        },
        "required": ["url"],
    }
    aliases = {"link": "url", "uri": "url", "query": "prompt", "question": "prompt"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("url", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = str(args["url"]).strip()
        if not re.match(r"^https?://", url):
            url = "https://" + url
        ua = ctx.rt.settings.get("web.user_agent") or "Mozilla/5.0 (compatible; rigcode)"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30, headers={
                    "User-Agent": ua, "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*"}) as c:
                async with c.stream("GET", url) as r:
                    chunks, size = [], 0
                    async for chunk in r.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > MAX_DOWNLOAD:
                            break
                    status, ctype, final_url = r.status_code, r.headers.get("content-type", ""), str(r.url)
                    encoding = r.encoding or "utf-8"
        except httpx.HTTPError as e:
            return ToolResult.error(f"Fetch failed for {url}: {type(e).__name__}: {e}")
        data = b"".join(chunks)
        if status >= 400:
            return ToolResult.error(f"HTTP {status} fetching {final_url}")
        budget = int(ctx.rt.settings.get("context.max_tool_output_chars", 12000))
        if "pdf" in ctype:
            return ToolResult.error(f"{final_url} is a PDF ({len(data):,} bytes); download it with Bash (curl -o) and Read it.")
        try:
            text = data.decode(encoding, errors="replace")
        except LookupError:
            text = data.decode("utf-8", errors="replace")
        title = ""
        if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            title, text = html_to_markdown(text, final_url)
        text = select_relevant(text, str(args.get("prompt") or ""), budget)
        header = f"# {title}\n" if title else ""
        redirect = f"(redirected to {final_url})\n" if final_url.rstrip("/") != url.rstrip("/") else ""
        return ToolResult(content=f"{header}{redirect}Source: {final_url}\n\n{text}",
                          summary=f"Fetched {len(text):,} chars from {urlparse(final_url).netloc}")


class WebSearchTool(Tool):
    name = "WebSearch"
    kind = "web"
    read_only = True
    description = (
        "Search the web. Returns titles, URLs and snippets; follow up with WebFetch for details. Use it for "
        "current information, documentation and error messages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Number of results (default 8)"},
        },
        "required": ["query"],
    }
    aliases = {"q": "query", "search": "query", "num_results": "max_results", "count": "max_results"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("query", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = str(args["query"]).strip()
        n = max(1, min(int(args.get("max_results") or 8), 20))
        s = ctx.rt.settings
        provider = (s.get("web.search_provider") or "auto").lower()
        try:
            if provider in ("searxng",) or (provider == "auto" and s.get("web.searxng_url")):
                results = await self._searxng(s.get("web.searxng_url"), query, n)
            elif provider == "brave" or (provider == "auto" and s.get("web.brave_api_key")):
                results = await self._brave(s.get("web.brave_api_key"), query, n)
            elif provider == "tavily" or (provider == "auto" and s.get("web.tavily_api_key")):
                results = await self._tavily(s.get("web.tavily_api_key"), query, n)
            else:
                results = await self._duckduckgo(query, n, s.get("web.user_agent"))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            return ToolResult.error(f"Search failed: {type(e).__name__}: {e}")
        if not results:
            return ToolResult(content=f"No results for: {query}", summary="No results")
        lines = []
        for i, (title, url, snippet) in enumerate(results[:n], 1):
            lines.append(f"{i}. {one_line(title, 120)}\n   {url}\n   {one_line(snippet, 300)}")
        return ToolResult(content="\n".join(lines), summary=f"{len(results[:n])} results")

    async def _duckduckgo(self, query: str, n: int, ua: str | None) -> list[tuple[str, str, str]]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                     headers={"User-Agent": ua or "Mozilla/5.0"}) as c:
            r = await c.post("https://html.duckduckgo.com/html/", data={"q": query})
            r.raise_for_status()
            page = r.text
        results = []
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                             page, re.S):
            href, title_html, rest = m.group(1), m.group(2), m.group(3)
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(html.unescape(href)).query).get("uddg", [href])[0])
            if "duckduckgo.com/y.js" in href:
                continue  # ads
            snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
            clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()  # noqa: E731
            results.append((clean(title_html), href, clean(snip.group(1) if snip else "")))
            if len(results) >= n:
                break
        return results

    async def _searxng(self, base: str, query: str, n: int) -> list[tuple[str, str, str]]:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{base.rstrip('/')}/search?q={quote_plus(query)}&format=json")
            r.raise_for_status()
            data = r.json()
        return [(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in data.get("results", [])[:n]]

    async def _brave(self, key: str, query: str, n: int) -> list[tuple[str, str, str]]:
        async with httpx.AsyncClient(timeout=20, headers={"X-Subscription-Token": key, "Accept": "application/json"}) as c:
            r = await c.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n})
            r.raise_for_status()
            data = r.json()
        return [(x.get("title", ""), x.get("url", ""), x.get("description", ""))
                for x in (data.get("web") or {}).get("results", [])[:n]]

    async def _tavily(self, key: str, query: str, n: int) -> list[tuple[str, str, str]]:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "max_results": n})
            r.raise_for_status()
            data = r.json()
        return [(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in data.get("results", [])[:n]]
