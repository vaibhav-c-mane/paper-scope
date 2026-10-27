// PaperScope front end: one search page, a trace of how each search worked, and a paper drawer.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const FEATURE_NAMES = {
  bm25: "word match (BM25)", semantic: "meaning (embeddings)", title_coverage: "query words in title",
  term_coverage: "query words anywhere", domain_match: "same research field", citations: "citations per year",
  recency: "recency", peer_reviewed: "peer reviewed", venue_quality: "venue quality",
};
let current = { page: 1, exact: false };

async function getJSON(url) {
  const r = await fetch(url);
  const data = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

// ------------------------------------------------------------------ state
async function refreshState() {
  try {
    const s = await getJSON("/api/state");
    $("state").textContent = `${s.papers_in_memory.toLocaleString()} / ${s.capacity.toLocaleString()} papers in memory · ` +
      `ranker: ${s.ranker.kind}${s.ranker.trees ? ` (${s.ranker.trees} trees)` : ""} · ` + (s.online ? "online" : "offline");
    if ($("field").options.length === 1) {
      for (const f of s.fields) $("field").insertAdjacentHTML("beforeend", `<option>${esc(f)}</option>`);
    }
  } catch (e) { $("state").textContent = "server not reachable"; }
}

// ------------------------------------------------------------------ search
function params(page) {
  const p = new URLSearchParams({ q: $("q").value.trim(), page });
  for (const id of ["year_from", "year_to", "author", "field", "sort", "online", "mode"]) {
    if ($(id).value) p.set(id, $(id).value);
  }
  if ($("strict").checked) p.set("strict", "1");
  if (current.exact) p.set("exact", "1");
  return p;
}

async function search(page = 1, exact = false) {
  const q = $("q").value.trim();
  if (!q) return;
  if (page === 1) current.exact = exact;          // "search instead for ..." asks for the words as typed
  hideSuggest();
  current.page = page;
  $("results").innerHTML = `<div class="empty">Searching… (the first search on a topic asks the internet)</div>`;
  try {
    const res = await getJSON("/api/search?" + params(page));
    renderResults(res);
    renderTrace(res.trace);
  } catch (e) {
    $("results").innerHTML = `<div class="empty err">${esc(e.message)}</div>`;
    $("trace").classList.add("hidden");
  }
  refreshState();
}

function highlight(text, words) {
  let out = esc(text);
  for (const w of words || []) {
    const stem = w.replace(/\*$/, "");
    if (stem.length < 3) continue;
    out = out.replace(new RegExp(`\\b(${stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\w*)`, "gi"), "<mark>$1</mark>");
  }
  return out;
}

function chipsFor(p) {
  const c = [`<span class="chip">${esc(p.field)}</span>`];
  if (p.published === true) c.push(`<span class="chip good">peer reviewed</span>`);
  else if (p.published === false) c.push(`<span class="chip warn">preprint</span>`);
  if (p.venue_quality >= 0.75) c.push(`<span class="chip good">${p.venue_quality >= 1 ? "top" : "strong"} venue</span>`);
  if (p.citations != null) c.push(`<span class="chip">${p.citations.toLocaleString()} citations</span>`);
  c.push(`<span class="chip">${esc(p.source)}</span>`);
  return c.join("");
}

function renderResults(res) {
  window.lastResults = res.results;              // the drawer shows these papers' ranking features
  const words = res.trace.query?.words || [];
  const box = $("results");
  if (!res.results.length) {
    box.innerHTML = understoodHTML(res.trace.understood) + `<div class="empty">No papers found. Try other words, or remove filters.</div>`;
    return;
  }
  const first = (res.page - 1) * res.per_page;
  box.innerHTML = understoodHTML(res.trace.understood) +
    `<div class="summary">${res.total} papers · ${res.trace.total_ms} ms</div>` +
    res.results.map((p, i) => `
      <article class="paper" data-key="${esc(p.key)}">
        <h3>${first + i + 1}. ${highlight(p.title, words)}</h3>
        <div class="meta">${esc(p.year || "?")} · ${esc((p.authors || "").slice(0, 110))}${p.venue ? " · <i>" + esc(p.venue) + "</i>" : ""}</div>
        <div class="abs">${highlight(p.abstract || "(no abstract)", words)}</div>
        ${p.missing && p.missing.length ? `<div class="missing">Missing: ${p.missing.map((w) => `<s>${esc(w)}</s>`).join(" ")}</div>` : ""}
        <div class="chips">${chipsFor(p)}${matchChip(p)}</div>
      </article>`).join("") +
    `<div class="pager">${res.page > 1 ? `<button data-page="${res.page - 1}">‹ Previous</button>` : ""}` +
    `${first + res.per_page < res.total ? `<button data-page="${res.page + 1}">Next ›</button>` : ""}</div>`;
}

// What query understanding changed (engine/understand.py), with a way back to the words as typed.
function understoodHTML(u) {
  if (!u || !u.changed) return "";
  const notes = [];
  if (u.corrections.length) notes.push("spelling corrected: " + u.corrections.map((c) => `${esc(c.from)} → <b>${esc(c.to)}</b>`).join(", "));
  if (u.recent_from) notes.push(`only papers from ${u.recent_from} on, because you asked for recent work`);
  return `<div class="understood">Showing results for <b>${esc(u.query)}</b>` +
    (notes.length ? `<div class="small">${notes.join(" · ")}</div>` : "") +
    `<div class="small">Search instead for <a href="#" id="exact">${esc(u.original)}</a></div></div>`;
}

function renderTrace(t) {
  const el = $("trace");
  el.classList.remove("hidden");
  const rows = [];
  const u = t.understood;
  if (u) {
    const how = [u.kind === "question" ? "a question" : u.kind === "request" ? "a request" : "keywords"];
    if (u.semantic) how.push("also search by meaning");
    if (u.title) how.push("also search titles");
    rows.push(`<h4>Understood as</h4><div>${how.map(esc).join(" · ")}` +
      (u.removed.length ? `<br>left out: <code>${esc(u.removed.join(" "))}</code>` : "") +
      (u.corrections.length ? `<br>spelling: ${u.corrections.map((c) => `<code>${esc(c.from)}</code> → <code>${esc(c.to)}</code>`).join(", ")}` : "") +
      (u.recent_from ? `<br>recent: from ${u.recent_from}` : "") +
      (u.relaxed ? `<br>also searched with fewer words: <code>${esc(u.relaxed)}</code> (without ${u.relax_dropped.map((w) => `<code>${esc(w)}</code>`).join(", ")})` : "") + "</div>");
  }
  rows.push(`<h4>Query</h4><div>words: <code>${esc((t.query?.words || []).join(" ") || "–")}</code>` +
    (t.query?.excluded?.length ? `<br>excluded: <code>${esc(t.query.excluded.join(" "))}</code>` : "") + "</div>");
  if (t.identifier) {
    rows.push(`<h4>Identifier</h4><div>${esc(t.identifier.kind)} <code>${esc(t.identifier.value)}</code>: ` +
      `${t.identifier.in_memory ? "already in memory" : "looked up online"}</div>`);
    const errs = t.identifier.online?.errors || [];
    if (errs.length) rows.push(`<div class="err small">${errs.map(esc).join("<br>")}</div>`);
  }
  if (t.boolean && t.boolean.error) {
    rows.push(`<h4>AND / OR / NOT</h4><div class="err">${esc(t.boolean.error)}</div>`);
  } else if (t.boolean) {
    rows.push(`<h4>AND / OR / NOT on a stack</h4><div>postfix: <code>${esc(t.boolean.postfix.join(" "))}</code></div>` +
      `<table>${t.boolean.steps.map((s) => `<tr><td><code>${esc(s.token)}</code></td><td><div class="stack">` +
        `${s.stack.map((x) => `<span>${esc(x)}</span>`).join("")}</div></td></tr>`).join("")}</table>` +
      `<div class="small muted">${t.boolean.matches} papers in memory match</div>`);
  }
  if (t.online) {
    const o = t.online;
    rows.push(`<h4>Internet</h4><div>${o.used ? `asked because ${esc(o.reason)}` : `not needed: ${esc(o.reason)}`}</div>`);
    if (o.used) {
      rows.push(`<table>${o.providers.map((p) => `<tr><td>${esc(p.provider)}</td><td class="${p.ok ? "ok" : "err"}">` +
        `${p.ok ? `${p.results} papers${p.cached ? " (cached)" : ""}` : esc(p.error || "failed")}</td><td>${p.ms ?? ""} ms</td></tr>`).join("")}</table>` +
        `<div class="small muted">${o.fetched} fetched, ${o.new} new to memory, ${o.ms} ms</div>`);
    }
  }
  if (t.candidates) {
    const c = t.candidates;
    rows.push(`<h4>Candidates</h4><table>` +
      `<tr><td>inverted index (BM25)</td><td>${c.lexical}</td><td>${c.ms.lexical} ms</td></tr>` +
      `<tr><td>vector search</td><td>${c.semantic}</td><td>${c.ms.semantic} ms</td></tr>` +
      `<tr><td>from the internet</td><td>${c.from_online}</td><td></td></tr>` +
      `<tr><td><b>re-ranked</b></td><td><b>${c.total}</b></td><td></td></tr></table>` +
      `<div class="small muted">${c.in_memory.toLocaleString()} papers in memory, ${c.allowed_by_filters.toLocaleString()} pass the filters</div>`);
  }
  if (t.field) {
    rows.push(`<h4>Research field</h4><div>${t.field.chosen ? `chosen: <b>${esc(t.field.chosen)}</b><br>` : ""}` +
      t.field.detected.map((d) => `${esc(d.name)} ${(d.p * 100).toFixed(0)}%`).join("<br>") + "</div>");
  }
  if (t.rerank) rows.push(`<h4>Ranking</h4><div>${esc(t.rerank.ranker)} · confident above ${t.rerank.threshold}` +
    (t.fusion ? `<br>combined with the order of ${esc(t.fusion.with)} (reciprocal rank fusion)` : "") +
    (t.exact_title ? `<br>exact title match moved to the top` : "") +
    (t.sorted_on_topic != null ? `<br>${t.sort === "year" ? "newest" : "most cited"} first among the ${t.sorted_on_topic} papers that match best, then the rest by relevance` : "") + "</div>");
  el.innerHTML = rows.join("");
}

function matchChip(p) {
  // topic match: share of the query covered in words and meaning (engine/pipeline.py, match_scores)
  const pct = Math.round(p.match * 100);
  return `<span class="chip match m-${esc(p.match_level)}" title="Covers about ${pct}% of your search, in words and meaning. Citations and year are not counted here.">` +
    `${esc(p.match_level)} match <span class="bar"><i style="width:${pct}%"></i></span> ${pct}%</span>`;
}

// ------------------------------------------------------------------ paper drawer
function relatedHTML(p) {
  // Dijkstra's shortest paths over citation, related-work and similarity links (dsa/graph.py)
  const LINK = { cites: "cited by this paper", "cited by": "cites this paper", related: "listed as related by OpenAlex", similar: "similar topic" };
  const rel = (p.related || []).map((r) =>
    `<li><a href="#" data-key="${esc(r.key)}">${esc(r.title)}</a> <span class="small muted">${esc(r.year || "?")} · ` +
    `${r.via ? `via “${esc(r.via)}”` : esc(LINK[r.link] || r.link)} · closeness ${Math.round(r.closeness * 100)}%</span></li>`).join("");
  // breadth-first search over citation links only (dsa/graph.py)
  const how = (r) => r.hops === 1 ? (r.relation === "reference" ? "cited by this paper" : "cites this paper")
                                   : `2 links away, via “${esc(r.via)}”`;
  const cit = (p.citation_links || []).map((r) =>
    `<li><a href="#" data-key="${esc(r.key)}">${esc(r.title)}</a> <span class="small muted">${esc(r.year || "?")} · ${how(r)}</span></li>`).join("");
  return `<h4>Related papers <span class="small muted">closest first, by citations, OpenAlex and meaning</span></h4>` +
    (rel ? `<ul class="related">${rel}</ul>` : `<p class="small muted">No related papers in memory yet. Search its topic to bring more in.</p>`) +
    `<h4>Citation links <span class="small muted">papers it cites or that cite it</span></h4>` +
    (cit ? `<ul class="related">${cit}</ul>` : `<p class="small muted">${p.citation_data
      ? "None of the papers it cites or that cite it are in memory yet."
      : "No citation data for this paper (only OpenAlex lists references)."}</p>`);
}

async function openPaper(key) {
  $("drawer").classList.remove("hidden");
  $("drawer-body").innerHTML = `<div class="empty">Loading…</div>`;
  let feats = null;
  try {
    const p = await getJSON("/api/paper?key=" + encodeURIComponent(key));
    const shown = (window.lastResults || []).find((r) => r.key === key);
    feats = shown ? shown.features : null;
    const link = p.doi ? `https://doi.org/${p.doi}` : p.url;
    $("drawer-body").innerHTML = `
      <h2>${esc(p.title)}</h2>
      <div class="meta">${esc(p.year || "?")}${p.venue ? " · <i>" + esc(p.venue) + "</i>" : ""}</div>
      <p class="small">${esc(p.authors)}</p>
      <div class="chips">${chipsFor({ ...p, field: shown ? shown.field : (p.topic || "").split(" | ").pop() || "?" })}</div>
      <p>${esc(p.abstract || "(no abstract)")}</p>
      ${p.topic ? `<p class="small muted">Topic: ${esc(p.topic)}</p>` : ""}
      <p class="small">${link ? `<a href="${esc(link)}" target="_blank" rel="noopener">Publisher page</a>` : ""}
        ${p.oa_url ? ` · <a href="${esc(p.oa_url)}" target="_blank" rel="noopener">Free full text</a>` : ""}</p>
      ${relatedHTML(p)}
      ${feats ? `<h4>Why it ranked here</h4><table class="feat">${Object.keys(FEATURE_NAMES).filter((k) => k in feats).map((k) => [k, feats[k]]).map(([k, v]) =>
        `<tr><td>${esc(FEATURE_NAMES[k] || k)}</td><td>${v < 0 ? "unknown" : v.toFixed(2)}</td></tr>`).join("")}</table>` : ""}`;
  } catch (e) {
    $("drawer-body").innerHTML = `<div class="empty err">${esc(e.message)}</div>`;
  }
}

// ------------------------------------------------------------------ suggestions
let sugTimer = null, sugIndex = -1, sugSeq = 0;
function hideSuggest() { clearTimeout(sugTimer); sugSeq++; $("suggest").classList.add("hidden"); sugIndex = -1; }
function currentWord() { const parts = $("q").value.split(/\s+/); return parts[parts.length - 1]; }
async function suggest() {
  const w = currentWord();
  if ((w.length < 2 || /^[A-Z]+$/.test(w)) && $("q").value.trim().length < 3) return hideSuggest();
  const seq = ++sugSeq;
  const { suggestions, titles = [] } = await getJSON("/api/suggest?prefix=" + encodeURIComponent(w) + "&q=" + encodeURIComponent($("q").value))
    .catch(() => ({ suggestions: [] }));
  if (seq !== sugSeq || document.activeElement !== $("q")) return;   // a search started meanwhile
  if (!suggestions.length && !titles.length) return hideSuggest();
  // words (binary search in the vocabulary), then paper titles that start with the text (radix trie)
  $("suggest").innerHTML = suggestions.slice(0, titles.length ? 4 : 8).map((s) => `<li>${esc(s)}</li>`).join("") +
    titles.map((t) => `<li class="title" data-key="${esc(t.key)}">${esc(t.title)} <span class="small muted">${esc(t.year || "")}</span></li>`).join("");
  $("suggest").classList.remove("hidden");
}
function pickItem(li) {
  if (li.dataset.key) { hideSuggest(); openPaper(li.dataset.key); return; }
  pick(li.textContent);
}
function pick(word) {
  const parts = $("q").value.split(/(\s+)/);
  parts[parts.length - 1] = word;
  $("q").value = parts.join("") + " ";
  hideSuggest();
  $("q").focus();
}

// ------------------------------------------------------------------ events
$("search").addEventListener("submit", (e) => { e.preventDefault(); search(1); });
$("q").addEventListener("input", () => { clearTimeout(sugTimer); sugTimer = setTimeout(suggest, 150); });
$("q").addEventListener("keydown", (e) => {
  const items = [...$("suggest").querySelectorAll("li")];
  if ($("suggest").classList.contains("hidden") || !items.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    sugIndex = (sugIndex + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length;
    items.forEach((li, i) => li.classList.toggle("on", i === sugIndex));
  } else if (e.key === "Tab" || (e.key === "Enter" && sugIndex >= 0)) {
    e.preventDefault();
    pickItem(items[Math.max(0, sugIndex)]);
  } else if (e.key === "Escape") hideSuggest();
});
$("suggest").addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pickItem(li); } });
$("q").addEventListener("blur", () => setTimeout(hideSuggest, 100));
$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (!b) return;
  $("q").value = b.dataset.q;
  $("field").value = b.dataset.field || "";
  search(1);
});
$("results").addEventListener("click", (e) => {
  if (e.target.id === "exact") { e.preventDefault(); search(1, true); return; }
  const pg = e.target.closest("button[data-page]");
  if (pg) { search(Number(pg.dataset.page), current.exact); window.scrollTo(0, 0); return; }
  const card = e.target.closest(".paper");
  if (card) openPaper(card.dataset.key);
});
$("drawer-body").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-key]");
  if (a) { e.preventDefault(); openPaper(a.dataset.key); }
});
$("drawer-close").addEventListener("click", () => $("drawer").classList.add("hidden"));
$("drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") $("drawer").classList.add("hidden"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("drawer").classList.add("hidden"); });
refreshState();
