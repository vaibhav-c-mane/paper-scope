// PaperScope front end: a home page, the results list, a paper sheet, this tab's search history,
// light and dark themes, and a "How this search worked" sheet with the engine's trace.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const FILTERS = ["year_from", "year_to", "field", "sort", "author"];
const FEATURE_NAMES = {
  bm25: "Word match (BM25)", semantic: "Meaning (embeddings)", title_coverage: "Your words in the title",
  term_coverage: "Your words anywhere", domain_match: "Same research field", citations: "Citations per year",
  recency: "Recency", peer_reviewed: "Peer reviewed", venue_quality: "Venue quality",
};
const state = { page: 1, exact: false, trace: null, results: [], fieldsLoaded: false, lastFocus: null };

async function getJSON(url) {
  const r = await fetch(url);
  const data = await r.json().catch(() => ({ error: `The server answered with HTTP ${r.status}.` }));
  if (!r.ok) throw new Error(data.error || `The server answered with HTTP ${r.status}.`);
  return data;
}

// Browser storage can be missing or blocked (private windows); everything works without it.
const store = {
  get(area, key, fallback) { try { const v = window[area].getItem(key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(area, key, value) { try { window[area].setItem(key, JSON.stringify(value)); } catch (e) { /* not saved */ } },
  remove(area, key) { try { window[area].removeItem(key); } catch (e) { /* nothing to remove */ } },
};

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 2200);
}

// ------------------------------------------------------------------ theme
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
function currentTheme() { return document.documentElement.dataset.theme || (darkQuery.matches ? "dark" : "light"); }
function paintThemeButton() {
  const dark = currentTheme() === "dark";
  $("theme-btn").querySelector("use").setAttribute("href", dark ? "#i-sun" : "#i-moon");
  $("theme-btn").title = dark ? "Switch to light mode" : "Switch to dark mode";
}
$("theme-btn").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("paperscope.theme", next); } catch (e) { /* this visit only */ }
  paintThemeButton();
});
darkQuery.addEventListener("change", paintThemeButton);

// ------------------------------------------------------------------ home or results
function showHome(home) {
  document.body.classList.toggle("is-home", home);
  $("home").classList.toggle("hidden", !home);
  $("results-view").classList.toggle("hidden", home);
  if (home) $("home-slot").appendChild($("search"));
  else document.querySelector(".bar").insertBefore($("search"), document.querySelector(".bar-actions"));
  if (home) renderHomeHistory();
}

// ------------------------------------------------------------------ search history (this tab only)
const HISTORY_KEY = "paperscope.history";
function pastSearches() { return store.get("sessionStorage", HISTORY_KEY, []); }
function filtersNow() {
  const f = {};
  for (const id of FILTERS) if ($(id).value && !(id === "sort" && $(id).value === "relevance")) f[id] = $(id).value;
  if ($("strict").checked) f.strict = "1";
  return f;
}
function remember(q, total) {
  const f = filtersNow();
  const same = (h) => h.q.toLowerCase() === q.toLowerCase() && JSON.stringify(h.f) === JSON.stringify(f);
  const list = [{ q, f, total, at: Date.now() }, ...pastSearches().filter((h) => !same(h))].slice(0, 20);
  store.set("sessionStorage", HISTORY_KEY, list);
}
function ago(t) {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}
function filterWords(f) {
  const out = [];
  if (f.year_from || f.year_to) out.push(f.year_from && f.year_to ? `${f.year_from} to ${f.year_to}` : f.year_from ? `from ${f.year_from}` : `until ${f.year_to}`);
  if (f.field) out.push(f.field);
  if (f.author) out.push(`by ${f.author}`);
  if (f.sort) out.push(f.sort === "year" ? "newest first" : "most cited");
  return out.join(", ");
}
function historyHTML(limit) {
  const list = pastSearches().slice(0, limit);
  if (!list.length) return `<p class="history-empty">Your searches in this tab will show here.</p>`;
  return `<ul class="history-list">${list.map((h, i) => `<li><button type="button" class="history-item" data-h="${i}">
    <span class="h-q">${esc(h.q)}</span>
    <span class="h-meta">${[filterWords(h.f), `${h.total.toLocaleString()} papers`, ago(h.at)].filter(Boolean).map(esc).join(", ")}</span>
  </button></li>`).join("")}</ul>`;
}
function renderHomeHistory() {
  if (!pastSearches().length) { $("home-history").innerHTML = ""; return; }
  $("home-history").innerHTML = `<div class="section-head"><h2>Recent searches</h2>
    <button type="button" class="link small" data-clear-history>Clear history</button></div>${historyHTML(8)}`;
}
function renderHistoryPop() {
  $("history-pop").innerHTML = `<div class="section-head"><h2>Recent searches</h2>
    ${pastSearches().length ? `<button type="button" class="link small" data-clear-history>Clear history</button>` : ""}</div>
    ${historyHTML(20)}<p class="popover-note">Kept only in this browser tab, with no account. Closing the tab clears it.</p>`;
}
function runHistory(i) {
  const h = pastSearches()[i];
  if (!h) return;
  closePopover();
  $("q").value = h.q;
  for (const id of FILTERS) $(id).value = h.f[id] || (id === "sort" ? "relevance" : "");
  $("strict").checked = h.f.strict === "1";
  search(1);
}
document.addEventListener("click", (e) => {
  const item = e.target.closest("[data-h]");
  if (item) { runHistory(Number(item.dataset.h)); return; }
  if (e.target.closest("[data-clear-history]")) {
    store.remove("sessionStorage", HISTORY_KEY);
    renderHomeHistory();
    renderHistoryPop();
    toast("History cleared");
  }
});

function openPopover() {
  renderHistoryPop();
  $("history-pop").classList.remove("hidden");
  $("history-btn").setAttribute("aria-expanded", "true");
}
function closePopover() {
  $("history-pop").classList.add("hidden");
  $("history-btn").setAttribute("aria-expanded", "false");
}
$("history-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  $("history-pop").classList.contains("hidden") ? openPopover() : closePopover();
});
document.addEventListener("click", (e) => { if (!e.target.closest(".pop-wrap")) closePopover(); });

// ------------------------------------------------------------------ the address bar mirrors the search
function toURL(page) {
  const p = new URLSearchParams();
  p.set("q", $("q").value.trim());
  for (const [k, v] of Object.entries(filtersNow())) p.set(k, v);
  if (page > 1) p.set("page", page);
  if (state.exact) p.set("exact", "1");
  return "?" + p.toString();
}
function fromURL() {
  const p = new URLSearchParams(location.search);
  $("q").value = p.get("q") || "";
  for (const id of FILTERS) $(id).value = p.get(id) || (id === "sort" ? "relevance" : "");
  $("strict").checked = p.get("strict") === "1";
  state.exact = p.get("exact") === "1";
  return { q: $("q").value.trim(), page: Number(p.get("page")) || 1 };
}
window.addEventListener("popstate", () => {
  const { q, page } = fromURL();
  closeSheets();
  if (q) search(page, state.exact, false); else showHome(true);
});

// ------------------------------------------------------------------ search
function apiParams(page) {
  const p = new URLSearchParams({ q: $("q").value.trim(), page });
  for (const [k, v] of Object.entries(filtersNow())) p.set(k, v);
  for (const id of ["online", "mode"]) if ($(id).value) p.set(id, $(id).value);
  if (state.exact) p.set("exact", "1");
  return p;
}

let searchSeq = 0;
async function search(page = 1, exact = state.exact, push = true) {
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  if (page === 1) state.exact = exact;
  const seq = ++searchSeq;
  hideSuggest();
  state.page = page;
  showHome(false);
  updateFilterBadge();
  if (push) window.history.pushState(null, "", toURL(page));
  document.title = `${q} - PaperScope`;
  document.body.classList.add("is-loading");
  $("progress").classList.add("on");
  if (!state.results.length) $("results").innerHTML = "";
  try {
    const res = await getJSON("/api/search?" + apiParams(page));
    if (seq !== searchSeq) return;                              // a newer search replaced this one
    state.trace = res.trace;
    state.results = res.results;
    renderResults(res);
    if (page === 1) remember(q, res.total);
    if (!$("how-sheet").classList.contains("hidden")) renderHow();
  } catch (e) {
    if (seq !== searchSeq) return;
    state.results = [];
    $("notice").innerHTML = "";
    $("summary").innerHTML = "";
    $("pager").innerHTML = "";
    $("results").innerHTML = `<li class="state error"><h2>The search could not finish</h2><p>${esc(e.message)}</p>
      <p>Check that PaperScope is still running, then search again.</p></li>`;
  } finally {
    if (seq === searchSeq) {
      document.body.classList.remove("is-loading");
      $("progress").classList.remove("on");
    }
  }
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

function shortAuthors(authors, n = 3) {
  const list = (authors || "").split(",").map((a) => a.trim()).filter(Boolean);
  if (!list.length) return "Unknown authors";
  if (list.length <= n) return list.join(", ");
  return `${list.slice(0, n).join(", ")} and ${list.length - n} more`;
}

function paperLink(p) { return p.doi ? `https://doi.org/${p.doi}` : p.url || ""; }

function matchHTML(p) {
  // how much of the search the paper covers, in words and meaning (engine/pipeline.py, match_scores)
  const pct = Math.round((p.match || 0) * 100);
  const label = { strong: "Strong match", good: "Good match", partial: "Partial match" }[p.match_level] || "Match";
  return `<span class="match m-${esc(p.match_level)}" title="Covers about ${pct}% of your search, in words and meaning. Citations and year are not counted here.">` +
    `<span class="meter" aria-hidden="true"><i style="width:${pct}%"></i></span>${label} ${pct}%</span>`;
}

function factsHTML(p, withMatch = true, withLink = true) {
  const f = [];
  if (withMatch && p.match != null) f.push(matchHTML(p));
  if (p.published === true) f.push(`<span class="fact-good">Peer reviewed</span>`);
  else if (p.published === false) f.push(`<span class="fact-warn">Preprint</span>`);
  if (p.venue_quality >= 0.75) f.push(`<span>${p.venue_quality >= 1 ? "Top" : "Strong"} venue</span>`);
  if (p.citations != null) f.push(`<span>${p.citations.toLocaleString()} ${p.citations === 1 ? "citation" : "citations"}</span>`);
  if (p.field) f.push(`<span>${esc(p.field)}</span>`);
  if (p.oa_url && withLink) f.push(`<a href="${esc(p.oa_url)}" target="_blank" rel="noopener">Free full text</a>`);
  return `<div class="facts">${f.join("")}</div>`;
}

function byline(p) {
  const where = [p.venue ? `<span class="venue">${esc(p.venue)}</span>` : "", p.year ? esc(p.year) : ""].filter(Boolean).join(", ");
  return `${esc(shortAuthors(p.authors))}${where ? `. ${where}` : ""}`;
}

function renderResults(res) {
  const words = res.trace.query?.words || [];
  $("notice").innerHTML = noticeHTML(res.trace.understood);
  if (!res.results.length) {
    $("summary").innerHTML = "";
    $("pager").innerHTML = "";
    const filtered = Object.keys(filtersNow()).some((k) => k !== "sort");
    $("results").innerHTML = `<li class="state"><h2>No papers match this search</h2>
      <p>Try fewer or more general words${filtered ? ", or clear the filters" : ""}.</p>
      ${filtered ? `<button type="button" class="btn" data-reset>Clear filters</button>` : ""}</li>`;
    return;
  }
  const first = (res.page - 1) * res.per_page;
  $("summary").innerHTML = `<span>${res.total.toLocaleString()} ${res.total === 1 ? "paper" : "papers"}` +
    `${res.page > 1 ? `, page ${res.page}` : ""}</span>
    <button type="button" class="link small" id="how-btn">How this search worked</button>`;
  $("results").innerHTML = res.results.map((p, i) => `
    <li class="result" value="${first + i + 1}">
      <h2 class="result-title"><a href="${esc(paperLink(p) || "#")}" data-key="${esc(p.key)}">${esc(p.title)}</a></h2>
      <div class="byline">${byline(p)}</div>
      ${p.abstract ? `<p class="abstract">${highlight(p.abstract, words)}</p>` : ""}
      ${p.missing && p.missing.length ? `<div class="missing">Does not mention ${p.missing.map((w) => `<s>${esc(w)}</s>`).join(", ")}</div>` : ""}
      ${factsHTML(p)}
    </li>`).join("");
  const more = first + res.per_page < res.total;
  $("pager").innerHTML = (res.page > 1 ? `<button type="button" class="pg" data-page="${res.page - 1}">Previous page</button>` : "") +
    (more ? `<button type="button" class="pg" data-page="${res.page + 1}">Next page</button>` : "");
}

// What the query understanding changed (engine/understand.py), with a way back to the words as typed.
function noticeHTML(u) {
  if (!u || !u.changed) return "";
  const notes = [];
  if (u.corrections.length) notes.push("Spelling corrected: " + u.corrections.map((c) => `${esc(c.from)} to <b>${esc(c.to)}</b>`).join(", ") + ".");
  if (u.recent_from) notes.push(`Only papers from ${u.recent_from} on, because you asked for recent work.`);
  return `<div class="notice">Showing results for <b>${esc(u.query)}</b>` +
    (notes.length ? `<div class="sub">${notes.join(" ")}</div>` : "") +
    `<div class="sub">Search instead for <button type="button" class="link" id="exact">${esc(u.original)}</button></div></div>`;
}

// ------------------------------------------------------------------ filters
function updateFilterBadge() {
  const extra = ($("author").value ? 1 : 0) + ($("strict").checked ? 1 : 0);
  $("more-btn").innerHTML = `<svg aria-hidden="true"><use href="#i-sliders"/></svg>More filters${extra ? ` <span class="count">${extra}</span>` : ""}`;
  $("reset-filters").classList.toggle("hidden", !Object.keys(filtersNow()).some((k) => k !== "sort"));
}
function resetFilters() {
  for (const id of FILTERS) $(id).value = id === "sort" ? "relevance" : "";
  $("strict").checked = false;
  search(1);
}
$("more-btn").addEventListener("click", () => {
  const open = $("more").classList.toggle("hidden") === false;
  $("more-btn").setAttribute("aria-expanded", String(open));
  if (open) $("author").focus();
});
$("reset-filters").addEventListener("click", resetFilters);
for (const id of ["field", "sort", "strict", "mode", "online"]) $(id).addEventListener("change", () => search(1));
for (const id of ["year_from", "year_to", "author"]) {
  $(id).addEventListener("change", () => search(1));
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); search(1); } });
}

async function loadFields() {
  if (state.fieldsLoaded) return;
  try {
    const s = await getJSON("/api/state");
    const want = new URLSearchParams(location.search).get("field") || $("field").value;
    for (const f of s.fields) $("field").insertAdjacentHTML("beforeend", `<option>${esc(f)}</option>`);
    $("field").value = want;
    state.fieldsLoaded = true;
  } catch (e) { /* the field list arrives with the next search */ }
}

// ------------------------------------------------------------------ sheets
function openSheet(id) {
  state.lastFocus = state.lastFocus || document.activeElement;
  for (const s of document.querySelectorAll(".sheet")) s.classList.toggle("hidden", s.id !== id);
  $("backdrop").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  $(id).querySelector("[data-close]").focus();
}
function closeSheets() {
  for (const s of document.querySelectorAll(".sheet")) s.classList.add("hidden");
  $("backdrop").classList.add("hidden");
  document.body.style.overflow = "";
  if (state.lastFocus && document.contains(state.lastFocus)) state.lastFocus.focus();
  state.lastFocus = null;
}
document.addEventListener("click", (e) => { if (e.target.closest("[data-close]") || e.target.id === "backdrop") closeSheets(); });

// ------------------------------------------------------------------ paper sheet
function citation(p) {
  const authors = shortAuthors(p.authors, 6);
  const bits = [`${authors} (${p.year || "n.d."}). ${p.title}.`];
  if (p.venue) bits.push(`${p.venue}.`);
  const link = paperLink(p);
  if (link) bits.push(link);
  return bits.join(" ");
}

function relatedHTML(p) {
  // Dijkstra's shortest paths over citation, related-work and similarity links (dsa/graph.py)
  const LINK = { cites: "Cited by this paper", "cited by": "Cites this paper", related: "Listed as related by OpenAlex", similar: "Similar topic" };
  const rel = (p.related || []).map((r) => `<li><a href="#" data-key="${esc(r.key)}">${esc(r.title)}</a>
    <span class="why"><span class="closeness" title="Closeness ${Math.round(r.closeness * 100)}%"><i style="width:${Math.round(r.closeness * 100)}%"></i></span>` +
    `${r.via ? `Through “${esc(r.via)}”` : esc(LINK[r.link] || r.link)}${r.year ? `, ${esc(r.year)}` : ""}</span></li>`).join("");
  // breadth-first search over citation links only (dsa/graph.py)
  const how = (r) => r.hops === 1 ? (r.relation === "reference" ? "Cited by this paper" : "Cites this paper") : `Two links away, through “${esc(r.via)}”`;
  const cit = (p.citation_links || []).map((r) =>
    `<li><a href="#" data-key="${esc(r.key)}">${esc(r.title)}</a><span class="why">${how(r)}${r.year ? `, ${esc(r.year)}` : ""}</span></li>`).join("");
  return `<h3 class="block-h">Related papers</h3><p class="block-sub">Closest first, through citations, OpenAlex's related works and similar topics.</p>` +
    (rel ? `<ul class="paper-list">${rel}</ul>` : `<p class="block-sub">No related papers yet. Search this paper's topic to bring more in.</p>`) +
    (cit ? `<h3 class="block-h">Citation links</h3><p class="block-sub">Papers it cites, or that cite it.</p><ul class="paper-list">${cit}</ul>` : "");
}

async function openPaper(key) {
  openSheet("paper-sheet");
  $("paper-body").innerHTML = `<p class="muted">Loading the paper…</p>`;
  try {
    const p = await getJSON("/api/paper?key=" + encodeURIComponent(key));
    const shown = state.results.find((r) => r.key === key);
    const feats = shown ? shown.features : null;
    const view = { ...p, field: shown ? shown.field : "", match: shown ? shown.match : null, match_level: shown ? shown.match_level : "" };
    const link = paperLink(p);
    $("paper-body").innerHTML = `
      <h2 class="sheet-title" id="paper-title">${esc(p.title)}</h2>
      <p class="paper-authors">${esc(shortAuthors(p.authors, 12))}</p>
      <p class="paper-where">${[p.venue ? `<i>${esc(p.venue)}</i>` : "", p.year ? esc(p.year) : ""].filter(Boolean).join(", ")}</p>
      <div class="actions">
        ${link ? `<a class="btn btn-primary" href="${esc(link)}" target="_blank" rel="noopener">Open the paper</a>` : ""}
        ${p.oa_url ? `<a class="btn" href="${esc(p.oa_url)}" target="_blank" rel="noopener">Free full text</a>` : ""}
        <button type="button" class="btn" id="copy-cite">Copy citation</button>
      </div>
      <div class="paper-facts">${factsHTML(view, view.match != null, false)}</div>
      <p class="paper-abstract">${p.abstract ? esc(p.abstract) : `<span class="muted">No abstract available.</span>`}</p>
      ${p.topic ? `<p class="block-sub">Topic: ${esc(p.topic.split(" | ").join(", "))}</p>` : ""}
      ${relatedHTML(p)}
      ${feats ? `<details class="ranking"><summary>Why it ranked here</summary>
        <table class="feature-table">${Object.keys(FEATURE_NAMES).filter((k) => k in feats).map((k) =>
          `<tr><td>${esc(FEATURE_NAMES[k])}</td><td>${feats[k] < 0 ? "unknown" : feats[k].toFixed(2)}</td></tr>`).join("")}</table></details>` : ""}`;
    $("copy-cite").addEventListener("click", () => copyText(citation(p)));
  } catch (e) {
    $("paper-body").innerHTML = `<div class="state error"><h2>This paper could not be opened</h2><p>${esc(e.message)}</p></div>`;
  }
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const t = document.createElement("textarea");
    t.value = text; document.body.appendChild(t); t.select();
    try { document.execCommand("copy"); } finally { t.remove(); }
  }
  toast("Citation copied");
}

// ------------------------------------------------------------------ how this search worked
function duration(s) { return s >= 5400 ? `${Math.round(s / 3600)} h` : s >= 90 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`; }

function renderHow() {
  const t = state.trace;
  if (!t) { $("how-body").innerHTML = `<p class="muted">Search for something first.</p>`; return; }
  const rows = [];
  const u = t.understood;
  if (u) {
    const how = [u.kind === "question" ? "A question" : u.kind === "request" ? "A request" : "Keywords"];
    if (u.semantic) how.push("also searched by meaning");
    if (u.title) how.push("also searched titles");
    rows.push(`<h3 class="how-h">Understood as</h3><div>${how.map(esc).join(", ")}.` +
      (u.removed.length ? `<br>Left out: <code>${esc(u.removed.join(" "))}</code>` : "") +
      (u.corrections.length ? `<br>Spelling: ${u.corrections.map((c) => `<code>${esc(c.from)}</code> to <code>${esc(c.to)}</code>`).join(", ")}` : "") +
      (u.recent_from ? `<br>Recent: from ${u.recent_from}` : "") +
      (u.relaxed ? `<br>Also searched with fewer words: <code>${esc(u.relaxed)}</code>, without ${u.relax_dropped.map((w) => `<code>${esc(w)}</code>`).join(", ")}` : "") + "</div>");
  }
  rows.push(`<h3 class="how-h">Words searched</h3><div><code>${esc((t.query?.words || []).join(" ") || "none")}</code>` +
    (t.query?.excluded?.length ? `<br>Excluded: <code>${esc(t.query.excluded.join(" "))}</code>` : "") + "</div>");
  if (t.identifier) {
    rows.push(`<h3 class="how-h">Identifier</h3><div>${esc(t.identifier.kind)} <code>${esc(t.identifier.value)}</code>, ` +
      `${t.identifier.in_memory ? "already in memory" : "looked up online"}</div>`);
    const errs = t.identifier.online?.errors || [];
    if (errs.length) rows.push(`<div class="err">${errs.map(esc).join("<br>")}</div>`);
  }
  if (t.boolean && t.boolean.error) {
    rows.push(`<h3 class="how-h">AND / OR / NOT</h3><div class="err">${esc(t.boolean.error)}</div>`);
  } else if (t.boolean) {
    rows.push(`<h3 class="how-h">AND / OR / NOT, evaluated on a stack</h3><div>Postfix: <code>${esc(t.boolean.postfix.join(" "))}</code></div>` +
      `<table>${t.boolean.steps.map((s) => `<tr><td><code>${esc(s.token)}</code></td><td><div class="stack">` +
        `${s.stack.map((x) => `<span>${esc(x)}</span>`).join("")}</div></td></tr>`).join("")}</table>` +
      `<div class="muted">${t.boolean.matches} papers in memory match</div>`);
  }
  if (t.online) {
    const o = t.online;
    rows.push(`<h3 class="how-h">Internet</h3><div>${o.used ? `Asked because ${esc(o.reason)}.` : `Not needed: ${esc(o.reason)}.`}</div>`);
    if (o.used) {
      rows.push(`<table>${o.providers.map((p) => `<tr><td>${esc(p.provider)}</td><td class="${p.ok ? "ok" : "err"}">` +
        `${p.ok ? `${p.results} papers${p.cached ? ", from cache" : ""}` : esc(p.error || "failed")}</td><td class="num">${p.ms ?? ""} ms</td></tr>`).join("")}</table>` +
        `<div class="muted">${o.fetched} fetched, ${o.new} new to memory, ${o.ms} ms` +
        (o.providers.some((p) => /arXiv|Europe PMC/.test(p.provider)) ? `. The free source was picked for ${esc(o.field_source)}.` : ".") + `</div>` +
        (o.saver ? `<div class="saver">Saver mode: ${esc(o.saver.reason)}. Free sources were asked first` +
          (o.saver.skipped.length ? `, and the extra searches ${o.saver.skipped.map(esc).join(", ")} were skipped` : "") + ".</div>" : ""));
    }
  }
  if (t.candidates) {
    const c = t.candidates;
    rows.push(`<h3 class="how-h">Candidates</h3><table>` +
      `<tr><td>Inverted index (BM25)</td><td class="num">${c.lexical}</td><td class="num">${c.ms.lexical} ms</td></tr>` +
      `<tr><td>Vector search</td><td class="num">${c.semantic}</td><td class="num">${c.ms.semantic} ms</td></tr>` +
      `<tr><td>From the internet</td><td class="num">${c.from_online}</td><td></td></tr>` +
      `<tr><td><b>Re-ranked</b></td><td class="num"><b>${c.total}</b></td><td></td></tr></table>` +
      `<div class="muted">${c.in_memory.toLocaleString()} papers in memory, ${c.allowed_by_filters.toLocaleString()} pass the filters</div>`);
  }
  if (t.field) {
    rows.push(`<h3 class="how-h">Research field</h3><div>${t.field.chosen ? `Chosen: <b>${esc(t.field.chosen)}</b><br>` : ""}` +
      t.field.detected.map((d) => `${esc(d.name)} ${(d.p * 100).toFixed(0)}%`).join("<br>") + "</div>");
  }
  if (t.rerank) rows.push(`<h3 class="how-h">Ranking</h3><div>${esc(t.rerank.ranker)}, confident above ${t.rerank.threshold}` +
    (t.fusion ? `<br>Combined with the order of ${esc(t.fusion.with)} (reciprocal rank fusion)` : "") +
    (t.exact_title ? `<br>Exact title match moved to the top` : "") +
    (t.sorted_on_topic != null ? `<br>${t.sort === "year" ? "Newest" : "Most cited"} first among the ${t.sorted_on_topic} papers that match best, then the rest by relevance` : "") +
    `<br>Total ${t.total_ms} ms</div>`);
  $("how-body").innerHTML = rows.join("");
  renderSystem();
}

async function renderSystem() {
  try {
    const s = await getJSON("/api/state");
    const b = s.budget || {};
    const lines = [`${s.papers_in_memory.toLocaleString()} of ${s.capacity.toLocaleString()} papers in memory.`,
      `Ranker: ${esc(s.ranker.kind)}${s.ranker.trees ? `, ${s.ranker.trees} trees` : ""}.`];
    if (b.known) lines.push(`OpenAlex budget today: $${b.remaining_usd.toFixed(3)} of $${b.limit_usd.toFixed(2)} left` +
      (b.resets_in_s != null ? `, full again in ${duration(b.resets_in_s)}` : "") + ".");
    if (b.saver) lines.push(`<span class="saver">Saver mode is on: ${esc(b.saver_reason)}.</span>`);
    lines.push(`Spent since the app started: $${(b.spent_usd || 0).toFixed(4)} in ${b.requests || 0} OpenAlex requests` +
      (b.cache_file ? `, ${b.cache_file_hits} answers from the cache file` : "") + ".");
    $("system").innerHTML = `<h3 class="how-h">This app</h3><p>${lines.join("<br>")}</p>`;
  } catch (e) { $("system").innerHTML = ""; }
}

// ------------------------------------------------------------------ suggestions
let sugTimer = null, sugIndex = -1, sugSeq = 0;
function hideSuggest() { clearTimeout(sugTimer); sugSeq++; $("suggest").classList.add("hidden"); sugIndex = -1; $("q").removeAttribute("aria-activedescendant"); }
function currentWord() { const parts = $("q").value.split(/\s+/); return parts[parts.length - 1]; }
function showItems(html) {
  if (!html) return hideSuggest();
  $("suggest").innerHTML = html;
  $("suggest").classList.remove("hidden");
  sugIndex = -1;
}
function recentItems(text, n) {
  const t = text.trim().toLowerCase();
  return pastSearches().map((h, i) => ({ h, i })).filter(({ h }) => !t || h.q.toLowerCase().startsWith(t)).slice(0, n)
    .map(({ h, i }) => `<li role="option" data-h-sug="${i}"><span class="kind">Recent</span><span>${esc(h.q)}</span></li>`).join("");
}
async function suggest() {
  const text = $("q").value;
  if (!text.trim()) return document.body.classList.contains("is-home") ? hideSuggest() : showItems(recentItems("", 6));
  const w = currentWord();
  if ((w.length < 2 || /^[A-Z]+$/.test(w)) && text.trim().length < 3) return showItems(recentItems(text, 4));
  const seq = ++sugSeq;
  const { suggestions = [], titles = [] } = await getJSON("/api/suggest?prefix=" + encodeURIComponent(w) + "&q=" + encodeURIComponent(text))
    .catch(() => ({}));
  if (seq !== sugSeq || document.activeElement !== $("q")) return;   // a search started meanwhile
  // recent searches, then words (binary search in the vocabulary), then paper titles (radix trie)
  showItems(recentItems(text, 2) +
    suggestions.slice(0, titles.length ? 4 : 6).map((s) => `<li role="option" data-word="${esc(s)}"><span class="kind">Word</span><span>${esc(s)}</span></li>`).join("") +
    titles.map((t) => `<li role="option" data-key="${esc(t.key)}"><span class="kind">Paper</span><span class="title-text">${esc(t.title)}</span>` +
      `<span class="year">${esc(t.year || "")}</span></li>`).join(""));
}
function pickItem(li) {
  if (li.dataset.hSug != null) { hideSuggest(); runHistory(Number(li.dataset.hSug)); return; }
  if (li.dataset.key) { hideSuggest(); openPaper(li.dataset.key); return; }
  const parts = $("q").value.split(/(\s+)/);
  parts[parts.length - 1] = li.dataset.word;
  $("q").value = parts.join("") + " ";
  hideSuggest();
  $("q").focus();
}

// ------------------------------------------------------------------ events
$("search").addEventListener("submit", (e) => { e.preventDefault(); search(1, false); });
$("q").addEventListener("input", () => { clearTimeout(sugTimer); sugTimer = setTimeout(suggest, 140); });
$("q").addEventListener("focus", () => { if (!$("q").value.trim()) suggest(); });
$("q").addEventListener("keydown", (e) => {
  const items = [...$("suggest").querySelectorAll("li")];
  if ($("suggest").classList.contains("hidden") || !items.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    sugIndex = (sugIndex + (e.key === "ArrowDown" ? 1 : items.length - 1 + (sugIndex < 0 ? 1 : 0))) % items.length;
    items.forEach((li, i) => { li.classList.toggle("on", i === sugIndex); li.id = i === sugIndex ? "sug-on" : ""; });
    $("q").setAttribute("aria-activedescendant", "sug-on");
  } else if (e.key === "Enter" && sugIndex >= 0) {
    e.preventDefault();
    pickItem(items[sugIndex]);
  } else if (e.key === "Tab" && items[0].dataset.word && sugIndex < 0) {
    e.preventDefault();
    pickItem(items.find((li) => li.dataset.word) || items[0]);
  } else if (e.key === "Escape") hideSuggest();
});
$("suggest").addEventListener("mousedown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pickItem(li); } });
$("q").addEventListener("blur", () => setTimeout(hideSuggest, 120));

$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (!b) return;
  $("q").value = b.dataset.q;
  search(1, false);
});
$("home-link").addEventListener("click", (e) => {
  e.preventDefault();
  $("q").value = "";
  state.results = []; state.trace = null;
  document.title = "PaperScope";
  window.history.pushState(null, "", "/");
  showHome(true);
  $("q").focus();
});
document.querySelector(".results-view").addEventListener("click", (e) => {
  if (e.target.id === "exact") { search(1, true); return; }
  if (e.target.id === "how-btn") { openSheet("how-sheet"); renderHow(); return; }
  if (e.target.closest("[data-reset]")) { resetFilters(); return; }
  const pg = e.target.closest("button[data-page]");
  if (pg) { search(Number(pg.dataset.page)); window.scrollTo({ top: 0 }); return; }
  const a = e.target.closest("a[data-key]");
  if (a && !(e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1)) { e.preventDefault(); openPaper(a.dataset.key); }
});
$("paper-body").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-key]");
  if (a) { e.preventDefault(); openPaper(a.dataset.key); $("paper-sheet").scrollTop = 0; }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeSheets(); closePopover(); }
  if (e.key === "/" && !/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName)) { e.preventDefault(); $("q").focus(); }
});

// ------------------------------------------------------------------ start
const narrow = window.matchMedia("(max-width: 720px)");
function placeholder() { $("q").placeholder = narrow.matches ? "Search papers or a DOI" : "Ask about a topic, or paste a DOI"; }
narrow.addEventListener("change", placeholder);
placeholder();
paintThemeButton();
loadFields().then(() => {
  const { q, page } = fromURL();
  if (q) search(page, state.exact, false);
  else showHome(true);
});
showHome(!new URLSearchParams(location.search).get("q"));
