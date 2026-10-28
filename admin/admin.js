// Admin tools, sent only to the signed-in admin (admin/__init__.py) and never to visitors:
// an "Admin" mark with a sign-out button, the "How this search worked" sheet with the engine's
// trace and the test settings, the OpenAlex budget, and why each paper ranked where it did.
(() => {
  const { $, esc, getJSON, state, search, openSheet } = window.paperscope;
  const BASE = new URL(".", document.currentScript.src).pathname;       // the secret admin address
  const FEATURE_NAMES = {
    bm25: "Word match (BM25)", semantic: "Meaning (embeddings)", title_coverage: "Your words in the title",
    term_coverage: "Your words anywhere", domain_match: "Same research field", citations: "Citations per year",
    recency: "Recency", peer_reviewed: "Peer reviewed", venue_quality: "Venue quality",
  };
  let trace = null;

  document.querySelector(".bar-actions").insertAdjacentHTML("afterbegin", `
    <form class="admin-out" method="post" action="${esc(BASE)}logout">
      <span class="admin-badge" title="Signed in as the admin. Visitors do not see the admin tools.">Admin</span>
      <button class="icon-btn" type="submit" title="Sign out">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h4M15 16l4-4-4-4M19 12H9" fill="none"
          stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="sr">Sign out</span>
      </button>
    </form>`);

  document.body.insertAdjacentHTML("beforeend", `
    <aside class="sheet sheet-narrow hidden" id="how-sheet" role="dialog" aria-modal="true" aria-labelledby="how-title">
      <button class="icon-btn sheet-close" type="button" data-close title="Close"><svg aria-hidden="true"><use href="#i-close"/></svg><span class="sr">Close</span></button>
      <div class="sheet-body">
        <h2 id="how-title" class="sheet-title">How this search worked</h2>
        <p class="muted small">The steps PaperScope took for this search, and the data structures behind them. Only you see this.</p>
        <div id="how-body" class="how"></div>
        <h3 class="how-h">Test settings</h3>
        <label class="stack-field">Ranking method
          <select id="mode"><option value="learned">Learned ranker (default)</option><option value="hybrid">Hand-set weights</option>
            <option value="bm25">Word match only (BM25)</option><option value="semantic">Meaning only (embeddings)</option></select></label>
        <label class="stack-field">Ask the internet
          <select id="online"><option value="auto">When memory is not enough (default)</option><option value="always">Always</option>
            <option value="never">Never, memory only</option></select></label>
        <div id="system" class="how small"></div>
      </div>
    </aside>`);

  for (const id of ["mode", "online"]) {
    $(id).addEventListener("change", () => {
      state.extra = { mode: $("mode").value, online: $("online").value };
      search(1);
    });
  }

  function showResults(res) {
    trace = res.trace;
    $("summary").insertAdjacentHTML("beforeend", `<button type="button" class="link small" id="how-btn">How this search worked</button>`);
    if (!$("how-sheet").classList.contains("hidden")) renderHow();
  }
  document.addEventListener("paperscope:results", (e) => showResults(e.detail));
  if (state.last) showResults(state.last);                // results that came before this script did

  $("results-view").addEventListener("click", (e) => {
    if (e.target.id === "how-btn") { openSheet("how-sheet"); renderHow(); }
  });

  document.addEventListener("paperscope:paper", (e) => {
    const feats = e.detail.shown?.features;
    if (!feats) return;
    $("paper-body").insertAdjacentHTML("beforeend", `<details class="ranking"><summary>Why it ranked here</summary>
      <table class="feature-table">${Object.keys(FEATURE_NAMES).filter((k) => k in feats).map((k) =>
        `<tr><td>${esc(FEATURE_NAMES[k])}</td><td>${feats[k] < 0 ? "unknown" : feats[k].toFixed(2)}</td></tr>`).join("")}</table></details>`);
  });

  // ---------------------------------------------------------------- how this search worked
  function duration(s) { return s >= 5400 ? `${Math.round(s / 3600)} h` : s >= 90 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`; }

  function renderHow() {
    const t = trace;
    if (!t) { $("how-body").innerHTML = `<p class="muted">Search for something first.</p>`; renderSystem(); return; }
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
})();
