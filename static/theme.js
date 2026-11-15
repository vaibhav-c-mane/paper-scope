// Light or dark, set in <head> before the page draws so it never flashes the other one.
// A choice made with the moon / sun button wins. Otherwise the page is dark when the device
// asks for it, or at night where the reader is (7 pm to 7 am), which is easier on the eyes.
// Also adds the fonts' stylesheet (see the end).
(() => {
  const KEY = "paperscope.theme";
  const NIGHT_FROM = 19, NIGHT_UNTIL = 7;
  const BAR = { light: "#f2f5f1", dark: "#131a16" };      // a phone's browser bar: --paper in style.css
  const deviceDark = window.matchMedia("(prefers-color-scheme: dark)");
  let picked = null;                                  // this visit's choice, for when storage is blocked
  function chosen() {
    try { const t = localStorage.getItem(KEY); if (t === "light" || t === "dark") return t; } catch (e) { /* blocked */ }
    return picked;
  }
  function automatic() {
    const h = new Date().getHours();
    return deviceDark.matches || h >= NIGHT_FROM || h < NIGHT_UNTIL ? "dark" : "light";
  }
  function show(t) {
    document.documentElement.dataset.theme = t;
    const bar = document.querySelector('meta[name="theme-color"]');
    if (bar) bar.content = BAR[t];
  }
  function apply() { show(chosen() || automatic()); }
  // Picking what the page would show anyway forgets the choice, so it follows the clock again.
  function choose(t) {
    picked = t === automatic() ? null : t;
    try { if (picked) localStorage.setItem(KEY, picked); else localStorage.removeItem(KEY); } catch (e) { /* this visit only */ }
    show(t);
  }
  apply();
  window.paperscopeTheme = { current: () => document.documentElement.dataset.theme, choose, apply, deviceDark };

  // The fonts' stylesheet is added from here instead of in the page: a stylesheet added by a script
  // never holds up the first paint, so on a slow phone connection the text shows at once, in a
  // system font until the fonts arrive (display=swap). The page keeps a preconnect to both hosts.
  const fonts = document.createElement("link");
  fonts.rel = "stylesheet";
  fonts.href = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600"
    + "&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap";
  document.head.appendChild(fonts);
})();
