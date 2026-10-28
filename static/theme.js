// Light or dark, set in <head> before the page draws so it never flashes the other one.
// A choice made with the moon / sun button wins. Otherwise the page is dark when the device
// asks for it, or at night where the reader is (7 pm to 7 am), which is easier on the eyes.
(() => {
  const KEY = "paperscope.theme";
  const NIGHT_FROM = 19, NIGHT_UNTIL = 7;
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
  function apply() { document.documentElement.dataset.theme = chosen() || automatic(); }
  // Picking what the page would show anyway forgets the choice, so it follows the clock again.
  function choose(t) {
    picked = t === automatic() ? null : t;
    try { if (picked) localStorage.setItem(KEY, picked); else localStorage.removeItem(KEY); } catch (e) { /* this visit only */ }
    document.documentElement.dataset.theme = t;
  }
  apply();
  window.paperscopeTheme = { current: () => document.documentElement.dataset.theme, choose, apply, deviceDark };
})();
