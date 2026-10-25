// Runs an axe-core accessibility audit (WCAG 2 A/AA and best practices) on
// every page of the built site in light and dark mode, and exits non-zero
// if anything is flagged.
//
// Usage: node scripts/check-a11y.js [site-dir]   (default: _site)
// Needs the playwright and axe-core npm packages and a Chromium browser.

const fs = require("fs");
const http = require("http");
const path = require("path");
const { chromium } = require("playwright");

const siteDir = path.resolve(process.argv[2] || "_site");
const axeSource = fs.readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");

const types = {
  ".html": "text/html", ".css": "text/css", ".js": "text/javascript",
  ".svg": "image/svg+xml", ".png": "image/png", ".pdf": "application/pdf",
  ".xml": "application/xml", ".txt": "text/plain",
};

function findPages(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return findPages(full);
    return entry.name.endsWith(".html") ? [full] : [];
  });
}

function urlPath(file) {
  const rel = "/" + path.relative(siteDir, file).split(path.sep).join("/");
  return rel.endsWith("/index.html") ? rel.slice(0, -"index.html".length) : rel;
}

const server = http.createServer((req, res) => {
  let file = path.join(siteDir, decodeURIComponent(req.url.split("?")[0]));
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!file.startsWith(siteDir) || !fs.existsSync(file)) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});

(async () => {
  await new Promise((resolve) => server.listen(0, resolve));
  const base = `http://localhost:${server.address().port}`;
  const browser = await chromium.launch();
  let failures = 0;

  for (const scheme of ["light", "dark"]) {
    const page = await browser.newPage({ colorScheme: scheme });
    // Keep embedded YouTube players out of the audit; they aren't our markup
    await page.route(/youtube\.com/, (route) => route.abort());
    for (const file of findPages(siteDir)) {
      const url = urlPath(file);
      await page.goto(base + url);
      await page.addScriptTag({ content: axeSource });
      const violations = await page.evaluate(async () => {
        const result = await axe.run(document, { runOnly: ["wcag2a", "wcag2aa", "best-practice"] });
        return result.violations.map((v) => ({
          id: v.id,
          help: v.help,
          targets: v.nodes.slice(0, 3).map((n) => n.target.join(" ")),
          count: v.nodes.length,
        }));
      });
      for (const v of violations) {
        failures++;
        console.log(`[${scheme}] ${url}: ${v.id} (${v.count}) - ${v.help}`);
        v.targets.forEach((t) => console.log(`    ${t}`));
      }
    }
    await page.close();
  }

  await browser.close();
  server.close();
  console.log(failures ? `${failures} accessibility issue(s) found` : "No accessibility issues found");
  process.exit(failures ? 1 : 0);
})();
