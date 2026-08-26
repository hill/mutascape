/// <reference types="bun" />
export {};

const DIST = "./dist";
const ROOT = ".";

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".ttf": "font/ttf",
  ".fasta": "text/plain; charset=utf-8",
};

async function build() {
  const result = await Bun.build({
    entrypoints: ["src/main.tsx"],
    outdir: DIST,
    target: "browser",
    sourcemap: "inline",
    splitting: true,
  });
  if (!result.success) {
    console.error("build failed:");
    for (const m of result.logs) console.error(m);
  } else {
    console.log("built.");
  }
}

await build();

if (process.argv.includes("--watch")) {
  const watcher = (await import("node:fs")).watch(
    "./src",
    { recursive: true },
    () => {
      console.log("change; rebuilding…");
      build();
    },
  );
  process.on("SIGINT", () => {
    watcher.close();
    process.exit(0);
  });
}

function contentTypeFor(path: string): string {
  const dot = path.lastIndexOf(".");
  const ext = dot >= 0 ? path.slice(dot) : "";
  return CONTENT_TYPES[ext] ?? "application/octet-stream";
}

async function serveFrom(base: string, path: string): Promise<Response | null> {
  const file = Bun.file(`${base}${path}`);
  if (await file.exists()) {
    return new Response(file, {
      headers: { "content-type": contentTypeFor(path) },
    });
  }
  return null;
}

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  async fetch(req) {
    const url = new URL(req.url);
    let path = decodeURIComponent(url.pathname);
    if (
      ["/", "/landscapes", "/leaderboards"].includes(path) ||
      path.startsWith("/landscapes/") ||
      path === "/docs" ||
      path.startsWith("/docs/")
    ) {
      path = "/index.html";
    }

    if (path.startsWith("/vendor/katex/")) {
      const r = await serveFrom(
        "./node_modules/katex/dist",
        path.slice("/vendor/katex".length),
      );
      if (r) return r;
    }

    if (path === "/assets/logo.png") {
      return new Response(Bun.file("../docs/logo_ai_image.png"), {
        headers: { "content-type": "image/png" },
      });
    }

    if (path.startsWith("/dist/")) {
      const r = await serveFrom(".", path);
      if (r) return r;
    }

    return (
      (await serveFrom(ROOT, path)) ??
      new Response("Not Found", { status: 404 })
    );
  },
});

console.log(`http://localhost:${server.port}`);
