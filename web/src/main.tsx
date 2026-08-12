import "@cloudflare/kumo/styles/standalone";
import { TooltipProvider } from "@cloudflare/kumo";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider, router } from "./router";
import "../styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing application root");

createRoot(root).render(
  <StrictMode>
    <TooltipProvider delay={200} closeDelay={80}>
      <RouterProvider router={router} />
    </TooltipProvider>
  </StrictMode>,
);
