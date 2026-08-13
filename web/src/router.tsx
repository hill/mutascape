import {
  createRootRoute,
  createRoute,
  createRouter,
  Navigate,
  Outlet,
  type RouterHistory,
  RouterProvider,
  useLocation,
  useNavigate,
} from "@tanstack/react-router"
import App, { type Page } from "./App"

const PAGE_PATHS: Record<Exclude<Page, "explore">, string> = {
  studio: "/",
  models: "/landscapes",
  leaderboards: "/leaderboards",
}

function pageFromPath(pathname: string): Page {
  if (pathname === "/landscapes") return "models"
  if (pathname === "/leaderboards") return "leaderboards"
  if (pathname.startsWith("/landscapes/")) return "explore"
  return "studio"
}

function RoutedApp() {
  const pathname = useLocation({ select: (location) => location.pathname })
  const navigate = useNavigate()
  const landscapeId = pathname.startsWith("/landscapes/")
    ? decodeURIComponent(pathname.slice("/landscapes/".length))
    : undefined
  return (
    <>
      <App
        page={pageFromPath(pathname)}
        landscapeId={landscapeId}
        onPageChange={(page) => {
          if (page !== "explore") void navigate({ to: PAGE_PATHS[page] })
        }}
        onExploreLandscape={(modelId) =>
          void navigate({
            to: "/landscapes/$landscapeId",
            params: { landscapeId: modelId },
          })
        }
      />
      <Outlet />
    </>
  )
}

const rootRoute = createRootRoute({
  component: RoutedApp,
  notFoundComponent: () => <Navigate to="/" replace />,
})

const builderRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => null,
})

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/landscapes",
  component: () => null,
})

const leaderboardsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/leaderboards",
  component: () => null,
})

const landscapeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/landscapes/$landscapeId",
  component: () => null,
})

const routeTree = rootRoute.addChildren([
  builderRoute,
  modelsRoute,
  leaderboardsRoute,
  landscapeRoute,
])

export function createAppRouter(history?: RouterHistory) {
  return createRouter({
    routeTree,
    history,
    defaultPreload: "intent",
    scrollRestoration: true,
  })
}

export const router = createAppRouter()

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}

export { RouterProvider }
