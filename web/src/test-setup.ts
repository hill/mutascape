import { Window } from "happy-dom";

const browserWindow = new Window({ url: "http://localhost/" });

Object.defineProperty(browserWindow.Element.prototype, "getAnimations", {
  configurable: true,
  value: () => [],
});

Object.defineProperty(browserWindow.Element.prototype, "scrollIntoView", {
  configurable: true,
  value: () => undefined,
});

Object.assign(globalThis, {
  window: browserWindow,
  document: browserWindow.document,
  navigator: browserWindow.navigator,
  history: browserWindow.history,
  location: browserWindow.location,
  sessionStorage: browserWindow.sessionStorage,
  HTMLElement: browserWindow.HTMLElement,
  Element: browserWindow.Element,
  Node: browserWindow.Node,
  Event: browserWindow.Event,
  MouseEvent: browserWindow.MouseEvent,
  KeyboardEvent: browserWindow.KeyboardEvent,
  File: browserWindow.File,
  DataTransfer: browserWindow.DataTransfer,
  MutationObserver: browserWindow.MutationObserver,
  ResizeObserver: browserWindow.ResizeObserver,
  getComputedStyle: browserWindow.getComputedStyle.bind(browserWindow),
  requestAnimationFrame: browserWindow.requestAnimationFrame.bind(browserWindow),
  cancelAnimationFrame: browserWindow.cancelAnimationFrame.bind(browserWindow),
  scrollTo: browserWindow.scrollTo.bind(browserWindow),
});
