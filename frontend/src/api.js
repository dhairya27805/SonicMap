// Shared API helper. The base is relative so the Vite dev server can proxy
// /api to the FastAPI backend (see vite.config.js); override with VITE_API_BASE.
export const API = import.meta.env.VITE_API_BASE || "/api";

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    credentials: "include",
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    const error = new Error(text || `Request failed: ${response.status}`);
    error.status = response.status;
    try {
      error.detail = JSON.parse(text)?.detail;
    } catch {
      error.detail = undefined;
    }
    throw error;
  }

  return response.json();
}

// Human-readable message from an apiRequest error ({detail: str | {message}}).
export function apiErrorMessage(err) {
  const detail = err?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && detail.message) return detail.message;
  return err?.message || "Request failed.";
}
