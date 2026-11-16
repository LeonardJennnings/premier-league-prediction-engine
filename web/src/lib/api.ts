/**
 * Thin read-only client for the Epic 4 Predictions API.
 *
 * Base URL: set `PUBLIC_PREDICTIONS_API_BASE` (e.g. http://127.0.0.1:8000).
 * No trailing slash. The UI never triggers fit/publish/settle.
 */

import type {
  ApiResult,
  ModelMeta,
  UpcomingResponse,
  WeekResponse,
  WeeksResponse,
} from "./types";

const DEFAULT_BASE = "http://127.0.0.1:8000";

export function getApiBase(): string {
  const raw = import.meta.env.PUBLIC_PREDICTIONS_API_BASE as string | undefined;
  const base = (raw && raw.trim()) || DEFAULT_BASE;
  return base.replace(/\/$/, "");
}

async function getJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const url = `${getApiBase()}${path}`;
  try {
    const res = await fetch(url, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers ?? {}),
      },
    });
    if (!res.ok) {
      let detail = res.statusText || `HTTP ${res.status}`;
      try {
        const body = (await res.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        /* keep statusText */
      }
      return { ok: false, status: res.status, detail };
    }
    return { ok: true, data: (await res.json()) as T };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Network error";
    return { ok: false, status: 0, detail: message };
  }
}

export function fetchUpcoming(modelVersion?: string): Promise<ApiResult<UpcomingResponse>> {
  const q = modelVersion
    ? `?model_version=${encodeURIComponent(modelVersion)}`
    : "";
  return getJson<UpcomingResponse>(`/v1/predictions/upcoming${q}`);
}

export function fetchWeeks(modelVersion?: string): Promise<ApiResult<WeeksResponse>> {
  const q = modelVersion
    ? `?model_version=${encodeURIComponent(modelVersion)}`
    : "";
  return getJson<WeeksResponse>(`/v1/predictions/weeks${q}`);
}

export function fetchWeek(
  isoWeek: string,
  modelVersion?: string,
): Promise<ApiResult<WeekResponse>> {
  const q = modelVersion
    ? `?model_version=${encodeURIComponent(modelVersion)}`
    : "";
  return getJson<WeekResponse>(
    `/v1/predictions/weeks/${encodeURIComponent(isoWeek)}${q}`,
  );
}

export function fetchLatestModel(): Promise<ApiResult<ModelMeta>> {
  return getJson<ModelMeta>(`/v1/models/latest`);
}

export function fetchModel(modelVersion: string): Promise<ApiResult<ModelMeta>> {
  return getJson<ModelMeta>(`/v1/models/${encodeURIComponent(modelVersion)}`);
}
