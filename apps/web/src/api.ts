import type { BatchInferenceResponse, HealthResponse, ImageInferenceResponse } from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return parseJson<HealthResponse>(await fetch("/api/v1/health", { signal }));
}

export async function inferImage(file: File): Promise<ImageInferenceResponse> {
  const body = new FormData();
  body.append("file", file);
  return parseJson<ImageInferenceResponse>(
    await fetch("/api/v1/infer/image", { method: "POST", body }),
  );
}

export async function inferBatch(files: File[]): Promise<BatchInferenceResponse> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return parseJson<BatchInferenceResponse>(
    await fetch("/api/v1/infer/batch", { method: "POST", body }),
  );
}
