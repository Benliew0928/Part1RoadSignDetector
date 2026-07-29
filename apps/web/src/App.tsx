import {
  Activity,
  Cpu,
  Files,
  Gauge,
  ImagePlus,
  Maximize2,
  Minimize2,
  Radio,
  RotateCcw,
  ShieldCheck,
  Upload,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { getHealth, inferBatch, inferImage } from "./api";
import { BatchResults, type BatchDisplayItem } from "./components/BatchResults";
import { PipelineExplorer } from "./components/PipelineExplorer";
import type { FrameResult, HealthResponse, ProcessingTrace, SourceMode } from "./types";

function revokeObjectUrl(url: string | null): void {
  if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode>("image");
  const [result, setResult] = useState<FrameResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [processing, setProcessing] = useState<ProcessingTrace | null>(null);
  const [batchItems, setBatchItems] = useState<BatchDisplayItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [presenterMode, setPresenterMode] = useState(false);
  const [operationError, setOperationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const batchItemsRef = useRef<BatchDisplayItem[]>([]);

  const refreshHealth = useCallback(async () => {
    try {
      setHealthError(null);
      setHealth(await getHealth());
    } catch (cause) {
      setHealthError(cause instanceof Error ? cause.message : "Backend unavailable");
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  useEffect(() => () => revokeObjectUrl(imageUrl), [imageUrl]);

  useEffect(() => {
    batchItemsRef.current = batchItems;
  }, [batchItems]);

  useEffect(
    () => () => {
      batchItemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    },
    [],
  );

  const switchMode = useCallback((mode: SourceMode) => {
    setSourceMode(mode);
    setResult(null);
    setProcessing(null);
    setOperationError(null);
  }, []);

  const handleImage = useCallback(async (file: File) => {
    setSourceMode("image");
    setBusy(true);
    setOperationError(null);
    setProcessing(null);
    const nextUrl = URL.createObjectURL(file);
    setImageUrl((current) => {
      revokeObjectUrl(current);
      return nextUrl;
    });
    try {
      const response = await inferImage(file);
      setProcessing(response.processing);
      setResult(response.result);
    } catch (cause) {
      setOperationError(cause instanceof Error ? cause.message : "Image analysis failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const handleBatch = useCallback(async (files: File[]) => {
    setSourceMode("batch");
    setBusy(true);
    setResult(null);
    setProcessing(null);
    setOperationError(null);
    const pending: BatchDisplayItem[] = files.slice(0, 100).map((file) => ({
      filename: file.name,
      previewUrl: URL.createObjectURL(file),
    }));
    batchItemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setBatchItems(pending);
    try {
      const response = await inferBatch(files.slice(0, 100));
      setBatchItems((current) =>
        current.map((item, index) => ({
          ...item,
          result: response.results[index]?.result,
          error: response.results[index]?.error,
        })),
      );
      setResult(response.results.find((item) => item.result)?.result ?? null);
    } catch (cause) {
      setOperationError(cause instanceof Error ? cause.message : "Batch analysis failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const backendOnline = health?.status === "ok" && !healthError;
  const runtimeLabel = health?.models.detector_device ?? "cpu";

  return (
    <main className={`app-shell ${presenterMode ? "presenter-mode" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={23} aria-hidden="true" />
          </div>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Part 1 colour and shape vision</span>
          </div>
        </div>
        <div className="system-summary">
          <span className={`status-pill ${backendOnline ? "online" : "offline"}`}>
            {backendOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
            {backendOnline ? "System ready" : "Backend offline"}
          </span>
          <span className="status-pill">
            <Cpu size={15} />
            OpenCV baseline
          </span>
          <button className="icon-button" onClick={() => void refreshHealth()} title="Refresh status">
            <RotateCcw size={17} />
            <span className="sr-only">Refresh status</span>
          </button>
          <button
            className="icon-button"
            onClick={() => setPresenterMode((current) => !current)}
            title={presenterMode ? "Exit presenter mode" : "Presenter mode"}
            aria-pressed={presenterMode}
          >
            {presenterMode ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
            <span className="sr-only">{presenterMode ? "Exit presenter mode" : "Presenter mode"}</span>
          </button>
        </div>
      </header>

      <div className="workspace">
        <aside className="control-rail">
          <section>
            <span className="rail-label">Input source</span>
            <div className="segmented-control">
              <button className={sourceMode === "image" ? "active" : ""} onClick={() => switchMode("image")}>
                <ImagePlus size={17} />
                Image
              </button>
              <button className={sourceMode === "batch" ? "active" : ""} onClick={() => switchMode("batch")}>
                <Files size={17} />
                Batch
              </button>
            </div>
          </section>

          <section className="source-actions">
            <input
              ref={sourceMode === "image" ? fileInputRef : batchInputRef}
              className="sr-only"
              type="file"
              multiple={sourceMode === "batch"}
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                if (files.length) {
                  if (sourceMode === "image") void handleImage(files[0]);
                  else void handleBatch(files);
                }
                event.target.value = "";
              }}
            />
            <button
              className="primary-command"
              onClick={() => (sourceMode === "image" ? fileInputRef.current : batchInputRef.current)?.click()}
              disabled={!backendOnline || busy}
            >
              <Upload size={18} />
              {busy ? "Analyzing" : sourceMode === "image" ? "Choose image" : "Choose images"}
            </button>
          </section>

          <section className="metrics-stack">
            <span className="rail-label">Current metrics</span>
            <div className="metric-row"><Activity size={16} /><span>Latency</span><strong>{result ? `${Math.round(result.latency_ms)} ms` : "—"}</strong></div>
            <div className="metric-row"><Gauge size={16} /><span>FPS</span><strong>{result && result.latency_ms > 0 ? (1000 / result.latency_ms).toFixed(1) : "—"}</strong></div>
            <div className="metric-row"><Radio size={16} /><span>Candidates</span><strong>{result?.events.length ?? 0}</strong></div>
            <div className="metric-row"><Cpu size={16} /><span>Runtime</span><strong>{runtimeLabel}</strong></div>
          </section>
        </aside>

        <section className="primary-work">
          {sourceMode === "batch" ? (
            <BatchResults items={batchItems} busy={busy} />
          ) : (
            <PipelineExplorer trace={processing} fallbackImageUrl={imageUrl} result={result} busy={busy} />
          )}
          {operationError || healthError ? <div className="error-banner" role="alert">{operationError || healthError}</div> : null}
          <div className="work-footer">
            <span><span className="status-dot" />{busy ? "processing" : sourceMode}</span>
            <span>Classical OpenCV pipeline</span>
            <span>Frame {result?.frame_id ?? "—"}</span>
          </div>
        </section>
      </div>
    </main>
  );
}
