import {
  Files,
  History,
  ImagePlus,
  Maximize2,
  Minimize2,
  RefreshCw,
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

interface AnalysisHistoryItem {
  id: string;
  filename: string;
  previewUrl: string;
  result: FrameResult;
  processing: ProcessingTrace;
  source: "image" | "batch";
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode>("image");
  const [result, setResult] = useState<FrameResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [processing, setProcessing] = useState<ProcessingTrace | null>(null);
  const [batchItems, setBatchItems] = useState<BatchDisplayItem[]>([]);
  const [historyItems, setHistoryItems] = useState<AnalysisHistoryItem[]>([]);
  const [openedBatchPreviewUrl, setOpenedBatchPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [presenterMode, setPresenterMode] = useState(false);
  const [operationError, setOperationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const batchItemsRef = useRef<BatchDisplayItem[]>([]);
  const historyItemsRef = useRef<AnalysisHistoryItem[]>([]);
  const imageUrlRef = useRef<string | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealthError(null);
      setHealth(await getHealth());
    } catch (cause) {
      setHealthError(cause instanceof Error ? cause.message : "Backend unavailable");
    }
  }, []);

  useEffect(() => {
    const refreshTimer = window.setTimeout(() => {
      void refreshHealth();
    }, 0);
    return () => window.clearTimeout(refreshTimer);
  }, [refreshHealth]);

  useEffect(() => {
    batchItemsRef.current = batchItems;
  }, [batchItems]);

  useEffect(() => {
    historyItemsRef.current = historyItems;
  }, [historyItems]);

  useEffect(() => {
    imageUrlRef.current = imageUrl;
  }, [imageUrl]);

  useEffect(
    () => () => {
      const urls = new Set([
        imageUrlRef.current,
        ...batchItemsRef.current.map((item) => item.previewUrl),
        ...historyItemsRef.current.map((item) => item.previewUrl),
      ]);
      urls.forEach(revokeObjectUrl);
    },
    [],
  );

  const switchMode = useCallback((mode: SourceMode) => {
    setSourceMode(mode);
    setOpenedBatchPreviewUrl(null);
    setOperationError(null);
  }, []);

  const handleImage = useCallback(async (file: File) => {
    setSourceMode("image");
    setBusy(true);
    setOperationError(null);
    setResult(null);
    setProcessing(null);
    const nextUrl = URL.createObjectURL(file);
    setImageUrl(nextUrl);
    setOpenedBatchPreviewUrl(null);
    try {
      const response = await inferImage(file);
      setProcessing(response.processing);
      setResult(response.result);
      setHistoryItems((current) => [
        {
          id: `image-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          filename: file.name,
          previewUrl: nextUrl,
          result: response.result,
          processing: response.processing,
          source: "image",
        },
        ...current,
      ]);
    } catch (cause) {
      setOperationError(cause instanceof Error ? cause.message : "Image analysis failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const handleBatch = useCallback(async (files: File[]) => {
    setSourceMode("batch");
    setBusy(true);
    setOpenedBatchPreviewUrl(null);
    setOperationError(null);
    const pending: BatchDisplayItem[] = files.slice(0, 100).map((file) => ({
      filename: file.name,
      previewUrl: URL.createObjectURL(file),
    }));
    setBatchItems(pending);
    try {
      const response = await inferBatch(files.slice(0, 100));
      const completed: BatchDisplayItem[] = pending.map((item, index) => ({
        ...item,
        result: response.results[index]?.result,
        processing: response.results[index]?.processing,
        error: response.results[index]?.error,
      }));
      setBatchItems(completed);
      const successfulItems = completed.filter(
        (item): item is BatchDisplayItem & { result: FrameResult; processing: ProcessingTrace } =>
          Boolean(item.result && item.processing && !item.error),
      );
      setHistoryItems((current) => [
        ...successfulItems.map((item) => ({
            id: `batch-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            filename: item.filename,
            previewUrl: item.previewUrl,
            result: item.result,
            processing: item.processing,
            source: "batch" as const,
          })),
        ...current,
      ]);
    } catch (cause) {
      setOperationError(cause instanceof Error ? cause.message : "Batch analysis failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const backendOnline = health?.status === "ok" && !healthError;
  const runtimeLabel = health?.models.detector_device ?? "cpu";
  const chooseImage = () => fileInputRef.current?.click();
  const chooseBatch = () => batchInputRef.current?.click();
  const openHistoryItem = (item: AnalysisHistoryItem) => {
    setSourceMode("image");
    setImageUrl(item.previewUrl);
    setResult(item.result);
    setProcessing(item.processing);
    setOpenedBatchPreviewUrl(item.source === "batch" ? item.previewUrl : null);
    setOperationError(null);
  };
  const openBatchItem = (item: BatchDisplayItem) => {
    if (!item.result || !item.processing || item.error) return;
    setSourceMode("image");
    setImageUrl(item.previewUrl);
    setResult(item.result);
    setProcessing(item.processing);
    setOpenedBatchPreviewUrl(item.previewUrl);
    setOperationError(null);
  };

  return (
    <main className={`app-shell detection-app ${presenterMode ? "presenter-mode" : ""}`}>
      <header className="detection-topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={23} aria-hidden="true" />
          </div>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Colour and shape vision</span>
          </div>
        </div>
        <div className="detection-topbar-actions">
          <span className={`system-status ${backendOnline ? "online" : "offline"}`}>
            {backendOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
            {backendOnline ? "System ready" : "Backend offline"}
          </span>
          <button className="quiet-icon-button" onClick={() => void refreshHealth()} title="Refresh status">
            <RefreshCw size={17} />
            <span className="sr-only">Refresh status</span>
          </button>
          <button
            className="quiet-icon-button"
            onClick={() => setPresenterMode((current) => !current)}
            title={presenterMode ? "Exit presenter mode" : "Presenter mode"}
            aria-pressed={presenterMode}
          >
            {presenterMode ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
            <span className="sr-only">{presenterMode ? "Exit presenter mode" : "Presenter mode"}</span>
          </button>
        </div>
      </header>

      <div className="detection-layout">
        <aside className="upload-rail" aria-label="Image selection">
          <div className="source-mode-switch" aria-label="Analysis mode">
            <button
              aria-pressed={sourceMode === "image"}
              className={sourceMode === "image" ? "active" : ""}
              onClick={() => switchMode("image")}
            >
              <ImagePlus size={17} />
              Image
            </button>
            <button
              aria-pressed={sourceMode === "batch"}
              className={sourceMode === "batch" ? "active" : ""}
              onClick={() => switchMode("batch")}
            >
              <Files size={17} />
              Batch
            </button>
          </div>

          <input
            ref={fileInputRef}
            className="sr-only"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleImage(file);
              event.target.value = "";
            }}
          />
          <input
            ref={batchInputRef}
            className="sr-only"
            type="file"
            multiple
            accept="image/png,image/jpeg,image/webp,image/bmp"
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              if (files.length) void handleBatch(files);
              event.target.value = "";
            }}
          />

          <section className="upload-panel">
            <span className="rail-label">{sourceMode === "image" ? "Upload image" : "Batch analysis"}</span>
            <button
              className="upload-dropzone"
              onClick={sourceMode === "image" ? chooseImage : chooseBatch}
              disabled={!backendOnline || busy}
              aria-label={sourceMode === "image" ? "Choose image" : "Choose images"}
            >
              <Upload size={34} aria-hidden="true" />
              <strong>{busy ? "Analysing image…" : sourceMode === "image" ? "Upload image" : "Choose images"}</strong>
              <span>{sourceMode === "image" ? "PNG, JPG, WEBP or BMP" : "Up to 100 image files"}</span>
            </button>
          </section>

          <section className="recent-section" aria-live="polite">
            <span className="rail-label">Recent analyses</span>
            {historyItems.length ? (
              <div className="recent-analysis-list">
                {historyItems.map((item) => (
                  <button className="recent-analysis" key={item.id} onClick={() => openHistoryItem(item)}>
                    <img src={item.previewUrl} alt="" />
                    <span>
                      <strong>{item.result.events[0]?.label ?? "No sign detected"}</strong>
                      <small>{item.source === "batch" ? "From batch" : "Single image"}</small>
                    </span>
                    {item.source === "batch" ? <Files size={16} aria-hidden="true" /> : <History size={16} aria-hidden="true" />}
                  </button>
                ))}
              </div>
            ) : (
              <p className="recent-empty">Results from this session will appear here.</p>
            )}
          </section>
        </aside>

        <section className="detection-main">
          {sourceMode === "batch" ? (
            <BatchResults items={batchItems} busy={busy} onOpenItem={openBatchItem} />
          ) : (
            <PipelineExplorer
              trace={processing}
              fallbackImageUrl={imageUrl}
              result={result}
              busy={busy}
              runtimeLabel={runtimeLabel}
              onChooseImage={chooseImage}
              onReturnToBatch={openedBatchPreviewUrl ? () => switchMode("batch") : undefined}
            />
          )}
          {operationError || healthError ? <div className="error-banner" role="alert">{operationError || healthError}</div> : null}
        </section>
      </div>
    </main>
  );
}
