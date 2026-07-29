import { AlertTriangle, ChevronRight, Files, ScanLine } from "lucide-react";
import type { CSSProperties } from "react";

import type { FrameResult, ProcessingTrace } from "../types";

export interface BatchDisplayItem {
  filename: string;
  previewUrl: string;
  result?: FrameResult | null;
  processing?: ProcessingTrace | null;
  error?: string | null;
}

interface BatchResultsProps {
  items: BatchDisplayItem[];
  busy: boolean;
  onOpenItem: (item: BatchDisplayItem) => void;
}

const PREVIEW_WIDTH = 76;
const PREVIEW_HEIGHT = 52;

function previewMediaBox(result: FrameResult): CSSProperties {
  const mediaRatio = result.width / result.height;
  const previewRatio = PREVIEW_WIDTH / PREVIEW_HEIGHT;
  if (previewRatio > mediaRatio) {
    const width = PREVIEW_HEIGHT * mediaRatio;
    return {
      left: (PREVIEW_WIDTH - width) / 2,
      top: 0,
      width,
      height: PREVIEW_HEIGHT,
    };
  }
  const height = PREVIEW_WIDTH / mediaRatio;
  return {
    left: 0,
    top: (PREVIEW_HEIGHT - height) / 2,
    width: PREVIEW_WIDTH,
    height,
  };
}

export function BatchResults({ items, busy, onOpenItem }: BatchResultsProps) {
  if (!items.length) {
    return (
      <section className="batch-empty" aria-label="Batch results">
        <Files size={34} aria-hidden="true" />
        <strong>{busy ? "Analyzing images" : "Select up to 100 images"}</strong>
        <span>Results and detected sign counts will appear here.</span>
      </section>
    );
  }

  return (
    <section className="batch-results" aria-label="Batch results">
      <header>
        <div>
          <span className="eyebrow">Batch analysis</span>
          <h2>{items.length} images</h2>
        </div>
        <span>{busy ? "Processing" : "Complete"}</span>
      </header>
      <div className="batch-table" role="table" aria-label="Batch inference results" tabIndex={0}>
        <div className="batch-row batch-heading" role="row">
          <span role="columnheader">Image</span>
          <span role="columnheader">Signs</span>
          <span role="columnheader">Runtime</span>
          <span role="columnheader">Result</span>
          <span className="sr-only">Open</span>
        </div>
        {items.map((item) => {
          const result = item.result ?? null;
          const primary = result?.events[0];
          const canOpen = Boolean(result && item.processing && !item.error);

          return (
            <button
              className="batch-row batch-row-button"
              type="button"
              key={`${item.filename}-${item.previewUrl}`}
              disabled={!canOpen}
              onClick={() => onOpenItem(item)}
              aria-label={canOpen ? `Open details for ${item.filename}` : `${item.filename} is still processing`}
            >
              <span className="batch-file">
                <span className="batch-preview">
                  <span
                    className="batch-preview-media"
                    style={result ? previewMediaBox(result) : { inset: 0 }}
                  >
                    <img src={item.previewUrl} alt="" />
                    {result ? (
                      <span
                        className="batch-preview-overlay"
                        aria-label={`${result.events.length} detected signs in ${item.filename}`}
                      >
                        {result.events.map((event) => (
                          <span
                            className={`batch-detection-box severity-${event.severity}`}
                            key={`${event.frame_id}-${event.track_id}`}
                            style={{
                              left: `${(event.bbox.x1 / result.width) * 100}%`,
                              top: `${(event.bbox.y1 / result.height) * 100}%`,
                              width: `${((event.bbox.x2 - event.bbox.x1) / result.width) * 100}%`,
                              height: `${((event.bbox.y2 - event.bbox.y1) / result.height) * 100}%`,
                            }}
                          />
                        ))}
                      </span>
                    ) : null}
                  </span>
                </span>
                <span title={item.filename}>{item.filename}</span>
              </span>
              <strong>{result?.events.length ?? "—"}</strong>
              <span>{result ? `${Math.round(result.latency_ms)} ms` : "—"}</span>
              <span className={item.error ? "batch-outcome error" : "batch-outcome"}>
                {item.error ? (
                  <>
                    <AlertTriangle size={14} />
                    <span>{item.error}</span>
                  </>
                ) : (
                  <>
                    <ScanLine size={14} />
                    <span>{primary ? primary.label : "No sign detected"}</span>
                  </>
                )}
              </span>
              <ChevronRight className="batch-row-chevron" size={18} aria-hidden="true" />
            </button>
          );
        })}
      </div>
    </section>
  );
}
