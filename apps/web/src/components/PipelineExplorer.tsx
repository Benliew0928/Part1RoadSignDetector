import {
  ArrowLeft,
  BadgeCheck,
  ChevronDown,
  Circle,
  CircleDotDashed,
  Cpu,
  Gauge,
  ImagePlus,
  ScanSearch,
  Shapes,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { FrameResult, ProcessingTrace } from "../types";

type Stage = "original" | "raw" | "clean" | "contours" | "final";

interface PipelineExplorerProps {
  trace: ProcessingTrace | null;
  fallbackImageUrl: string | null;
  result: FrameResult | null;
  busy: boolean;
  runtimeLabel: string;
  onChooseImage: () => void;
  onReturnToBatch?: () => void;
}

const stages: { id: Stage; label: string; description: string }[] = [
  { id: "original", label: "Original", description: "Uploaded image" },
  { id: "raw", label: "Colour mask", description: "Colour isolation" },
  { id: "clean", label: "Clean mask", description: "Refined mask" },
  { id: "contours", label: "Contour", description: "Shape outline" },
  { id: "final", label: "Result", description: "Detection result" },
];

function imageSource(base64: string, mimeType: "image/jpeg" | "image/png"): string {
  return `data:${mimeType};base64,${base64}`;
}

function rangeText(ranges: { lower: number[]; upper: number[] }[]): string {
  return ranges
    .map((range) => `H ${range.lower[0]}–${range.upper[0]}, S ${range.lower[1]}–${range.upper[1]}, V ${range.lower[2]}–${range.upper[2]}`)
    .join(" or ");
}

function characteristics(label: string | undefined): { colour: string; shape: string; title: string } {
  const normalised = label?.toLowerCase() ?? "";
  const colour = ["red", "blue", "yellow"].find((item) => normalised.includes(item)) ?? "Detected";
  const shape = normalised.includes("circle") || normalised.includes("round")
    ? "Circular"
    : normalised.includes("triangle")
      ? "Triangular"
      : normalised.includes("octagon")
        ? "Octagonal"
        : normalised.includes("square") || normalised.includes("rectangle")
          ? "Quadrilateral"
          : "Shape detected";
  const title = colour === "Detected" ? "Road sign detected" : `${colour[0].toUpperCase()}${colour.slice(1)} ${shape.toLowerCase()} sign`;

  return { colour, shape, title };
}

export function PipelineExplorer({
  trace,
  fallbackImageUrl,
  result,
  busy,
  runtimeLabel,
  onChooseImage,
  onReturnToBatch,
}: PipelineExplorerProps) {
  const [stage, setStage] = useState<Stage>("original");
  const [maskColour, setMaskColour] = useState("blue");

  useEffect(() => {
    const stageTimer = window.setTimeout(() => setStage(trace ? "final" : "original"), 0);
    return () => window.clearTimeout(stageTimer);
  }, [trace]);

  const availableColours = useMemo(
    () => Object.keys(trace?.raw_masks_png_base64 ?? { red: "", blue: "", yellow: "" }),
    [trace],
  );
  const selectedColour = availableColours.includes(maskColour) ? maskColour : availableColours[0] ?? "blue";
  const primaryEvent = result?.events[0];
  const detection = characteristics(primaryEvent?.label);
  const currentStage = stages.find((item) => item.id === stage) ?? stages[0];

  const stageImage = (stageId: Stage): { src: string | null; alt: string } => {
    if (!trace) {
      return { src: stageId === "original" ? fallbackImageUrl : null, alt: "Selected road-sign image" };
    }
    if (stageId === "original") return { src: imageSource(trace.original_jpeg_base64, "image/jpeg"), alt: "Original uploaded image" };
    if (stageId === "raw") return { src: imageSource(trace.raw_masks_png_base64[selectedColour], "image/png"), alt: `${selectedColour} colour mask` };
    if (stageId === "clean") return { src: imageSource(trace.clean_masks_png_base64[selectedColour], "image/png"), alt: `${selectedColour} clean mask` };
    if (stageId === "contours") return { src: imageSource(trace.contours_jpeg_base64, "image/jpeg"), alt: "Retained shape contours" };
    return { src: imageSource(trace.final_jpeg_base64, "image/jpeg"), alt: "Final colour and shape detection result" };
  };

  const activeImage = stageImage(stage);
  const hasAnalysis = Boolean(trace || fallbackImageUrl);

  if (!hasAnalysis) {
    return (
      <section className="detection-empty" aria-label="Image analysis">
        <div className="detection-empty-icon"><ScanSearch size={34} aria-hidden="true" /></div>
        <span className="eyebrow">Road-sign analysis</span>
        <h2>{busy ? "Analysing your image" : "Start with a road-sign image"}</h2>
        <p>{busy ? "The result and visual evidence will appear here." : "Upload one image to identify its colour and geometric shape."}</p>
        {!busy ? (
          <button className="primary-action" onClick={onChooseImage}>
            <ImagePlus size={18} />
            Upload image
          </button>
        ) : null}
      </section>
    );
  }

  return (
    <section className="detection-explorer" aria-label="Road-sign analysis result">
      {onReturnToBatch ? (
        <button className="back-to-batch" type="button" onClick={onReturnToBatch}>
          <ArrowLeft size={17} aria-hidden="true" />
          Back to batch results
        </button>
      ) : null}
      <div className="detection-result-grid">
        <section className="detection-media-card">
          <header>
            <div>
              <span className="eyebrow">{trace ? "Analysis result" : "Uploaded image"}</span>
              <h2>{currentStage.label}</h2>
            </div>
            {(stage === "raw" || stage === "clean") && trace ? (
              <div className="mask-selector" role="tablist" aria-label="Mask colour">
                {availableColours.map((colour) => (
                  <button
                    key={colour}
                    role="tab"
                    aria-selected={selectedColour === colour}
                    className={selectedColour === colour ? `active ${colour}` : colour}
                    onClick={() => setMaskColour(colour)}
                  >
                    {colour}
                  </button>
                ))}
              </div>
            ) : null}
          </header>
          <div className="detection-image-wrap">
            {activeImage.src ? <img src={activeImage.src} alt={activeImage.alt} /> : <ScanSearch size={34} aria-hidden="true" />}
          </div>
        </section>

        <aside className="sign-summary-card">
          <div className="summary-emblem"><BadgeCheck size={45} aria-hidden="true" /></div>
          {primaryEvent ? (
            <>
              <span className="eyebrow">Detection</span>
              <h2>{detection.title}</h2>
              <strong className="confidence-score">{Math.round(primaryEvent.confidence * 100)}%</strong>
              <span className="confidence-label">confidence</span>
              <div className="summary-divider" />
              <span className="eyebrow">Detected characteristics</span>
              <dl className="characteristic-list">
                <div><dt><Circle className={`colour-dot ${detection.colour}`} size={27} aria-hidden="true" />Colour</dt><dd>{detection.colour}</dd></div>
                <div><dt><Circle className="shape-dot" size={27} aria-hidden="true" />Shape</dt><dd>{detection.shape}</dd></div>
              </dl>
            </>
          ) : (
            <>
              <span className="eyebrow">Detection complete</span>
              <h2>No road sign detected</h2>
              <p>Try an image where the sign is larger, clearer, and well lit.</p>
            </>
          )}
          <button className="primary-action full-width" onClick={onChooseImage}>
            <ImagePlus size={18} />
            Analyse another image
          </button>
        </aside>
      </div>

      {trace ? (
        <details className="visual-evidence" open>
          <summary>
            <span>
              <span className="eyebrow">Transparent analysis</span>
              <strong>How it was recognised</strong>
            </span>
            <ChevronDown size={20} aria-hidden="true" />
          </summary>
          <div className="evidence-strip" role="tablist" aria-label="Visual evidence stages">
            {stages.map((item, index) => {
              const preview = stageImage(item.id);
              return (
                <button
                  key={item.id}
                  role="tab"
                  aria-selected={stage === item.id}
                  className={stage === item.id ? "active" : ""}
                  onClick={() => setStage(item.id)}
                >
                  <span>{index + 1}. {item.label}</span>
                  <div>{preview.src ? <img src={preview.src} alt="" /> : null}</div>
                </button>
              );
            })}
          </div>
        </details>
      ) : null}

      <details className="technical-details">
        <summary>
          <span>
            <span className="eyebrow">For assessment and troubleshooting</span>
            <strong>Technical details</strong>
          </span>
          <ChevronDown size={20} aria-hidden="true" />
        </summary>
        <div className="technical-content">
          <section className="technical-metrics">
            <div><Cpu size={17} aria-hidden="true" /><span>Runtime</span><strong>{runtimeLabel}</strong></div>
            <div><Gauge size={17} aria-hidden="true" /><span>Latency</span><strong>{result ? `${Math.round(result.latency_ms)} ms` : "—"}</strong></div>
            <div><Shapes size={17} aria-hidden="true" /><span>Candidates</span><strong>{result?.events.length ?? 0}</strong></div>
          </section>
          {trace ? (
            <section className="technical-parameters">
              <header><CircleDotDashed size={18} aria-hidden="true" /><h3>Fixed parameters</h3></header>
              <dl>
                {Object.entries(trace.parameters.hsv_ranges).map(([colour, settings]) => (
                  <div key={colour}><dt>{colour} HSV</dt><dd>{rangeText(settings.ranges)}</dd></div>
                ))}
                <div><dt>Mask cleaning</dt><dd>3×3 opening; red also 3×3 closing</dd></div>
                <div><dt>Contours</dt><dd>≥ {trace.parameters.minimum_contour_area_percent.toFixed(2)}% image area; hull ε = {trace.parameters.polygon_epsilon_fractions.join(", ")}P</dd></div>
                <div><dt>Candidate filter</dt><dd>extent ≥ {trace.parameters.minimum_extent}; solidity ≥ {trace.parameters.minimum_solidity}; AR ≤ {trace.parameters.maximum_aspect_ratio}</dd></div>
                <div><dt>Candidate ranking</dt><dd>geometry {trace.parameters.ranking_weights.geometry}; scale {trace.parameters.ranking_weights.scale}; mask support {trace.parameters.ranking_weights.color_support}</dd></div>
                <div><dt>Shape rule</dt><dd>best circle/ellipse, triangle, quadrilateral or octagon fit ≥ {trace.parameters.minimum_shape_fit_score}</dd></div>
                <div><dt>Perspective support</dt><dd>ellipse axis ratio ≥ {trace.parameters.perspective_ellipse_min_axis_ratio}; local mask context {trace.parameters.silhouette_refine_context_padding}px</dd></div>
              </dl>
            </section>
          ) : null}
        </div>
      </details>
    </section>
  );
}
