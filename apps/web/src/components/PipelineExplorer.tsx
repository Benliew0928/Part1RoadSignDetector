import { CheckCircle2, CircleDotDashed, ScanSearch, Shapes } from "lucide-react";
import { useMemo, useState } from "react";

import type { FrameResult, ProcessingTrace } from "../types";

type Stage = "original" | "raw" | "clean" | "contours" | "final";

interface PipelineExplorerProps {
  trace: ProcessingTrace | null;
  fallbackImageUrl: string | null;
  result: FrameResult | null;
  busy: boolean;
}

const stages: { id: Stage; label: string; description: string }[] = [
  { id: "original", label: "Original", description: "Input image" },
  { id: "raw", label: "HSV masks", description: "Colour thresholds" },
  { id: "clean", label: "Clean masks", description: "3×3 morphology" },
  { id: "contours", label: "Contours", description: "Area + geometry" },
  { id: "final", label: "Result", description: "Colour + shape" },
];

function imageSource(base64: string, mimeType: "image/jpeg" | "image/png"): string {
  return `data:${mimeType};base64,${base64}`;
}

function rangeText(ranges: { lower: number[]; upper: number[] }[]): string {
  return ranges
    .map((range) => `H ${range.lower[0]}–${range.upper[0]}, S ${range.lower[1]}–${range.upper[1]}, V ${range.lower[2]}–${range.upper[2]}`)
    .join(" or ");
}

export function PipelineExplorer({
  trace,
  fallbackImageUrl,
  result,
  busy,
}: PipelineExplorerProps) {
  const [stage, setStage] = useState<Stage>("original");
  const [maskColour, setMaskColour] = useState("red");

  const currentStage = stages.find((item) => item.id === stage) ?? stages[0];
  const availableColours = useMemo(
    () => Object.keys(trace?.raw_masks_png_base64 ?? { red: "", blue: "", yellow: "" }),
    [trace],
  );
  const selectedColour = availableColours.includes(maskColour) ? maskColour : availableColours[0] ?? "red";

  let source: string | null = fallbackImageUrl;
  let alt = "Selected road-sign image";
  if (trace) {
    if (stage === "original") {
      source = imageSource(trace.original_jpeg_base64, "image/jpeg");
      alt = "Original uploaded image";
    } else if (stage === "raw") {
      source = imageSource(trace.raw_masks_png_base64[selectedColour], "image/png");
      alt = `${selectedColour} raw HSV mask`;
    } else if (stage === "clean") {
      source = imageSource(trace.clean_masks_png_base64[selectedColour], "image/png");
      alt = `${selectedColour} mask after morphology`;
    } else if (stage === "contours") {
      source = imageSource(trace.contours_jpeg_base64, "image/jpeg");
      alt = "Retained contours and measured geometry";
    } else {
      source = imageSource(trace.final_jpeg_base64, "image/jpeg");
      alt = "Final colour and shape segmentation result";
    }
  }

  return (
    <section className="pipeline-explorer" aria-label="Classical vision processing stages">
      <header className="pipeline-header">
        <div>
          <span className="eyebrow">Part 1 processing evidence</span>
          <h2>HSV segmentation → contour geometry</h2>
        </div>
        <span className="baseline-badge">Classical method</span>
      </header>

      <div className="pipeline-steps" role="tablist" aria-label="Processing stages">
        {stages.map((item, index) => (
          <button
            key={item.id}
            role="tab"
            aria-selected={stage === item.id}
            className={stage === item.id ? "active" : ""}
            onClick={() => setStage(item.id)}
          >
            <span>{index + 1}</span>
            <strong>{item.label}</strong>
            <small>{item.description}</small>
          </button>
        ))}
      </div>

      <div className="pipeline-stage">
        <div className="pipeline-media-panel">
          <header>
            <div>
              <span className="eyebrow">Stage {stages.findIndex((item) => item.id === stage) + 1}</span>
              <h3>{currentStage.label}</h3>
            </div>
            {(stage === "raw" || stage === "clean") && trace ? (
              <div className="mask-colour-tabs" role="tablist" aria-label="Mask colour">
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

          <div className="pipeline-image-wrap">
            {source ? (
              <img src={source} alt={alt} />
            ) : (
              <div className="pipeline-empty">
                <ScanSearch size={34} aria-hidden="true" />
                <strong>{busy ? "Processing image…" : "Choose one image to begin"}</strong>
                <span>The dashboard will show every segmentation stage here.</span>
              </div>
            )}
          </div>
        </div>

        <aside className="pipeline-details">
          <section>
            <header>
              <Shapes size={17} aria-hidden="true" />
              <h3>Measured features</h3>
            </header>
            {result?.events.length ? (
              <ul className="feature-list">
                {result.events.map((event) => (
                  <li key={`${event.frame_id}-${event.track_id}`}>
                    <strong>{event.label}</strong>
                    <span>{event.evidence.filter((item) => /area_ratio|circularity|vertices|color_coverage|scale_evidence/.test(item)).join(" · ")}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No retained contour yet. The raw and clean masks still show why.</p>
            )}
          </section>

          <section>
            <header>
              <CircleDotDashed size={17} aria-hidden="true" />
              <h3>Fixed parameters</h3>
            </header>
            {trace ? (
              <dl className="parameter-list">
                {Object.entries(trace.parameters.hsv_ranges).map(([colour, settings]) => (
                  <div key={colour}>
                    <dt>{colour} HSV</dt>
                    <dd>{rangeText(settings.ranges)}</dd>
                  </div>
                ))}
                <div>
                  <dt>Mask cleaning</dt>
                  <dd>3×3 opening; red also 3×3 closing</dd>
                </div>
                <div>
                  <dt>Contours</dt>
                  <dd>≥ {trace.parameters.minimum_contour_area_percent.toFixed(2)}% image area; hull ε = {trace.parameters.polygon_epsilon_fractions.join(", ")}P</dd>
                </div>
                <div>
                  <dt>Candidate filter</dt>
                  <dd>extent ≥ {trace.parameters.minimum_extent}; solidity ≥ {trace.parameters.minimum_solidity}; AR ≤ {trace.parameters.maximum_aspect_ratio}</dd>
                </div>
                <div>
                  <dt>Candidate ranking</dt>
                  <dd>geometry {trace.parameters.ranking_weights.geometry}; scale {trace.parameters.ranking_weights.scale}; mask support {trace.parameters.ranking_weights.color_support}</dd>
                </div>
                <div>
                  <dt>Shape rule</dt>
                  <dd>triangle fit ≥ {trace.parameters.triangle_min_fit}; circle C ≥ {trace.parameters.circle_min_circularity}</dd>
                </div>
              </dl>
            ) : (
              <p>Parameters are shown after the first analysis.</p>
            )}
          </section>

          <section className="fairness-note">
            <CheckCircle2 size={17} aria-hidden="true" />
            <p>Uses no training images, filename rules, OCR, or deep-learning model. Every intermediate result is visible.</p>
          </section>
        </aside>
      </div>
    </section>
  );
}
