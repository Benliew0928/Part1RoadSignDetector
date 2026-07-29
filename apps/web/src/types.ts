export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SignEvent {
  frame_id: number;
  track_id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  severity: "information";
  latency_ms: number;
  evidence: string[];
}

export interface FrameResult {
  frame_id: number;
  width: number;
  height: number;
  mode: "baseline";
  latency_ms: number;
  events: SignEvent[];
  warnings: string[];
}

export interface ProcessingTrace {
  original_jpeg_base64: string;
  raw_masks_png_base64: Record<string, string>;
  clean_masks_png_base64: Record<string, string>;
  contours_jpeg_base64: string;
  final_jpeg_base64: string;
  parameters: {
    hsv_ranges: Record<string, { ranges: { lower: number[]; upper: number[] }[] }>;
    morphology: Record<string, number>;
    minimum_contour_area_percent: number;
    minimum_extent: number;
    minimum_solidity: number;
    maximum_aspect_ratio: number;
    minimum_color_coverage: number;
    preferred_area_percent: number;
    ranking_weights: {
      geometry: number;
      scale: number;
      color_support: number;
    };
    polygon_epsilon_fractions: number[];
    circle_min_circularity: number;
    perspective_ellipse_min_axis_ratio: number;
    minimum_shape_fit_score: number;
    silhouette_refine_context_padding: number;
    triangle_min_fit: number;
    near_square_aspect_ratio: [number, number];
  };
}

export interface ImageInferenceResponse {
  result: FrameResult;
  annotated_jpeg_base64: string;
  processing: ProcessingTrace;
}

export interface BatchInferenceItem {
  filename: string | null;
  result?: FrameResult | null;
  processing?: ProcessingTrace | null;
  error?: string | null;
}

export interface BatchInferenceResponse {
  count: number;
  results: BatchInferenceItem[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  diagnostics: {
    python: string;
    opencv: string;
    cuda_available: boolean;
    healthy: boolean;
  };
  models: {
    mode: "baseline";
    detector: string;
    detector_device: string;
    warnings: string[];
  };
}

export type SourceMode = "image" | "batch";
