import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import App from "./App";

vi.mock("./api", () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: "ok",
    version: "part1-hsv-contour",
    diagnostics: { python: "3.11", opencv: "4.13", cuda_available: false, healthy: true },
    models: {
      mode: "baseline",
      detector: "HSV masks, morphology and contour geometry",
      detector_device: "cpu",
      warnings: [],
    },
  }),
  inferImage: vi.fn(),
  inferBatch: vi.fn(),
}));

describe("Part 1 dashboard", () => {
  it("offers only the assignment image and batch inputs", async () => {
    render(<App />);

    expect(await screen.findByText("System ready")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Batch" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose image" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Camera" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Video" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Phone" })).not.toBeInTheDocument();
  });
});
