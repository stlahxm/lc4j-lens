package io.lc4jlens;

/** One plotted embedding: its id, source text (truncated for display), and 2D coordinate. */
public record LensPoint(String id, String text, float[] vector, double x, double y) {
}
