package io.lc4jlens;

import de.javagl.tsne.Tsne;

import java.util.List;

/**
 * Projects embedding vectors to 2D with t-SNE instead of PCA. Unlike PCA
 * (a linear projection along directions of max global variance), t-SNE
 * specifically optimizes for preserving local neighbor structure — points
 * that are close by cosine similarity in the original space end up close
 * in the 2D plot. That correlation is the entire value proposition of this
 * tool, so the cheaper PCA route was the wrong "lazy" choice here.
 *
 * Trade-off: t-SNE has no out-of-sample extension. There is no way to
 * project a single new point into an existing t-SNE layout — the whole
 * batch must be re-fit together, and the resulting coordinate frame is
 * arbitrary each run (points may end up rotated/mirrored/rescaled between
 * calls). So every query re-fits ALL points, not just the query vector.
 * For a local debugging tool operated by one developer at a time, that
 * cost (roughly O(n log n) per query, capped by maxIterations) is
 * acceptable up to a few thousand points; well beyond that, this
 * approach would need to move to an incremental/approximate method.
 */
final class TsneProjector {

    private static final int MAX_ITERATIONS = 500;

    private TsneProjector() {}

    /** Projects all given vectors together; index i of the result corresponds to vectors.get(i). */
    static double[][] project(List<float[]> vectors) {
        int n = vectors.size();
        if (n < 5) {
            throw new IllegalArgumentException(
                "t-SNE needs at least 5 points to satisfy its perplexity constraint, got " + n);
        }

        double[][] input = new double[n][];
        for (int i = 0; i < n; i++) {
            input[i] = toDouble(vectors.get(i));
        }

        Tsne tsne = new Tsne();
        tsne.setOutputDims(2);
        tsne.setMaxIterations(MAX_ITERATIONS);
        // Constraint enforced by the library: (n - 1) > 3 * perplexity. Stay comfortably under it.
        tsne.setPerplexity(Math.min(30.0, (n - 1) / 4.0));
        // The library's internal PCA pre-reduction step needs more samples than target
        // dimensions (default target: 55); with the small point counts this tool is
        // typically run on, that check fails before t-SNE even starts. Skip it and run
        // t-SNE directly on the raw embedding dimensionality instead.
        tsne.setInitialDims(0);
        tsne.setMessageConsumer(null);
        tsne.setProgressConsumer(null);

        return tsne.run(input);
    }

    private static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i];
        return out;
    }
}
