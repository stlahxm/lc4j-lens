package io.lc4jlens;

import java.util.List;
import java.util.Random;

/**
 * Minimal Lloyd's-algorithm k-means, used only to color the initial scatter
 * plot by topic before any query is run — so the view isn't a wall of
 * identical blue dots. Not a general clustering library: fixed iteration
 * count, no k-selection heuristic, cosine-ish behavior via raw Euclidean
 * distance on (roughly unit-norm) embedding vectors is good enough for
 * this purpose.
 */
final class KMeans {

    private static final int ITERATIONS = 25;

    private KMeans() {}

    /** Returns a cluster id (0..k-1) per input vector, same order as the input. */
    static int[] cluster(List<float[]> vectors, int k) {
        int n = vectors.size();
        k = Math.max(1, Math.min(k, n));
        int dim = vectors.get(0).length;

        Random random = new Random(42); // fixed seed: stable coloring across restarts
        double[][] centroids = new double[k][dim];
        boolean[] taken = new boolean[n];
        for (int c = 0; c < k; c++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (taken[idx]);
            taken[idx] = true;
            centroids[c] = toDouble(vectors.get(idx));
        }

        int[] assignment = new int[n];
        for (int iter = 0; iter < ITERATIONS; iter++) {
            for (int i = 0; i < n; i++) {
                assignment[i] = nearestCentroid(toDouble(vectors.get(i)), centroids);
            }
            double[][] sums = new double[k][dim];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                double[] v = toDouble(vectors.get(i));
                int c = assignment[i];
                counts[c]++;
                for (int d = 0; d < dim; d++) sums[c][d] += v[d];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) continue; // keep previous centroid for an emptied cluster
                for (int d = 0; d < dim; d++) centroids[c][d] = sums[c][d] / counts[c];
            }
        }
        return assignment;
    }

    private static int nearestCentroid(double[] v, double[][] centroids) {
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            double dist = squaredDistance(v, centroids[c]);
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i];
        return out;
    }
}
