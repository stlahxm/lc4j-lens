package io.lc4jlens;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class KMeansTest {

    @Test
    void separatesTwoObviouslyDistinctGroups() {
        List<float[]> vectors = List.of(
            new float[] {0f, 0f}, new float[] {0.1f, -0.1f}, new float[] {-0.1f, 0.1f},
            new float[] {100f, 100f}, new float[] {100.1f, 99.9f}, new float[] {99.9f, 100.1f}
        );

        int[] assignment = KMeans.cluster(vectors, 2);

        assertEquals(assignment[0], assignment[1]);
        assertEquals(assignment[0], assignment[2]);
        assertEquals(assignment[3], assignment[4]);
        assertEquals(assignment[3], assignment[5]);
        assertNotEquals(assignment[0], assignment[3]);
    }
}
