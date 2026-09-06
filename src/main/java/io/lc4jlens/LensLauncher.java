package io.lc4jlens;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * One-line entry point: point it at a LangChain4j {@link EmbeddingStore} and
 * the {@link EmbeddingModel} you used to fill it, and it serves a local
 * page plotting every stored chunk in 2D, with query highlighting.
 */
public final class LensLauncher {

    private static final int DEFAULT_PORT = 7477;
    private static final int DEFAULT_MAX_POINTS = 5000;

    private LensLauncher() {}

    public static LensServer launch(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel) {
        return launch(store, embeddingModel, DEFAULT_PORT, DEFAULT_MAX_POINTS);
    }

    public static LensServer launch(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel,
                                     int port, int maxPoints) {
        Map<String, EmbeddingModel> queryModels = new LinkedHashMap<>();
        queryModels.put("store model", embeddingModel);
        return launch(store, embeddingModel, queryModels, port, maxPoints);
    }

    /**
     * Like {@link #launch(EmbeddingStore, EmbeddingModel)}, but lets the UI switch
     * between multiple embedding models when embedding the query text — e.g. to see
     * what happens when a query is embedded with a different model than the one that
     * built the store. {@code queryModels} should include the store's own model (under
     * whatever name you like); its iteration order becomes the dropdown order, and the
     * first entry is the default.
     */
    public static LensServer launch(EmbeddingStore<TextSegment> store, EmbeddingModel embeddingModel,
                                     Map<String, EmbeddingModel> queryModels, int port, int maxPoints) {
        List<EmbeddingMatch<TextSegment>> matches = fetchMatches(store, embeddingModel, maxPoints);
        if (matches.size() < 5) {
            throw new IllegalStateException(
                "Need at least 5 embeddings in the store for t-SNE's perplexity constraint to hold; found "
                    + matches.size());
        }

        List<float[]> vectors = matches.stream().map(m -> m.embedding().vector()).collect(Collectors.toList());
        double[][] layout = TsneProjector.project(vectors);

        List<LensPoint> points = new ArrayList<>(matches.size());
        for (int i = 0; i < matches.size(); i++) {
            EmbeddingMatch<TextSegment> m = matches.get(i);
            String text = m.embedded() != null ? m.embedded().text() : "";
            points.add(new LensPoint(m.embeddingId(), text, m.embedding().vector(), layout[i][0], layout[i][1]));
        }

        // Highlighting half the plotted points isn't a meaningful "top matches" signal —
        // scale topK down for small stores instead of a flat default.
        int topK = Math.max(1, Math.min(5, points.size() / 4));

        // Cluster the raw embeddings (not the 2D layout) so the initial view — before
        // any query — still shows structure instead of a wall of identical dots.
        int k = Math.max(2, Math.min(8, points.size() / 4));
        int[] clusterIds = KMeans.cluster(vectors, k);

        try {
            LensServer server = new LensServer(points, queryModels, topK, clusterIds, port);
            server.start();
            System.out.println("lc4j-lens: plotted " + points.size() + " embeddings at http://localhost:" + server.port());
            return server;
        } catch (IOException e) {
            throw new RuntimeException("Failed to start lc4j-lens server on port " + port, e);
        }
    }

    // ponytail: EmbeddingStore has no universal "list everything" method, so we
    // approximate a full dump by searching nearest-to-a-neutral-probe with a large
    // maxResults. For stores holding more than maxPoints entries this is a sample,
    // not the full set — fine for a local debug view, revisit if a store-specific
    // export (e.g. InMemoryEmbeddingStore#serializeToJson) is worth wiring in.
    private static List<EmbeddingMatch<TextSegment>> fetchMatches(EmbeddingStore<TextSegment> store,
                                                                    EmbeddingModel embeddingModel,
                                                                    int maxPoints) {
        Embedding probe = embeddingModel.embed("lc4j-lens probe query").content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
            .queryEmbedding(probe)
            .maxResults(maxPoints)
            .minScore(0.0)
            .build();
        EmbeddingSearchResult<TextSegment> result = store.search(request);
        return result.matches();
    }
}
