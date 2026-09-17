package io.lc4jlens;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Parallel entry point for Spring AI: point it at a Spring AI {@link VectorStore} and
 * the {@link EmbeddingModel} you used to fill it, and it serves the same local 2D
 * embedding-space plot that {@link LensLauncher} serves for LangChain4j, opening
 * lc4j-lens up to the Spring AI half of the Java RAG ecosystem. The existing
 * LangChain4j entry point is untouched by this class.
 */
public final class SpringLensLauncher {

    private static final int DEFAULT_PORT = 7477;
    private static final int DEFAULT_MAX_POINTS = 5000;

    private SpringLensLauncher() {}

    public static LensServer launch(VectorStore store, EmbeddingModel embeddingModel) {
        return launch(store, embeddingModel, DEFAULT_PORT, DEFAULT_MAX_POINTS);
    }

    public static LensServer launch(VectorStore store, EmbeddingModel embeddingModel, int port, int maxPoints) {
        Map<String, EmbeddingModel> queryModels = new LinkedHashMap<>();
        queryModels.put("store model", embeddingModel);
        return launch(store, embeddingModel, queryModels, port, maxPoints);
    }

    /**
     * Like {@link #launch(VectorStore, EmbeddingModel)}, but lets the UI switch between
     * multiple embedding models when embedding the query text — see the LangChain4j
     * overload of the same name for the full rationale. {@code queryModels} should
     * include the store's own model; its iteration order becomes the dropdown order,
     * and the first entry is the default.
     */
    public static LensServer launch(VectorStore store, EmbeddingModel embeddingModel,
                                     Map<String, EmbeddingModel> queryModels, int port, int maxPoints) {
        List<Document> documents = fetchSample(store, maxPoints);
        if (documents.size() < 5) {
            throw new IllegalStateException(
                "Need at least 5 embeddings in the store for t-SNE's perplexity constraint to hold; found "
                    + documents.size());
        }

        // similaritySearch(SearchRequest) returns Documents, not their stored vectors — Spring
        // AI's VectorStore contract doesn't expose the embedding it computed internally. Since
        // the caller already told us which EmbeddingModel filled this store, we re-embed each
        // returned document's text with that same model; a deterministic model on the same
        // text reproduces the same vector, so this is equivalent to reading it back, without
        // depending on any store-specific internals.
        List<float[]> vectors = documents.stream()
            .map(doc -> embeddingModel.embed(doc.getText()))
            .collect(Collectors.toList());
        double[][] layout = TsneProjector.project(vectors);

        List<LensPoint> points = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            points.add(new LensPoint(doc.getId(), doc.getText(), vectors.get(i), layout[i][0], layout[i][1]));
        }

        int topK = Math.max(1, Math.min(5, points.size() / 4));
        int k = Math.max(2, Math.min(8, points.size() / 4));
        int[] clusterIds = KMeans.cluster(vectors, k);

        Map<String, dev.langchain4j.model.embedding.EmbeddingModel> bridgedModels = new LinkedHashMap<>();
        for (Map.Entry<String, EmbeddingModel> entry : queryModels.entrySet()) {
            bridgedModels.put(entry.getKey(), new SpringEmbeddingModelAdapter(entry.getValue()));
        }

        try {
            LensServer server = new LensServer(points, bridgedModels, topK, clusterIds, port);
            server.start();
            System.out.println("lc4j-lens: plotted " + points.size() + " embeddings at http://localhost:" + server.port());
            return server;
        } catch (IOException e) {
            throw new RuntimeException("Failed to start lc4j-lens server on port " + port, e);
        }
    }

    // SimpleVectorStore validates similarityThreshold into [0,1] and filters with a literal
    // score >= threshold comparison against RAW (possibly negative) cosine similarity — so
    // even the lowest allowed threshold (0.0) silently drops any document whose cosine
    // similarity to the probe text happens to be negative. A single neutral probe therefore
    // isn't a reliable full-dump approximation (see LensLauncher.fetchMatches for the
    // LangChain4j side of the "no universal list-everything" constraint this works around).
    // Querying with several topically distinct probes and taking the union sidesteps it: a
    // document would need negative cosine similarity to every probe to be missed, which is
    // very unlikely across enough of them. Unlike InMemoryEmbeddingStore on the LangChain4j
    // side, SimpleVectorStore exposes no public exact count/listing to special-case instead.
    private static final List<String> PROBES = List.of(
        "general information",
        "technology and software",
        "people and daily life",
        "science and nature",
        "history and places",
        "food and everyday objects"
    );

    private static List<Document> fetchSample(VectorStore store, int maxPoints) {
        Map<String, Document> byId = new LinkedHashMap<>();
        for (String probe : PROBES) {
            SearchRequest request = SearchRequest.builder()
                .query(probe)
                .topK(maxPoints)
                .similarityThreshold(0.0)
                .build();
            for (Document doc : store.similaritySearch(request)) {
                byId.putIfAbsent(doc.getId(), doc);
                if (byId.size() >= maxPoints) break;
            }
            if (byId.size() >= maxPoints) break;
        }
        return new ArrayList<>(byId.values());
    }
}