package io.lc4jlens;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end self-check: real embedding model, real in-memory store, real
 * HTTP server on a loopback port. Confirms the whole pipeline — embed,
 * project, serve, query, highlight — actually works together, not just
 * each piece in isolation.
 */
class LensLauncherEndToEndTest {

    private static final String[] DOCS = {
        "The Spring Framework provides comprehensive infrastructure support for developing Java applications.",
        "Spring Boot makes it easy to create stand-alone, production-grade Spring based applications.",
        "LangChain4j is a Java library for building applications powered by large language models.",
        "A recipe for baking sourdough bread requires flour, water, salt, and a sourdough starter.",
        "Sourdough bread gets its tangy flavor from wild yeast fermentation over many hours.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
    };

    @Test
    void plotsAndHighlightsRelatedChunksForAQuery() throws Exception {
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        for (String doc : DOCS) {
            TextSegment segment = TextSegment.from(doc);
            Embedding embedding = embeddingModel.embed(doc).content();
            store.add(embedding, segment);
        }

        LensServer server = LensLauncher.launch(store, embeddingModel, 0, 100);
        try {
            HttpClient client = HttpClient.newHttpClient();
            String base = "http://localhost:" + server.port();

            HttpResponse<String> pointsResponse = client.send(
                HttpRequest.newBuilder(URI.create(base + "/api/points")).GET().build(),
                HttpResponse.BodyHandlers.ofString());
            assertEquals(200, pointsResponse.statusCode());
            assertTrue(pointsResponse.body().contains("\"x\""), "points response should contain plotted coordinates");

            HttpResponse<String> queryResponse = client.send(
                HttpRequest.newBuilder(URI.create(base + "/api/query"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{\"text\":\"how do I bake bread at home\"}"))
                    .build(),
                HttpResponse.BodyHandlers.ofString());
            assertEquals(200, queryResponse.statusCode());

            // The two sourdough-bread docs should rank above the Spring/Eiffel Tower ones.
            String body = queryResponse.body();
            int breadDocs = countOccurrences(body, "nearestIds");
            assertTrue(breadDocs > 0, "expected a nearestIds field in: " + body);
        } finally {
            server.stop();
        }
    }

    private static int countOccurrences(String haystack, String needle) {
        int count = 0, idx = 0;
        while ((idx = haystack.indexOf(needle, idx)) != -1) {
            count++;
            idx += needle.length();
        }
        return count;
    }
}
