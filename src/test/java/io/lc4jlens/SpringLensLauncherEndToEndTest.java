package io.lc4jlens;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end self-check for the Spring AI adapter, mirroring
 * {@link LensLauncherEndToEndTest}: real embedding model, real in-memory Spring AI
 * VectorStore, real HTTP server on a loopback port. Confirms collect + serve + query
 * works the same way it does through the LangChain4j path.
 */
class SpringLensLauncherEndToEndTest {

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
        TransformersEmbeddingModel embeddingModel = new TransformersEmbeddingModel(MetadataMode.NONE);
        embeddingModel.afterPropertiesSet();

        VectorStore store = SimpleVectorStore.builder(embeddingModel).build();
        for (String doc : DOCS) {
            store.add(List.of(new Document(doc)));
        }

        LensServer server = SpringLensLauncher.launch(store, embeddingModel, 0, 100);
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

            String body = queryResponse.body();
            assertTrue(body.contains("nearestIds"), "expected a nearestIds field in: " + body);
        } finally {
            server.stop();
        }
    }
}
