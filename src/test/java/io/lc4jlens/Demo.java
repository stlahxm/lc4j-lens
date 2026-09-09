package io.lc4jlens;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.e5smallv2q.E5SmallV2QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Runnable demo: embeds a handful of sample documents locally (no API key),
 * starts the lens server, and blocks so you can open the browser yourself.
 * Not a test — run it directly, e.g. `mvn -q exec:java -Pdemo`.
 */
public class Demo {

    private static final String[] DOCS = {
        // Spring / Java cluster
        "The Spring Framework provides comprehensive infrastructure support for developing Java applications.",
        "Spring Boot makes it easy to create stand-alone, production-grade Spring based applications.",
        "Dependency injection in Spring lets you wire beans without manual instantiation.",
        "Spring MVC handles incoming HTTP requests and maps them to controller methods.",
        "The Spring container manages the full lifecycle of beans, from creation to destruction.",
        // LLM / RAG cluster
        "LangChain4j is a Java library for building applications powered by large language models.",
        "Retrieval-augmented generation combines a search step with an LLM's generation step.",
        "Vector embeddings capture the semantic meaning of text as points in high-dimensional space.",
        "A vector store lets you search for text chunks that are semantically similar to a query.",
        "Prompt engineering shapes how a large language model interprets and responds to input.",
        // Baking cluster
        "A recipe for baking sourdough bread requires flour, water, salt, and a sourdough starter.",
        "Sourdough bread gets its tangy flavor from wild yeast fermentation over many hours.",
        "Kneading dough develops gluten, which gives bread its chewy texture.",
        "A Dutch oven traps steam, giving artisan bread a crisp, crackling crust.",
        // Unrelated cluster
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "Paris is the capital of France and home to many famous landmarks.",
        "The Great Wall of China stretches for thousands of kilometers across northern China.",
        "Mount Everest is the tallest mountain above sea level on Earth."
    };

    public static void main(String[] args) throws Exception {
        System.out.println("Embedding " + DOCS.length + " sample documents locally...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        for (String doc : DOCS) {
            Embedding embedding = embeddingModel.embed(doc).content();
            store.add(embedding, TextSegment.from(doc));
        }

        // Two DIFFERENT local models, same 384-dim output — lets the UI show what
        // happens when a query is embedded with a model other than the one that
        // actually built the store (a real, easy-to-hit RAG misconfiguration).
        Map<String, EmbeddingModel> queryModels = new LinkedHashMap<>();
        queryModels.put("AllMiniLM (same model used to build the store)", embeddingModel);
        queryModels.put("BGE-small (different model, same dimensions)", new BgeSmallEnV15QuantizedEmbeddingModel());
        queryModels.put("E5-small (different model, same dimensions)", new E5SmallV2QuantizedEmbeddingModel());

        LensServer server = LensLauncher.launch(store, embeddingModel, queryModels, 7477, 100);
        System.out.println("Open http://localhost:" + server.port() + " in your browser.");
        System.out.println("Try queries like: \"how do I bake bread\" or \"tell me about Spring dependency injection\".");
        System.out.println("Switch the query model dropdown to BGE-small to see retrieval degrade with a mismatched embedding model.");
        System.out.println("Press Ctrl+C to stop.");
        Thread.currentThread().join();
    }
}
