package io.lc4jlens;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapts a Spring AI {@link org.springframework.ai.embedding.EmbeddingModel} to
 * LangChain4j's {@link dev.langchain4j.model.embedding.EmbeddingModel}, so a Spring AI
 * model can be handed to {@link LensServer} for live query embedding without LensServer
 * needing to know two different embedding-model APIs exist.
 */
final class SpringEmbeddingModelAdapter implements dev.langchain4j.model.embedding.EmbeddingModel {

    private final org.springframework.ai.embedding.EmbeddingModel delegate;

    SpringEmbeddingModelAdapter(org.springframework.ai.embedding.EmbeddingModel delegate) {
        this.delegate = delegate;
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>(textSegments.size());
        for (TextSegment segment : textSegments) {
            float[] vector = delegate.embed(segment.text());
            embeddings.add(Embedding.from(vector));
        }
        return Response.from(embeddings);
    }
}
