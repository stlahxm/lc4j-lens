# Contributing to lc4j-lens

This is a young, small project — issues and small, focused PRs are welcome.
Please keep PRs single-purpose (one feature or fix per PR, no drive-by
reformatting) and include a test where the change touches non-trivial logic.

## Good first issues

These are real, unclaimed gaps — not busywork:

- **Spring AI `VectorStore` adapter.** `LensLauncher` currently only accepts
  a LangChain4j `EmbeddingStore`. A parallel entry point for Spring AI's
  `VectorStore` would open this up to the other half of the Java RAG
  ecosystem. Spring AI's `VectorStore` has a `similaritySearch(SearchRequest)`
  method that plays the same role as `EmbeddingStore#search`.
- **Native "exact count/listing" support for more store types.**
  `LensLauncher.fetchMatches` already special-cases `InMemoryEmbeddingStore`
  (using its public `size()` to request every entry exactly, no sampling).
  Other stores (e.g. `PgVectorEmbeddingStore`) may expose a way to get an
  exact count or full listing too — adding an `instanceof` branch per store
  type (see the `ponytail:` comment in `LensLauncher.java`) would remove the
  approximation for those as well.
- **Additional local embedding models in the demo.** `Demo.java` currently
  registers AllMiniLM and BGE-small. Adding e.g. E5-small as a third option
  would strengthen the model-comparison feature without much code.
- **Persist zoom/pan state across queries.** Right now the view resets to
  fit-to-window on every query; some users may prefer it to stay put.

## Reporting a bug

Include: the LangChain4j/Spring AI version, the embedding model in use, and
(if possible) a minimal reproduction — a small `EmbeddingStore` with a
handful of entries that shows the issue.

## Running the test suite

```
mvn test
```

`KMeansTest`-style unit tests and `LensLauncherEndToEndTest` (a real local
embedding model + real HTTP server, no mocks) both need to pass before a
PR is merged.
