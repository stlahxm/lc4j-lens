package io.lc4jlens;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import dev.langchain4j.model.embedding.EmbeddingModel;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Serves the lc4j-lens scatter-plot UI over a plain JDK HTTP server — no
 * servlet container, no Spring context, nothing to configure. Local-only
 * debug tool for one developer at a time.
 *
 * Because t-SNE has no out-of-sample extension (see {@link TsneProjector}),
 * every query re-fits the whole layout (existing points + the query vector)
 * and the response carries the refreshed coordinates for everything, not
 * just the query. The frontend replaces its whole point set each time.
 */
public final class LensServer {

    // Empty map: query highlighting disabled entirely. Otherwise, one or more named models
    // the frontend can choose between for embedding the query text — see class javadoc.
    private final Map<String, EmbeddingModel> queryModels;
    private final int topK;
    private final int[] clusterIds; // index-aligned with `points`; stable across t-SNE re-fits (position never reorders)
    private final HttpServer server;

    private volatile List<LensPoint> points; // guarded by the intrinsic lock during re-fits

    LensServer(List<LensPoint> points, Map<String, EmbeddingModel> queryModels, int topK, int[] clusterIds, int port)
            throws IOException {
        this.points = points;
        this.queryModels = new LinkedHashMap<>(queryModels);
        this.topK = topK;
        this.clusterIds = clusterIds;
        // Bind to loopback only: an InetSocketAddress(port) with no host binds the wildcard
        // address (0.0.0.0), which would expose embedding content to anyone on the same network.
        this.server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        server.createContext("/", guarded(this::serveIndex));
        server.createContext("/api/points", guarded(this::servePoints));
        server.createContext("/api/models", guarded(this::serveModels));
        server.createContext("/api/query", guarded(this::serveQuery));
        server.setExecutor(null);
    }

    public int port() {
        return server.getAddress().getPort();
    }

    public void start() {
        server.start();
    }

    public void stop() {
        server.stop(0);
    }

    private void serveIndex(HttpExchange ex) throws IOException {
        byte[] html;
        try (InputStream in = LensServer.class.getResourceAsStream("/lens-ui.html")) {
            html = in.readAllBytes();
        }
        respond(ex, 200, "text/html; charset=utf-8", html);
    }

    private void servePoints(HttpExchange ex) throws IOException {
        respond(ex, 200, "application/json; charset=utf-8", pointsJson(points, null).getBytes(StandardCharsets.UTF_8));
    }

    private void serveModels(HttpExchange ex) throws IOException {
        StringBuilder json = new StringBuilder("[");
        int i = 0;
        for (String name : queryModels.keySet()) {
            if (i++ > 0) json.append(',');
            json.append('"').append(Json.escape(name)).append('"');
        }
        json.append(']');
        respond(ex, 200, "application/json; charset=utf-8", json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private synchronized void serveQuery(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            respond(ex, 405, "text/plain", "POST only".getBytes(StandardCharsets.UTF_8));
            return;
        }
        if (queryModels.isEmpty()) {
            respond(ex, 501, "application/json",
                "{\"error\":\"no embedding model configured; query highlighting disabled\"}".getBytes(StandardCharsets.UTF_8));
            return;
        }

        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String text;
        try {
            text = Json.extractStringField(body, "text");
        } catch (IllegalArgumentException e) {
            respond(ex, 400, "application/json",
                ("{\"error\":\"" + Json.escape(e.getMessage()) + "\"}").getBytes(StandardCharsets.UTF_8));
            return;
        }

        String modelName;
        try {
            modelName = Json.extractStringField(body, "model");
        } catch (IllegalArgumentException e) {
            modelName = queryModels.keySet().iterator().next(); // default: first registered model
        }
        EmbeddingModel embeddingModel = queryModels.get(modelName);
        if (embeddingModel == null) {
            respond(ex, 400, "application/json",
                ("{\"error\":\"unknown model \\\"" + Json.escape(modelName) + "\\\"\"}").getBytes(StandardCharsets.UTF_8));
            return;
        }

        float[] queryVector = embeddingModel.embed(text).content().vector();
        int storeDim = points.get(0).vector().length;
        if (queryVector.length != storeDim) {
            String msg = "Query model \"" + modelName + "\" produces " + queryVector.length
                + "-dimensional vectors, but the store's embeddings are " + storeDim
                + "-dimensional. Similarity can't be computed across different vector"
                + " spaces — the store and the query MUST use the same embedding model.";
            respond(ex, 200, "application/json; charset=utf-8",
                ("{\"error\":\"" + Json.escape(msg) + "\"}").getBytes(StandardCharsets.UTF_8));
            return;
        }
        List<LensPoint> nearest = nearestByCosine(points, queryVector, topK);
        List<Double> nearestScores = nearest.stream().map(p -> cosine(queryVector, p.vector())).toList();
        double[] allScores = new double[points.size()];
        for (int i = 0; i < points.size(); i++) allScores[i] = cosine(queryVector, points.get(i).vector());

        // Re-fit t-SNE over everything + the query vector — see class javadoc for why.
        List<float[]> vectors = new ArrayList<>(points.size() + 1);
        for (LensPoint p : points) vectors.add(p.vector());
        vectors.add(queryVector);
        double[][] refit = TsneProjector.project(vectors);

        List<LensPoint> updated = new ArrayList<>(points.size());
        for (int i = 0; i < points.size(); i++) {
            LensPoint p = points.get(i);
            updated.add(new LensPoint(p.id(), p.text(), p.vector(), refit[i][0], refit[i][1]));
        }
        this.points = updated;
        double[] queryXY = refit[refit.length - 1];

        boolean isStoreModel = modelName.equals(queryModels.keySet().iterator().next());
        StringBuilder json = new StringBuilder("{");
        json.append("\"modelUsed\":\"").append(Json.escape(modelName)).append("\",");
        json.append("\"matchesStoreModel\":").append(isStoreModel).append(',');
        json.append("\"points\":").append(pointsJson(updated, allScores)).append(',');
        json.append("\"queryPoint\":{\"x\":").append(queryXY[0]).append(",\"y\":").append(queryXY[1]).append("},");
        json.append("\"nearestIds\":[");
        for (int i = 0; i < nearest.size(); i++) {
            if (i > 0) json.append(',');
            json.append('"').append(Json.escape(nearest.get(i).id())).append('"');
        }
        json.append("],\"matches\":[");
        for (int i = 0; i < nearest.size(); i++) {
            if (i > 0) json.append(',');
            LensPoint p = nearest.get(i);
            json.append("{\"id\":\"").append(Json.escape(p.id())).append("\",")
                .append("\"score\":").append(nearestScores.get(i)).append(',')
                .append("\"text\":\"").append(Json.escape(truncate(p.text(), 200))).append("\"}");
        }
        json.append("]}");
        respond(ex, 200, "application/json; charset=utf-8", json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private String pointsJson(List<LensPoint> points, double[] scoresOrNull) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < points.size(); i++) {
            LensPoint p = points.get(i);
            if (i > 0) json.append(',');
            json.append("{\"id\":\"").append(Json.escape(p.id())).append("\",")
                .append("\"text\":\"").append(Json.escape(truncate(p.text(), 160))).append("\",")
                .append("\"x\":").append(p.x()).append(',')
                .append("\"y\":").append(p.y()).append(',')
                .append("\"cluster\":").append(clusterIds[i]);
            if (scoresOrNull != null) {
                json.append(",\"score\":").append(scoresOrNull[i]);
            }
            json.append('}');
        }
        json.append(']');
        return json.toString();
    }

    private static List<LensPoint> nearestByCosine(List<LensPoint> points, float[] query, int topK) {
        List<LensPoint> sorted = new ArrayList<>(points);
        sorted.sort(Comparator.comparingDouble((LensPoint p) -> -cosine(query, p.vector())));
        return sorted.subList(0, Math.min(topK, sorted.size()));
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }

    private static void respond(HttpExchange ex, int status, String contentType, byte[] body) throws IOException {
        ex.getResponseHeaders().add("Content-Type", contentType);
        ex.sendResponseHeaders(status, body.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(body);
        }
    }

    // Without this, an unexpected exception (e.g. the embedding model failing to load, or
    // an OOM on a huge store) propagates out of the JDK HttpServer's dispatch and the
    // connection is simply dropped — the browser shows a bare "connection reset" with no
    // explanation. Wrapping every handler turns that into a clean JSON error response.
    private static HttpHandler guarded(HttpHandler handler) {
        return ex -> {
            try {
                handler.handle(ex);
            } catch (Exception e) {
                try {
                    String message = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                    respond(ex, 500, "application/json; charset=utf-8",
                        ("{\"error\":\"" + Json.escape(message) + "\"}").getBytes(StandardCharsets.UTF_8));
                } catch (IOException alreadyBroken) {
                    // connection is gone; nothing more we can do
                }
            }
        };
    }
}
