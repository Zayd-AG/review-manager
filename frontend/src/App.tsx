import { FormEvent, useCallback, useEffect, useState } from "react";

type Category =
  | "bug"
  | "feature_request"
  | "praise"
  | "churn_risk"
  | "pricing_complaint"
  | "usability_complaint"
  | "other";
type Severity = "low" | "medium" | "high";

type DashboardCluster = {
  id: string;
  representative_text: string;
  category: Category | null;
  severity: Severity | null;
  count: number;
  priority_score: number;
  source_breakdown: Record<string, number>;
};

type FeedbackItem = {
  id: string;
  text: string;
  source: string;
  app_name: string;
  rating: number | null;
  date: string | null;
};

type ClusterDetail = DashboardCluster & { source_reviews: FeedbackItem[] };
type Classification = {
  category: Category;
  severity: Severity;
  justification: string;
};
type DashboardSummary = {
  review_count: number;
  cluster_count: number;
  classifier_name: string;
  embedding_model: string;
  evaluation: {
    gold_set_reviews: number;
    base_category_accuracy: number;
    lora_category_accuracy: number;
    teacher_category_accuracy: number;
  };
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const categories: Category[] = [
  "bug",
  "feature_request",
  "praise",
  "churn_risk",
  "pricing_complaint",
  "usability_complaint",
  "other",
];
const severityStyles: Record<string, string> = {
  high: "bg-rose-100 text-rose-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-sky-100 text-sky-700",
};

function readable(value: string | null) {
  return value ? value.replaceAll("_", " ") : "unlabeled";
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

async function errorMessage(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? "The API returned an unexpected error.";
  } catch {
    return "The API returned an unexpected error.";
  }
}

export default function App() {
  const [clusters, setClusters] = useState<DashboardCluster[]>([]);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [category, setCategory] = useState("");
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [details, setDetails] = useState<Record<string, ClusterDetail>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [reviewText, setReviewText] = useState("");
  const [classification, setClassification] = useState<Classification | null>(null);
  const [classifying, setClassifying] = useState(false);

  const loadDashboard = useCallback(async () => {
    const parameters = new URLSearchParams();
    if (category) parameters.set("category", category);
    if (source) parameters.set("source", source);

    setLoading(true);
    setError("");
    try {
      const [clustersResponse, summaryResponse] = await Promise.all([
        fetch(`${API_URL}/dashboard?${parameters}`),
        fetch(`${API_URL}/summary`),
      ]);
      if (!clustersResponse.ok) throw new Error(await errorMessage(clustersResponse));
      if (!summaryResponse.ok) throw new Error(await errorMessage(summaryResponse));
      setClusters((await clustersResponse.json()) as DashboardCluster[]);
      setSummary((await summaryResponse.json()) as DashboardSummary);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Could not load the dashboard. Is the API and Postgres stack running?",
      );
    } finally {
      setLoading(false);
    }
  }, [category, source]);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  async function toggleCluster(clusterId: string) {
    if (expandedId === clusterId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(clusterId);
    if (details[clusterId]) return;
    try {
      const response = await fetch(`${API_URL}/clusters/${clusterId}`);
      if (!response.ok) throw new Error(await errorMessage(response));
      const detail = (await response.json()) as ClusterDetail;
      setDetails((current) => ({ ...current, [clusterId]: detail }));
    } catch (detailError) {
      setError(
        detailError instanceof Error
          ? detailError.message
          : "Could not load cluster examples.",
      );
      setExpandedId(null);
    }
  }

  async function classify(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!reviewText.trim()) return;
    setClassifying(true);
    setClassification(null);
    setError("");
    try {
      const response = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: reviewText }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      setClassification((await response.json()) as Classification);
    } catch (classifyError) {
      setError(
        classifyError instanceof Error
          ? classifyError.message
          : "Could not classify the review.",
      );
    } finally {
      setClassifying(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-5 py-10 text-slate-800">
      <header className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="mb-2 text-sm font-medium uppercase tracking-[0.2em] text-indigo-600">
            Feedback Lens
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">Review clusters</h1>
          <p className="mt-2 text-slate-500">
            Repeated product feedback, ranked by frequency and severity.
          </p>
        </div>
        <div className="flex gap-2">
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="">All categories</option>
            {categories.map((item) => (
              <option key={item} value={item}>{readable(item)}</option>
            ))}
          </select>
          <select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="">All sources</option>
            <option value="google_play">Google Play</option>
            <option value="app_store">App Store</option>
          </select>
        </div>
      </header>

      <section className="mb-8 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-semibold">How it works</h2>
          <ol className="mt-3 grid gap-2 text-sm leading-5 text-slate-600 sm:grid-cols-4">
            <li><strong className="text-slate-800">1. Collect</strong><br />Mobile app reviews are normalized.</li>
            <li><strong className="text-slate-800">2. Classify</strong><br />A LoRA-tuned model assigns labels.</li>
            <li><strong className="text-slate-800">3. Cluster</strong><br />Embeddings group repeated issues.</li>
            <li><strong className="text-slate-800">4. Prioritize</strong><br />Frequency x severity ranks clusters.</li>
          </ol>
        </div>
        <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-5 text-sm">
          <h2 className="font-semibold text-indigo-950">Model and evaluation</h2>
          {summary ? (
            <div className="mt-3 space-y-1.5 text-indigo-900">
              <p>{summary.review_count.toLocaleString()} reviews to {summary.cluster_count} clusters</p>
              <p>{summary.classifier_name}</p>
              <p>Embeddings: {summary.embedding_model}</p>
              <p>Gold set: {summary.evaluation.gold_set_reviews} reviews</p>
              <p>Category accuracy: base {percent(summary.evaluation.base_category_accuracy)} / LoRA {percent(summary.evaluation.lora_category_accuracy)} / teacher {percent(summary.evaluation.teacher_category_accuracy)}</p>
            </div>
          ) : <p className="mt-3 text-indigo-700">Metrics load with the dashboard.</p>}
        </div>
      </section>

      {error && (
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-lg bg-rose-50 p-4 text-sm text-rose-800">
          <span>{error}</span>
          <button type="button" onClick={() => void loadDashboard()}>Retry dashboard</button>
        </div>
      )}

      <section className="grid gap-4">
        {loading && <p className="rounded-lg bg-white p-5 text-slate-500 shadow-sm">Loading clusters...</p>}
        {!loading && !error && !clusters.length && (
          <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-600">
            <p className="font-medium text-slate-800">No clusters match these filters.</p>
            <p className="mt-1 text-sm">Try a different category or source, or run the local pipeline to add reviews.</p>
          </div>
        )}
        {clusters.map((cluster) => {
          const detail = details[cluster.id];
          return (
            <article key={cluster.id} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-3 sm:flex-row sm:justify-between">
                <div>
                  <div className="mb-3 flex flex-wrap items-center gap-2 text-xs font-medium">
                    <span className="rounded-full bg-slate-100 px-2.5 py-1 capitalize text-slate-700">{readable(cluster.category)}</span>
                    <span className={`rounded-full px-2.5 py-1 capitalize ${severityStyles[cluster.severity ?? ""] ?? "bg-slate-100 text-slate-700"}`}>{cluster.severity ?? "unlabeled"}</span>
                  </div>
                  <p className="max-w-3xl leading-6">{cluster.representative_text}</p>
                </div>
                <div className="shrink-0 text-left sm:text-right">
                  <p className="text-2xl font-semibold">{cluster.count}</p>
                  <p className="text-xs text-slate-500">reviews | priority score {cluster.priority_score}</p>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                <div className="text-sm text-slate-500">
                  <p>{Object.entries(cluster.source_breakdown).map(([name, count]) => `${name}: ${count}`).join(" | ")}</p>
                  <p className="mt-1 text-xs">Priority = review frequency x severity multiplier.</p>
                </div>
                <button type="button" onClick={() => void toggleCluster(cluster.id)}>
                  {expandedId === cluster.id ? "Hide examples" : "Show examples"}
                </button>
              </div>
              {expandedId === cluster.id && (
                <div className="mt-4 space-y-3 rounded-lg bg-slate-50 p-4">
                  {!detail && <p className="text-sm text-slate-500">Loading examples...</p>}
                  {detail?.source_reviews.slice(0, 3).map((review) => (
                    <div key={review.id} className="border-l-2 border-indigo-200 pl-3">
                      <p className="text-sm leading-5">{review.text}</p>
                      <p className="mt-1 text-xs text-slate-500">{review.app_name} | {review.source} | {review.rating ?? "no"} stars</p>
                    </div>
                  ))}
                </div>
              )}
            </article>
          );
        })}
      </section>

      <section className="mt-12 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Live classification</h2>
        <p className="mt-1 text-sm text-slate-500">Paste a review to run the fine-tuned model locally.</p>
        <form className="mt-5" onSubmit={classify}>
          <textarea value={reviewText} onChange={(event) => setReviewText(event.target.value)} placeholder="Paste a review here..." rows={5} />
          <button className="mt-3" disabled={classifying} type="submit">{classifying ? "Classifying..." : "Classify review"}</button>
        </form>
        {classification && (
          <div className="mt-5 rounded-lg bg-indigo-50 p-4 text-sm">
            <p><strong className="capitalize">{readable(classification.category)}</strong> | <span className="capitalize">{classification.severity}</span></p>
            <p className="mt-2 text-slate-600">{classification.justification}</p>
          </div>
        )}
      </section>
    </main>
  );
}
