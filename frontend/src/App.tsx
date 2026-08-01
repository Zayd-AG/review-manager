import { FormEvent, useEffect, useState } from "react";

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

export default function App() {
  const [clusters, setClusters] = useState<DashboardCluster[]>([]);
  const [category, setCategory] = useState("");
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [details, setDetails] = useState<Record<string, ClusterDetail>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [reviewText, setReviewText] = useState("");
  const [classification, setClassification] = useState<Classification | null>(null);
  const [classifying, setClassifying] = useState(false);

  useEffect(() => {
    const parameters = new URLSearchParams();
    if (category) parameters.set("category", category);
    if (source) parameters.set("source", source);

    setLoading(true);
    setError("");
    fetch(`${API_URL}/dashboard?${parameters}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return response.json() as Promise<DashboardCluster[]>;
      })
      .then(setClusters)
      .catch(() => setError("Could not load clusters. Is the API running?"))
      .finally(() => setLoading(false));
  }, [category, source]);

  async function toggleCluster(clusterId: string) {
    if (expandedId === clusterId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(clusterId);
    if (details[clusterId]) return;
    const response = await fetch(`${API_URL}/clusters/${clusterId}`);
    if (response.ok) {
      const detail = (await response.json()) as ClusterDetail;
      setDetails((current) => ({
        ...current,
        [clusterId]: detail,
      }));
    }
  }

  async function classify(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!reviewText.trim()) return;
    setClassifying(true);
    setClassification(null);
    try {
      const response = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: reviewText }),
      });
      if (!response.ok) throw new Error(await response.text());
      setClassification((await response.json()) as Classification);
    } catch {
      setError("Could not classify the review. Check that the API is running.");
    } finally {
      setClassifying(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-6xl px-5 py-10 text-slate-800">
      <header className="mb-10 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="mb-2 text-sm font-medium uppercase tracking-[0.2em] text-indigo-600">Feedback Lens</p>
          <h1 className="text-3xl font-semibold tracking-tight">Review clusters</h1>
          <p className="mt-2 text-slate-500">Ranked by frequency and severity.</p>
        </div>
        <div className="flex gap-2">
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="">All categories</option>
            {categories.map((item) => <option key={item} value={item}>{readable(item)}</option>)}
          </select>
          <select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="">All sources</option>
            <option value="google_play">Google Play</option>
            <option value="app_store">App Store</option>
          </select>
        </div>
      </header>

      {error && <p className="mb-5 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
      <section className="grid gap-4">
        {loading && <p className="text-slate-500">Loading clusters…</p>}
        {!loading && !clusters.length && <p className="text-slate-500">No clusters match these filters.</p>}
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
                  <p className="text-xs text-slate-500">reviews · score {cluster.priority_score}</p>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                <p className="text-sm text-slate-500">{Object.entries(cluster.source_breakdown).map(([name, count]) => `${name}: ${count}`).join(" · ")}</p>
                <button type="button" onClick={() => void toggleCluster(cluster.id)}>
                  {expandedId === cluster.id ? "Hide examples" : "Show examples"}
                </button>
              </div>
              {expandedId === cluster.id && (
                <div className="mt-4 space-y-3 rounded-lg bg-slate-50 p-4">
                  {!detail && <p className="text-sm text-slate-500">Loading examples…</p>}
                  {detail?.source_reviews.slice(0, 3).map((review) => (
                    <div key={review.id} className="border-l-2 border-indigo-200 pl-3">
                      <p className="text-sm leading-5">{review.text}</p>
                      <p className="mt-1 text-xs text-slate-500">{review.app_name} · {review.source} · {review.rating ?? "no"} stars</p>
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
          <textarea value={reviewText} onChange={(event) => setReviewText(event.target.value)} placeholder="Paste a review here…" rows={5} />
          <button className="mt-3" disabled={classifying} type="submit">{classifying ? "Classifying…" : "Classify review"}</button>
        </form>
        {classification && (
          <div className="mt-5 rounded-lg bg-indigo-50 p-4 text-sm">
            <p><strong className="capitalize">{readable(classification.category)}</strong> · <span className="capitalize">{classification.severity}</span></p>
            <p className="mt-2 text-slate-600">{classification.justification}</p>
          </div>
        )}
      </section>
    </main>
  );
}
