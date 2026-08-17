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
type StoreSource = "google_play" | "app_store";
type AppSearchResult = {
  name: string;
  identifier: string;
  developer: string | null;
  icon_url: string | null;
  store_url: string | null;
};
type ImportJob = {
  id: string;
  source: StoreSource;
  app_name: string;
  requested_reviews: number;
  status: "queued" | "running" | "completed" | "failed";
  fetched_reviews: number;
  labeled_reviews: number;
  saved_reviews: number;
  error: string | null;
};
type PreviewReview = {
  text: string;
  rating: number | null;
  date: string;
  source: StoreSource;
  app_name: string;
};
type Recommendation = {
  provider: "local" | "anthropic";
  summary: string;
  actions: { priority: number; title: string; rationale: string; evidence: string }[];
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
  const [importSource, setImportSource] = useState<StoreSource>("google_play");
  const [appQuery, setAppQuery] = useState("");
  const [searchResults, setSearchResults] = useState<AppSearchResult[]>([]);
  const [searchingApps, setSearchingApps] = useState(false);
  const [selectedApp, setSelectedApp] = useState<AppSearchResult | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [previewReviews, setPreviewReviews] = useState<PreviewReview[]>([]);
  const [selectedReviewIndexes, setSelectedReviewIndexes] = useState<Set<number>>(new Set());
  const [previewingReviews, setPreviewingReviews] = useState(false);
  const [importJob, setImportJob] = useState<ImportJob | null>(null);
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [planning, setPlanning] = useState(false);

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

  async function searchApps(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (appQuery.trim().length < 2) return;
    setSearchingApps(true);
    setSelectedApp(null);
    setSearchResults([]);
    setError("");
    try {
      const parameters = new URLSearchParams({ source: importSource, query: appQuery.trim() });
      const response = await fetch(`${API_URL}/apps/search?${parameters}`);
      if (!response.ok) throw new Error(await errorMessage(response));
      setSearchResults((await response.json()) as AppSearchResult[]);
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : "Could not search the app store.");
    } finally {
      setSearchingApps(false);
    }
  }

  async function previewImport() {
    if (!selectedApp) return;
    setPreviewingReviews(true);
    setPreviewReviews([]);
    setSelectedReviewIndexes(new Set());
    setError("");
    try {
      const response = await fetch(`${API_URL}/imports/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: importSource, app_name: selectedApp.name, identifier: selectedApp.identifier, start_date: startDate || null, end_date: endDate || null }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const reviews = (await response.json()) as PreviewReview[];
      setPreviewReviews(reviews);
      setSelectedReviewIndexes(new Set(reviews.map((_, index) => index)));
    } catch (previewError) {
      setError(previewError instanceof Error ? previewError.message : "Could not preview reviews.");
    } finally {
      setPreviewingReviews(false);
    }
  }

  function toggleReview(index: number) {
    setSelectedReviewIndexes((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  async function startImport() {
    if (!selectedApp || !selectedReviewIndexes.size) return;
    setImportJob(null);
    setRecommendation(null);
    setError("");
    try {
      const response = await fetch(`${API_URL}/imports`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: importSource, app_name: selectedApp.name, reviews: previewReviews.filter((_, index) => selectedReviewIndexes.has(index)) }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const job = (await response.json()) as ImportJob;
      setImportJob(job);
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Could not start the import.");
    }
  }

  useEffect(() => {
    if (!importJob || ["completed", "failed"].includes(importJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/imports/${importJob.id}`);
        if (!response.ok) throw new Error(await errorMessage(response));
        const job = (await response.json()) as ImportJob;
        setImportJob(job);
        if (job.status === "completed") void loadDashboard();
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Could not check import progress.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [importJob, loadDashboard]);

  async function generatePlan(provider: "local" | "anthropic") {
    if (!selectedApp) return;
    setPlanning(true);
    setRecommendation(null);
    setError("");
    try {
      const response = await fetch(`${API_URL}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ app_name: selectedApp.name, provider, confirm_paid_request: provider === "anthropic" }),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      setRecommendation((await response.json()) as Recommendation);
    } catch (planError) {
      setError(planError instanceof Error ? planError.message : "Could not generate recommendations.");
    } finally {
      setPlanning(false);
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

      <section className="mb-8 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Import app reviews</h2>
        <p className="mt-1 text-sm text-slate-500">Search a store, select an app, then import and label a small review batch locally.</p>
        <form className="mt-4 flex flex-col gap-3 sm:flex-row" onSubmit={searchApps}>
          <select value={importSource} onChange={(event) => { setImportSource(event.target.value as StoreSource); setSearchResults([]); setSelectedApp(null); }}>
            <option value="google_play">Google Play</option>
            <option value="app_store">App Store</option>
          </select>
          <input className="min-w-0 flex-1" value={appQuery} onChange={(event) => setAppQuery(event.target.value)} placeholder="Search an app, e.g. Claude" />
          <button disabled={searchingApps} type="submit">{searchingApps ? "Searching..." : "Search"}</button>
        </form>
        {searchResults.length > 0 && (
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {searchResults.map((app) => (
              <button className={`flex items-center gap-3 rounded-lg border p-3 text-left ${selectedApp?.identifier === app.identifier ? "border-indigo-500 bg-indigo-50" : "border-slate-200"}`} key={app.identifier} onClick={() => setSelectedApp(app)} type="button">
                {app.icon_url && <img alt="" className="h-10 w-10 rounded-lg" src={app.icon_url} />}
                <span><strong className="block">{app.name}</strong><span className="text-xs text-slate-500">{app.developer ?? app.identifier}</span></span>
              </button>
            ))}
          </div>
        )}
        {selectedApp && (
          <div className="mt-4 rounded-lg bg-slate-50 p-4">
            <p className="text-sm"><strong>{selectedApp.name}</strong> selected. Preview up to 30 recent reviews, then choose up to 20 to label.</p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <label className="text-sm">From <input className="ml-1" onChange={(event) => setStartDate(event.target.value)} type="date" value={startDate} /></label>
              <label className="text-sm">To <input className="ml-1" onChange={(event) => setEndDate(event.target.value)} type="date" value={endDate} /></label>
              <button disabled={previewingReviews} onClick={() => void previewImport()} type="button">{previewingReviews ? "Fetching..." : "Preview reviews"}</button>
            </div>
          </div>
        )}
        {previewReviews.length > 0 && (
          <div className="mt-4 rounded-lg border border-slate-200 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-medium">Choose reviews to label ({selectedReviewIndexes.size} selected; maximum 20)</p><button disabled={!selectedReviewIndexes.size || selectedReviewIndexes.size > 20} onClick={() => void startImport()} type="button">Import and label selected</button></div>
            <div className="mt-3 max-h-96 space-y-2 overflow-y-auto">
              {previewReviews.map((review, index) => <label className="flex cursor-pointer gap-3 rounded-lg bg-slate-50 p-3 text-sm" key={`${review.date}-${index}`}><input checked={selectedReviewIndexes.has(index)} onChange={() => toggleReview(index)} type="checkbox" /><span><span className="block text-xs text-slate-500">{new Date(review.date).toLocaleDateString()} | {review.rating ?? "no"} stars</span>{review.text}</span></label>)}
            </div>
          </div>
        )}
        {importJob && (
          <div className="mt-4 rounded-lg bg-indigo-50 p-4 text-sm text-indigo-950">
            <p><strong className="capitalize">{importJob.status}</strong>: {importJob.fetched_reviews} fetched, {importJob.labeled_reviews} labeled, {importJob.saved_reviews} saved of {importJob.requested_reviews}.</p>
            {importJob.error && <p className="mt-1 text-rose-700">{importJob.error}</p>}
            {importJob.status === "completed" && <div className="mt-3 flex flex-wrap gap-2"><button disabled={planning} onClick={() => void generatePlan("local")} type="button">{planning ? "Generating..." : "Generate local action plan"}</button><button disabled={planning} onClick={() => void generatePlan("anthropic")} type="button">Use Claude for action plan</button></div>}
          </div>
        )}
        {recommendation && <div className="mt-4 rounded-lg bg-emerald-50 p-4 text-sm"><p><strong>Recommended next steps</strong> ({recommendation.provider})</p><p className="mt-1">{recommendation.summary}</p><ol className="mt-3 list-decimal space-y-2 pl-5">{recommendation.actions.map((action) => <li key={action.priority}><strong>{action.title}</strong> - {action.rationale}<p className="mt-1 text-slate-600">Evidence: {action.evidence}</p></li>)}</ol></div>}
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
