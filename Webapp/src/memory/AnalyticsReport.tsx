import type {Metric, MetricsReport} from './types';
import './analytics.css';

const LABELS: Record<string, string> = {
  'recall.count': 'Successful recalls',
  'recall.latency_ms.p50': 'Typical recall time',
  'recall.latency_ms.p95': '95th percentile recall time',
  'recall.degraded_count': 'Recalls with a failed retriever',
  'recall.byte_context_reduction.median': 'Median context reduction · bytes',
  'ingest.calls': 'Successful storage calls',
  'ingest.fresh_episodes': 'New episodes stored',
  'ingest.deduplicated': 'Duplicate records skipped',
  'ingest.dedup_fraction': 'Duplicate share',
  'ingest.stored_bytes': 'New content stored',
  'ledger.asserted': 'New claims recorded',
  'ledger.asserted_already_closed': 'Recorded as historical claims',
  'ledger.restated': 'Existing claims restated',
  'ledger.superseded': 'Claims superseded',
  'ledger.manual_closures': 'Claims closed manually',
  'feedback.judged_items': 'Items judged',
  'feedback.useful_share': 'Judged useful',
  'feedback.coverage': 'Returned items with feedback',
  'recall.top_item.both_lanes_share': 'Top excerpt found by both retrievers',
  'recall.top_item.vector_only_share': 'Top excerpt found by vector only',
  'recall.top_item.text_only_share': 'Top excerpt found by text only',
  'recall.latency_ms.embed.p50': 'Embedding',
  'recall.latency_ms.vector.p50': 'Vector search',
  'recall.latency_ms.text.p50': 'Text search',
};

function metricLabel(name: string): string {
  if (name.startsWith('recall.top_similarity.')) return `Top similarity · ${name.slice('recall.top_similarity.'.length)}`;
  return LABELS[name] ?? name.replace(/[._]/g, ' ');
}

function MetricValue({metric}: {metric: Metric}) {
  const value = metric.value;
  if (value === null) return <span className="analytics-unknown">No evidence yet</span>;
  if (typeof value === 'object') return <span className="analytics-distribution">{Object.entries(value).map(([key, number]) => (
    <span key={key}><small>{key}</small> {number === null ? 'Not reported' : number.toFixed(3)}</span>
  ))}</span>;
  if (metric.unit === 'share' || metric.unit === 'share of bytes') return <>{(value * 100).toFixed(1)}<small>%</small></>;
  if (metric.unit === 'ms') return <>{value.toFixed(1)} <small>ms</small></>;
  return <>{value.toLocaleString('en-US', {maximumFractionDigits: 3})}{metric.unit === 'bytes' && <small> bytes</small>}</>;
}

function MetricFigure({metric, headline = false}: {metric: Metric; headline?: boolean}) {
  return (
    <details className={`analytics-metric${headline ? ' analytics-headline' : ''}`}>
      <summary>
        <span className="analytics-metric-label">{metricLabel(metric.name)}</span>
        <strong className="analytics-value"><MetricValue metric={metric} /></strong>
        <span className="analytics-evidence">n = {metric.n.toLocaleString('en-US')} {metric.denominator}</span>
        <span className="analytics-definition-hint">How it’s measured <span aria-hidden="true">⌄</span></span>
      </summary>
      <div className="analytics-definition"><p>{metric.definition}</p>{metric.caveat && <p>{metric.caveat}</p>}</div>
    </details>
  );
}

function recordedTime(value: string | null): string {
  if (!value) return 'Not reported';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('en-US', {month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZone: 'UTC'}) + ' UTC';
}

export function AnalyticsReport({report}: {report: MetricsReport}) {
  const coverage = report.coverage;
  if (!coverage) return null;
  const metrics = new Map((report.metrics ?? []).map(metric => [metric.name, metric]));
  const pick = (names: string[]) => names.flatMap(name => {
    const metric = metrics.get(name);
    return metric ? [metric] : [];
  });
  const panel = (title: string, description: string, names: string[], index: string) => {
    const figures = pick(names);
    if (!figures.length) return null;
    return <section className="analytics-panel" aria-label={title}>
      <header><span className="analytics-section-number" aria-hidden="true">{index}</span><div><h3>{title}</h3><p>{description}</p></div></header>
      <div>{figures.map(metric => <MetricFigure key={metric.name} metric={metric} />)}</div>
    </section>;
  };
  const days = Object.entries(report.daily ?? {}).sort(([a], [b]) => a.localeCompare(b)).map(([date, kinds]) => ({date, total: Object.values(kinds).reduce((a, b) => a + b, 0)}));
  const peak = Math.max(1, ...days.map(day => day.total));
  const summary = pick(['recall.count', 'recall.latency_ms.p50', 'ingest.fresh_episodes', 'feedback.useful_share']);
  const retrievalNames = ['recall.top_item.both_lanes_share', 'recall.top_item.vector_only_share', 'recall.top_item.text_only_share', ...[...metrics.keys()].filter(name => name.startsWith('recall.top_similarity.'))];
  const stepNames = ['recall.latency_ms.embed.p50', 'recall.latency_ms.vector.p50', 'recall.latency_ms.text.p50'];
  return (
    <div className="analytics-workspace">
      <section className="analytics-overview" aria-label="Recorded activity">
        <div className="analytics-overview-copy">
          <span className="analytics-kicker">Recorded activity</span>
          <h2>{coverage.events_considered.toLocaleString('en-US')} <span>events considered</span></h2>
          <span className={`analytics-coverage${coverage.truncated ? ' analytics-partial' : ''}`}>{coverage.truncated ? 'Partial coverage · read truncated' : 'Complete log read'}</span>
          <p>Figures reflect the retained event log. A complete read may still exclude older activity.</p>
        </div>
        <div className="analytics-activity">
          <div className="analytics-chart-heading"><span>Events by recorded day</span>{days.length > 0 && <span>Peak {peak.toLocaleString('en-US')}</span>}</div>
          {days.length > 0 ? <>
            <div className="analytics-chart" role="img" aria-label={`Daily event counts for ${days.length} recorded days. ${days.map(day => `${day.date}: ${day.total}`).join('; ')}`}>
              {days.map(day => <div key={day.date} className="analytics-chart-column" title={`${day.date}: ${day.total.toLocaleString('en-US')} events`}><i style={{height: `${day.total / peak * 100}%`}} /></div>)}
            </div>
            <div className="analytics-chart-range"><span>{days[0].date}</span><span>{days.at(-1)?.date}</span></div>
          </> : <p className="analytics-chart-empty">Daily activity was not reported.</p>}
        </div>
      </section>

      {summary.length > 0 && <section className="analytics-headlines" aria-label="At a glance">{summary.map(metric => <MetricFigure key={metric.name} metric={metric} headline />)}</section>}

      <div className="analytics-section-heading"><h2>Memory in use</h2><p>Open any figure for its definition and caveats.</p></div>
      <div className="analytics-panels">
        {panel('Recall performance', 'Timing and context from successful recalls.', ['recall.latency_ms.p95', 'recall.degraded_count', 'recall.byte_context_reduction.median'], '01')}
        {panel('Stored memory', 'New content and duplicates in this log window.', ['ingest.calls', 'ingest.deduplicated', 'ingest.dedup_fraction', 'ingest.stored_bytes'], '02')}
        {panel('Claim activity', 'Ledger changes, not factual accuracy.', ['ledger.asserted', 'ledger.asserted_already_closed', 'ledger.restated', 'ledger.superseded', 'ledger.manual_closures'], '03')}
        {panel('Feedback', 'Judgements on returned items; usefulness is not overall recall quality.', ['feedback.judged_items', 'feedback.coverage'], '04')}
      </div>

      {pick([...retrievalNames, ...stepNames]).length > 0 && <details className="analytics-diagnostics">
        <summary><span><strong>Retrieval diagnostics</strong><small>Retriever agreement, similarity and step timing</small></span><span className="analytics-expand" aria-hidden="true">+</span></summary>
        <div className="analytics-panels">
          {panel('Retriever agreement', 'Which retrievers found the top excerpt. Agreement does not establish relevance.', retrievalNames, '05')}
          {panel('Time by step', 'Median latency, measured only when each step ran.', stepNames, '06')}
        </div>
      </details>}

      <footer className="analytics-provenance">
        <span className="analytics-kicker">Evidence window</span>
        <dl><div><dt>Earliest retained event</dt><dd>{recordedTime(coverage.earliest_retained)}</dd></div><div><dt>Latest event</dt><dd>{recordedTime(coverage.latest)}</dd></div>
          <div><dt>Recorded failures</dt><dd>{report.failures === undefined ? 'Not reported' : Object.entries(report.failures).map(([kind, count]) => `${kind.replace(/_/g, ' ')}: ${count.toLocaleString('en-US')}`).join(' · ') || 'None in this log window'}</dd></div>
          <div><dt>Embedding models</dt><dd>{report.embedders === undefined ? 'Not reported' : report.embedders.join(' · ') || 'None recorded'}</dd></div>
        </dl>
      </footer>
    </div>
  );
}
