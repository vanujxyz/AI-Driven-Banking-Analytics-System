import React from 'react';
import { LineChart } from '@carbon/charts-react';
import { Meaning, Stat, SEGMENT, chartBase, money, pct } from '../lib.jsx';

const COLORS = {
  [SEGMENT('France-active')]: '#0043ce', [SEGMENT('France-inactive')]: '#78a9ff',
  [SEGMENT('Germany-active')]: '#a2191f', [SEGMENT('Germany-inactive')]: '#ff8389',
  [SEGMENT('Spain-active')]: '#0e6027', [SEGMENT('Spain-inactive')]: '#6fdc8c',
};

const HR_TEXT = {
  Age: ['Each 10 years older', 'more likely to leave'],
  Germany: ['Living in Germany (vs France)', 'more likely to leave'],
  Active: ['Being an active member', ''],
  Prod2: ['Holding exactly 2 products (vs 1)', ''],
  Prod3plus: ['Holding 3 or more products (vs 1)', ''],
  Male: ['Being male', ''],
};

export default function Stay({ data }) {
  const s = data.survival;
  const line = Object.entries(s.km).flatMap(([seg, pts]) =>
    pts.map((p) => ({ group: SEGMENT(seg), year: p.year, value: +(p.retained * 100).toFixed(1) })));
  const cs = s.clv_vs_survival;
  const hr = s.hazard_ratios.filter((h) => HR_TEXT[h.covariate]);
  const hrText = (h) => {
    const r = h.hazard_ratio;
    return r >= 1 ? `${Math.round((r - 1) * 100)}% more likely to leave` : `${Math.round((1 - r) * 100)}% less likely to leave`;
  };
  const gi = cs.find((r) => r.segment === 'Germany-inactive');
  const worst = [...s.segments].sort((a, b) => a.S_at_5y - b.S_at_5y)[0];
  const best = [...s.segments].sort((a, b) => b.S_at_5y - a.S_at_5y)[0];
  return (
    <>
      <h1>How long do customers stay?</h1>
      <p className="lede">
        Churn models say <em>whether</em> someone might leave. Survival analysis says <em>when</em>. Combined with customer worth,
        it tells the bank how much of each customer’s value it can realistically expect to keep.
      </p>
      <div className="tile-row">
        <Stat num={pct(best.S_at_5y)} label={`still with the bank after 5 years: ${SEGMENT(best.segment)} (best group)`} />
        <Stat num={pct(worst.S_at_5y)} label={`still with the bank after 5 years: ${SEGMENT(worst.segment)} (worst group)`} />
        <Stat num={s.metrics.cox.concordance_cv_mean.toFixed(2)} label="ranking accuracy of the timing model (0.5 = guessing, 1 = perfect)" />
      </div>
      <div className="section card">
        <LineChart
          data={line}
          options={{
            ...chartBase, height: '400px', title: 'Share of customers still with the bank, by years since joining',
            axes: { bottom: { mapsTo: 'year', title: 'Years with the bank', scaleType: 'linear' }, left: { mapsTo: 'value', title: 'Still customers (%)', domain: [0, 100] } },
            curve: 'curveStepAfter', points: { enabled: false }, legend: { truncation: { type: 'none' } },
            color: { scale: COLORS },
          }}
        />
      </div>
      <Meaning>
        Each line starts at 100% and falls as customers leave. A line that drops faster is a group the bank loses faster.
        The groups clearly differ: a statistical test (the log-rank test) says the chance this is a coincidence is effectively zero.
        German inactive customers fall furthest; French and Spanish active customers are the most loyal.
      </Meaning>

      <div className="section">
        <h2>How much value is really safe? Worth vs. expected stay</h2>
        <table className="plain-table">
          <thead><tr>
            <th>Group</th><th className="n">Average 5-year value</th><th className="n">Chance of staying 5 years</th>
            <th className="n">Years kept (of 5)</th><th className="n">Value after allowing for leaving</th>
          </tr></thead>
          <tbody>
            {cs.map((r) => (
              <tr key={r.segment}>
                <td>{SEGMENT(r.segment)}</td><td className="n">{money(r.clv_pred)}</td><td className="n">{pct(r.surv_5y)}</td>
                <td className="n">{r.exp_years.toFixed(1)}</td><td className="n"><strong>{money(r.sa_clv)}</strong></td>
              </tr>))}
          </tbody>
        </table>
        <Meaning>
          German inactive customers look very valuable on paper ({money(gi.clv_pred)}),
          but once the bank allows for the likelihood they leave, about {Math.round((1 - gi.sa_clv / gi.clv_pred) * 100)}% of that value disappears. This is why “how much” and “how soon” must be read together.
        </Meaning>
      </div>

      <div className="section">
        <h2>What makes a customer leave sooner?</h2>
        <table className="plain-table">
          <thead><tr><th>Characteristic</th><th>Effect on the chance of leaving, at any point in time</th></tr></thead>
          <tbody>
            {hr.map((h) => <tr key={h.covariate}><td>{HR_TEXT[h.covariate][0]}</td><td>{hrText(h)}</td></tr>)}
          </tbody>
        </table>
        <p style={{ fontSize: '.875rem', color: '#525252', marginTop: '.5rem' }}>
          Holding two products is protective, but three or more is a warning sign: customers with many products here are far more likely to leave.
          These come from a Cox regression model, and each effect is measured with the other characteristics held equal.
        </p>
      </div>
    </>
  );
}
