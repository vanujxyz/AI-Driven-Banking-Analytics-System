import React, { useMemo, useState } from 'react';
import { Slider } from '@carbon/react';
import { LineChart } from '@carbon/charts-react';
import { ACTION, Meaning, Stat, chartBase, int, pct } from '../lib.jsx';

function useStrategies(rows) {
  return useMemo(() => {
    const n = rows.length;
    const lost = rows.map((r) => (r.Exited ? r.sa_clv : 0));
    const total = lost.reduce((a, b) => a + b, 0);
    const prefix = (key) => {
      const idx = rows.map((_, i) => i).sort((a, b) => rows[b][key] - rows[a][key]);
      const out = new Float64Array(n + 1);
      idx.forEach((i, j) => { out[j + 1] = out[j] + lost[i]; });
      return out;
    };
    return { n, total, byValue: prefix('value_at_risk'), byChurn: prefix('churn_prob') };
  }, [rows]);
}

export default function Overview({ data }) {
  const rows = data.action_list;
  const S = useStrategies(rows);
  const [budget, setBudget] = useState(10);
  const k = Math.round((budget / 100) * S.n);
  const share = (arr) => arr[k] / S.total;

  const curve = useMemo(() => {
    const pts = [];
    for (let b = 1; b <= 50; b += 1) {
      const kk = Math.round((b / 100) * S.n);
      pts.push({ group: 'Rank by value at risk (our method)', budget: b, value: +(100 * S.byValue[kk] / S.total).toFixed(1) });
      pts.push({ group: 'Rank by chance of leaving only', budget: b, value: +(100 * S.byChurn[kk] / S.total).toFixed(1) });
      pts.push({ group: 'Contact at random', budget: b, value: b });
    }
    return pts;
  }, [S]);

  const counts = data.integration.action_counts;
  const q = data.churn.deciles;

  return (
    <>
      <h1>Which customers need attention first?</h1>
      <p className="lede">
        A bank has thousands of customers and a small team. This dashboard combines five analyses
        (leaving risk, suspicious payments, customer worth, how long they stay, and what to offer next)
        into one ranked list, so the team knows who to call first.
      </p>

      <div className="tile-row">
        <Stat num={int(S.n)} label="customers analysed in a fictional European bank" />
        <Stat num={pct(rows.reduce((a, r) => a + r.Exited, 0) / S.n)} label="of these customers actually left the bank" />
        <Stat num={`${q[0].lift.toFixed(1)}×`} label="how much more likely the 10% we flag as riskiest are to leave, versus an average customer" />
        <Stat num={pct(q[0].cum_capture)} label="of all customers who left were already in that riskiest 10%" />
      </div>

      <div className="section">
        <h2>What should the team do? Suggested actions for all {int(S.n)} customers</h2>
        <table className="plain-table">
          <thead><tr><th>Action</th><th className="n">Customers</th><th>When it applies</th></tr></thead>
          <tbody>
            {[
              ['FRAUD_REVIEW', 'Our payment-checking model flagged unusual activity.'],
              ['RETAIN_HIGH_VALUE', 'Among the 20% most likely to leave, and worth more than the typical customer.'],
              ['RETAIN_STANDARD', 'Among the 20% most likely to leave, but worth less than the typical customer.'],
              ['CROSS_SELL', 'No urgent problem. Suggest the product they are most likely to want.'],
            ].map(([a, why]) => (
              <tr key={a}><td>{ACTION[a].label}</td><td className="n">{int(counts[a])}</td><td>{why}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section">
        <h2>Try it: how much do we save by choosing wisely?</h2>
        <p className="lede" style={{ marginBottom: '1rem' }}>
          Imagine the retention team can only contact a fraction of customers. Move the slider to see how much of the
          value lost to customers leaving each approach reaches.
        </p>
        <div className="card">
          <Slider labelText="Share of customers the team can contact" min={1} max={50} step={1} value={budget}
            onChange={({ value }) => setBudget(value)} formatLabel={(v) => `${v}%`} />
          <div className="tile-row" style={{ marginTop: '1.5rem' }}>
            <Stat num={pct(share(S.byValue))} label={`of lost value reached by ranking by value at risk (contacting ${int(k)} customers)`} />
            <Stat num={pct(share(S.byChurn))} label="reached by ranking by chance of leaving only" />
            <Stat num={pct(budget / 100)} label="reached by contacting customers at random" />
          </div>
          <LineChart
            data={curve}
            options={{
              ...chartBase, title: 'Lost value reached, by share of customers contacted',
              axes: { bottom: { mapsTo: 'budget', title: 'Share of customers contacted (%)', scaleType: 'linear' },
                      left: { mapsTo: 'value', title: 'Share of lost value reached (%)' } },
              curve: 'curveMonotoneX', points: { enabled: false },
            }}
          />
        </div>
        <Meaning>
          “Value at risk” is a customer’s chance of leaving multiplied by what they would be worth over five years if they stayed.
          Ranking by it prioritises people who are both likely to leave and worth keeping. At a 10% budget it reaches
          {' '}{pct(data.integration.rank_by_value_at_risk.share_of_value)} of lost value, against {pct(data.integration.rank_by_churn_prob_only.share_of_value)} by
          chance of leaving alone and {pct(data.integration.random_contact.share_of_value)} at random. Money is shown in euros as an illustration; the data does not state a currency.
        </Meaning>
      </div>
    </>
  );
}
