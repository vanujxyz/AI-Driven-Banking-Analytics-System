import React from 'react';
import { SimpleBarChart } from '@carbon/charts-react';
import { Meaning, Stat, chartBase, pct } from '../lib.jsx';

export default function Churn({ data }) {
  const d = data.churn.deciles;
  const m = data.churn.models;
  const bars = d.map((r) => ({ group: 'Customers who left', key: `Group ${r.decile}`, value: +(r.rate * 100).toFixed(1) }));
  const base = d.reduce((a, r) => a + r.churners, 0) / d.reduce((a, r) => a + r.customers, 0);
  return (
    <>
      <h1>Who might leave?</h1>
      <p className="lede">
        We built a model that scores every customer on how likely they are to leave the bank, using things like age,
        country, number of products and whether they are active. Then we split customers into ten equal groups, from most to least at risk.
      </p>
      <div className="tile-row">
        <Stat num={pct(d[0].rate)} label="of the riskiest group (Group 1, 1,000 customers) really did leave" />
        <Stat num={pct(d[9].rate, 1)} label="of the safest group (Group 10) left" />
        <Stat num={pct(base)} label="of all customers left (the baseline)" />
        <Stat num={pct(d[1].cum_capture)} label="of everyone who left sits in the two riskiest groups (20% of customers)" />
      </div>
      <div className="section card">
        <SimpleBarChart
          data={bars}
          options={{
            ...chartBase, title: 'Share of each risk group that actually left',
            axes: { left: { mapsTo: 'value', title: 'Left the bank (%)' }, bottom: { mapsTo: 'key', scaleType: 'labels', title: 'Risk group (1 = highest predicted risk)' } },
            legend: { enabled: false }, bars: { maxWidth: 56 }, color: { scale: { 'Customers who left': '#0f62fe' } },
          }}
        />
      </div>
      <Meaning>
        If the model were useless, every group would look the same, around {pct(base)}. Instead the bars fall steeply: the highest-risk
        group loses {pct(d[0].rate)} of its customers and the lowest-risk group almost none. The team can focus effort on groups 1 and 2.
        The scores were checked on customers the model had not seen when predicting them (5-fold cross-validation), so they are not flattering.
      </Meaning>
      <div className="section">
        <h2>How good is the model?</h2>
        <table className="plain-table">
          <thead><tr><th>Model</th><th className="n">Accuracy score (0.5 = guessing, 1 = perfect)</th><th className="n">Riskiest 10% vs average</th></tr></thead>
          <tbody>
            <tr><td>Simple model (logistic regression)</td><td className="n">{m.logistic_regression.roc_auc_oof.toFixed(3)}</td><td className="n">{m.logistic_regression.top_decile_lift.toFixed(2)}×</td></tr>
            <tr><td><strong>Chosen model (gradient boosting)</strong></td><td className="n"><strong>{m.xgboost.roc_auc_oof.toFixed(3)}</strong></td><td className="n"><strong>{m.xgboost.top_decile_lift.toFixed(2)}×</strong></td></tr>
          </tbody>
        </table>
        <p style={{ fontSize: '.875rem', color: '#525252', marginTop: '.5rem' }}>
          The accuracy score is the “ROC-AUC”. 0.86 or above matches the best published results for this kind of data.
          A column called “Complain” was deliberately left out: it almost perfectly gives away who left, so using it would be cheating.
        </p>
      </div>
    </>
  );
}
