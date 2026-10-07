import React from 'react';
import { GroupedBarChart } from '@carbon/charts-react';
import { Caveat, Meaning, Stat, chartBase, int, pct } from '../lib.jsx';

const NAMES = { rules_only: 'Simple rules', random_forest: 'Decision-forest model', keras_mlp: 'Neural network' };
const BUDGETS = [['budget_0.005', '0.5% of payments'], ['budget_0.01', '1% of payments'], ['budget_0.02', '2% of payments']];

export default function Fraud({ data }) {
  const f = data.fraud;
  const bars = Object.keys(NAMES).flatMap((m) => BUDGETS.map(([k, label]) => (
    { group: NAMES[m], key: label, value: +(f[m][k].recall * 100).toFixed(1) })));
  const nn = f.keras_mlp;
  return (
    <>
      <h1>Suspicious payments</h1>
      <p className="lede">
        Fraud analysts cannot check every payment. They have a fixed “alert budget”: the number of payments they can review each day.
        So the right question is not “how accurate is the model?” but “how much fraud do we catch for the alerts we can afford?”
      </p>
      <div className="tile-row">
        <Stat num={int(nn['budget_0.01'].alerts)} label="payments flagged for review at a 1% budget (out of about 162,000 tested)" />
        <Stat num={pct(nn['budget_0.01'].recall)} label="of the real fraud caught within those alerts" />
        <Stat num={pct(nn['budget_0.01'].precision)} label="of the flagged payments were really fraud" />
        <Stat num={pct(f.rules_only['budget_0.01'].recall, 1)} label="caught by simple rules (speed, distance, time gap) alone" />
      </div>
      <div className="section card">
        <GroupedBarChart
          data={bars}
          options={{
            ...chartBase, title: 'Share of all fraud caught, by how many payments analysts can review',
            axes: { left: { mapsTo: 'value', title: 'Fraud caught (%)', domain: [0, 100] }, bottom: { mapsTo: 'key', scaleType: 'labels', title: 'Alert budget' } },
          }}
        />
      </div>
      <Meaning>
        About 1 in 100 payments in our test set is fraud. If analysts can only review half of that number (0.5%), no model can catch more than
        about half the fraud, which is why the first group of bars stops near 49%. At 1% the models catch about 98% of fraud with almost no wasted checks.
        At 2% half the alerts are false alarms, which costs analyst time without catching more.
      </Meaning>
      <Caveat title="Why the scores look almost too good">
        The payment data comes from a simulator (PaySim), and its fraud is easy to spot because fraudsters empty an account in one go.
        Real banks face much harder fraud, so treat these scores as a demonstration of the method, not a promise of real-world performance.
        The simple rules score near zero because the locations and times were generated at random, so they carry no fraud signal by design.
      </Caveat>
    </>
  );
}
