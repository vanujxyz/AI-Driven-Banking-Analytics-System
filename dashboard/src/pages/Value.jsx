import React from 'react';
import { GroupedBarChart } from '@carbon/charts-react';
import { Caveat, Meaning, Stat, SEGMENT, chartBase, int, money } from '../lib.jsx';

export default function Value({ data }) {
  const segs = data.clv.segments;
  const m = data.clv.metrics;
  const bars = segs.flatMap((s) => [
    { group: 'Actual value', key: SEGMENT(s.segment), value: s.mean_clv_true },
    { group: 'Predicted value', key: SEGMENT(s.segment), value: s.mean_clv_pred },
  ]);
  return (
    <>
      <h1>What customers are worth</h1>
      <p className="lede">
        “Customer lifetime value” is how much money a customer is expected to bring the bank over the next five years
        (interest earned, product fees, card usage), with future money counted slightly less than money today.
      </p>
      <div className="tile-row">
        <Stat num={money(Math.max(...segs.map((s) => s.mean_clv_true)))} label="average five-year value of the most valuable group (German active customers)" />
        <Stat num={money(Math.min(...segs.map((s) => s.mean_clv_true)))} label="average value of the least valuable group (Spanish inactive customers)" />
        <Stat num={int(m.test_n)} label="customers held back to test the predictions fairly" />
      </div>
      <div className="section card">
        <GroupedBarChart
          data={bars}
          options={{
            ...chartBase, height: '380px', title: 'Average five-year value per customer: actual vs predicted, by group',
            axes: { left: { mapsTo: 'value', title: 'Five-year value (€)' }, bottom: { mapsTo: 'key', scaleType: 'labels' } },
          }}
        />
      </div>
      <Meaning>
        The predictions track the real averages reasonably closely for every group (within about 11%, and within 4% for most), so the bank can trust the model to say
        <em> which kinds of customer</em> are worth more. German customers are worth about 70% more than French or Spanish ones, mostly because
        they hold larger balances.
      </Meaning>
      <Caveat title="What the model cannot do">
        For any single person, the prediction is rough: it explains only about {Math.round(m.test_r2 * 100)}% of the differences between individual customers.
        It uses an automated tool (H2O AutoML) that tried a dozen models and kept the best. The “value” itself follows a formula we chose
        (2% of balance, plus product and card income), so it is an estimate under stated assumptions, not an accounting figure.
      </Caveat>
    </>
  );
}
