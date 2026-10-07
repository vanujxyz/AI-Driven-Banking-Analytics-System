import React from 'react';
import { GroupedBarChart } from '@carbon/charts-react';
import { Caveat, Meaning, Stat, chartBase, int } from '../lib.jsx';

const METHODS = [
  ['popularity', 'Most popular product (baseline)'],
  ['cf', 'People like you (collaborative filtering)'],
  ['rules_score', 'Bought-together rules'],
  ['segment_pop', 'Popular with your age and wealth group'],
  ['hybrid', 'Our blend'],
];

export default function Recs({ data }) {
  const r = data.recommender;
  const bars = ['warm', 'cold'].flatMap((g) => METHODS.map(([k, name]) => ({
    group: name, key: g === 'warm' ? 'With product history' : 'New, no history',
    value: +(r[g][k]['hit_rate@1'] * 100).toFixed(1) })));
  return (
    <>
      <h1>What should we offer next?</h1>
      <p className="lede">
        For each customer the bank suggests one product they do not have yet. We tested this by hiding one product a customer really holds
        and checking whether the recommender puts it first.
      </p>
      <div className="tile-row">
        <Stat num={`${r.warm.hybrid['lift_vs_popularity@1'].toFixed(2)}×`} label="how much better our blend is than “just recommend the most popular product” for existing customers" />
        <Stat num={`${r.cold.hybrid['lift_vs_popularity@1'].toFixed(2)}×`} label="and for brand-new customers with no history, where ordinary methods cannot work at all" />
        <Stat num={int(r.warm.n + r.cold.n)} label="customers used to test it" />
      </div>
      <div className="section card">
        <GroupedBarChart
          data={bars}
          options={{
            ...chartBase, height: '400px', title: 'How often the hidden product was recommended first',
            legend: { truncation: { type: 'none' } },
            axes: { left: { mapsTo: 'value', title: 'Correct first guess (%)' }, bottom: { mapsTo: 'key', scaleType: 'labels' } },
          }}
        />
      </div>
      <Meaning>
        “People like you” methods need a purchase history, so for a new customer they fall back to guessing the popular product (same score as the baseline).
        Using the customer’s age and wealth group instead lifts the success rate by about half. Blending both signals is best when a history exists.
      </Meaning>
      <Caveat title="Important: this data was generated">
        The public dataset says <em>how many</em> products each customer has, not <em>which</em> ones. We generated the product list, with realistic patterns
        (young people take overdrafts, home loans come with insurance), keeping each customer’s real product count.
        So this page shows that the method works, not how well it would sell real products.
      </Caveat>
    </>
  );
}
