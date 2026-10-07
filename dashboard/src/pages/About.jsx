import React from 'react';
import { Caveat } from '../lib.jsx';

export default function About({ data }) {
  const p = data.provenance;
  return (
    <>
      <h1>About the data</h1>
      <p className="lede">
        This is a research prototype for a fictional bank. It is important to know which parts come from real-world sources and which parts we had to create.
      </p>
      <table className="plain-table">
        <thead><tr><th>Piece</th><th>Where it comes from</th><th>Real or generated?</th></tr></thead>
        <tbody>
          <tr><td>Customer profiles (10,000)</td><td>Public Kaggle “Bank Customer Churn” dataset</td><td>Published data, widely believed to be partly synthetic</td></tr>
          <tr><td>Payments ({p.transactions.rows.toLocaleString('en-US')} sampled)</td><td>Public Kaggle “PaySim” dataset</td><td>Produced by a payments simulator</td></tr>
          <tr><td>Which customer made which payment</td><td>Our pipeline</td><td><strong>Generated</strong> (random assignment)</td></tr>
          <tr><td>Payment locations and customer home cities</td><td>Our pipeline</td><td><strong>Generated</strong></td></tr>
          <tr><td>Which products each customer holds</td><td>Our pipeline</td><td><strong>Generated</strong>, matching each customer’s real product count</td></tr>
          <tr><td>“Customer worth” formula</td><td>Our assumption</td><td>An estimate, not accounting data</td></tr>
        </tbody>
      </table>
      <Caveat title="What that means for you">
        Because the payments and the customer profiles come from different sources, a customer’s fraud flag has no real connection to their chance of leaving.
        The combined list shows <em>how a bank could join these analyses together</em>; it does not discover a real relationship.
        Likewise, the recommendation scores measure how well the method works on generated data.
      </Caveat>
      <h2 style={{ marginTop: '2rem' }}>Other things we did to keep it honest</h2>
      <ul style={{ lineHeight: 1.7, paddingLeft: '1.25rem' }}>
        {p.caveats.map((c, i) => <li key={i}>{c}</li>)}
      </ul>
      <p style={{ marginTop: '1.5rem', lineHeight: 1.6 }}>
        Project by Vanuj Gangrade and Aryan Bhandari, VIT Vellore, guided by Dr. Gopinath M. Full technical results are in <code>docs/RESULTS.md</code>.
        The interface uses the open-source IBM Carbon Design System.
      </p>
    </>
  );
}
