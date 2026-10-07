import React from 'react';

export const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
export const money = (x) => `€${Math.round(x).toLocaleString('en-US')}`;
export const int = (x) => Math.round(x).toLocaleString('en-US');

export const ACTION = {
  FRAUD_REVIEW: { label: 'Check for fraud', type: 'red' },
  RETAIN_HIGH_VALUE: { label: 'Personal retention call', type: 'purple' },
  RETAIN_STANDARD: { label: 'Send retention offer', type: 'blue' },
  CROSS_SELL: { label: 'Offer a new product', type: 'green' },
};

export const PRODUCT = {
  savings: 'a savings account', fixed_deposit: 'a fixed deposit', personal_loan: 'a personal loan',
  home_loan: 'a home loan', insurance: 'insurance', investment: 'an investment account', overdraft: 'an overdraft',
};
export const productName = (p) => p.replace('_', ' ');

export const SEGMENT = (s) => s.replace('-active', ' · active customers').replace('-inactive', ' · inactive customers');

export const Meaning = ({ title = 'What this means', children }) => (
  <div className="meaning"><strong>{title}</strong>{children}</div>
);
export const Caveat = ({ title = 'Keep in mind', children }) => (
  <div className="caveat"><strong>{title}</strong>{children}</div>
);
export const Stat = ({ num, label }) => (
  <div className="stat"><div className="num">{num}</div><div className="label">{label}</div></div>
);

export const chartBase = { height: '340px', theme: 'white', toolbar: { enabled: false }, resizable: true };
