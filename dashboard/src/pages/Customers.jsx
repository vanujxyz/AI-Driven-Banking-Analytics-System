import React, { useMemo, useState } from 'react';
import {
  DataTable, Table, TableHead, TableRow, TableHeader, TableBody, TableCell, TableContainer, Search, Dropdown,
  Pagination, Tag, Modal,
} from '@carbon/react';
import { ACTION, PRODUCT, int, money, pct, productName } from '../lib.jsx';

const HEADERS = [
  { key: 'rank', header: 'Priority' }, { key: 'id', header: 'Customer ID' }, { key: 'country', header: 'Country' },
  { key: 'age', header: 'Age' }, { key: 'leave', header: 'Chance of leaving' }, { key: 'worth', header: 'Value, allowing for leaving' },
  { key: 'action', header: 'Suggested action' },
];
const FILTERS = [{ id: 'ALL', text: 'All actions' }, ...Object.entries(ACTION).map(([id, a]) => ({ id, text: a.label }))];

const actionText = (r) => (r.action === 'CROSS_SELL' ? `Offer ${productName(r.rec_product)}` : ACTION[r.action].label);

function explain(r) {
  const lines = [
    `Customer ${r.customer_id} is ${r.Age} years old, lives in ${r.Geography}, has been with the bank ${r.Tenure} ${r.Tenure === 1 ? 'year' : 'years'} and holds ${r.NumOfProducts} ${r.NumOfProducts === 1 ? 'product' : 'products'}.`,
    `Chance of leaving: ${pct(r.churn_prob)} (risk group ${r.churn_decile} of 10, where 1 is the riskiest).`,
    `Expected chance of still being a customer in 5 years: ${pct(r.surv_5y)}.`,
    `Worth about ${money(r.clv_pred)} over 5 years, or ${money(r.sa_clv)} after allowing for the chance they leave.`,
  ];
  if (r.fraud_flagged) lines.push('Our payment-checking model flagged unusual activity on this customer’s account, so it comes first.');
  lines.push(`Next best product to offer: ${PRODUCT[r.rec_product]}.`);
  lines.push(r.Exited ? 'In the data, this customer did leave the bank.' : 'In the data, this customer stayed.');
  return lines;
}

export default function Customers({ data }) {
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState(FILTERS[0]);
  const [page, setPage] = useState(1);
  const [size, setSize] = useState(10);
  const [open, setOpen] = useState(null);
  const rows = data.action_list;

  const filtered = useMemo(() => rows.filter((r) =>
    (filter.id === 'ALL' || r.action === filter.id) && (!q.trim() || String(r.customer_id).includes(q.trim()))), [rows, q, filter]);
  const slice = filtered.slice((page - 1) * size, page * size);
  const byId = useMemo(() => Object.fromEntries(rows.map((r) => [String(r.customer_id), r])), [rows]);
  const tableRows = slice.map((r) => ({
    id: String(r.customer_id), rank: int(r.priority_rank), country: r.Geography, age: r.Age, leave: pct(r.churn_prob),
    worth: money(r.sa_clv), action: r.action,
  }));

  return (
    <>
      <h1>Customer list</h1>
      <p className="lede">
        Every customer, ranked by how urgently the bank should act. Fraud checks come first, then customers who are likely to leave and worth the most.
        Search by customer ID, filter by action, and click a row for a plain-English summary.
      </p>
      <div className="toolbar">
        <Search size="lg" labelText="Search" placeholder="Search by customer ID" value={q}
          onChange={(e) => { setQ(e.target.value); setPage(1); }} />
        <Dropdown id="action-filter" titleText="Show" label="Action" items={FILTERS} itemToString={(i) => (i ? i.text : '')}
          selectedItem={filter} onChange={({ selectedItem }) => { setFilter(selectedItem); setPage(1); }} />
      </div>
      <DataTable rows={tableRows} headers={HEADERS}>
        {({ rows: rs, headers, getTableProps, getHeaderProps, getRowProps }) => (
          <TableContainer title={`${int(filtered.length)} ${filtered.length === 1 ? 'customer' : 'customers'}`}>
            <Table {...getTableProps()}>
              <TableHead>
                <TableRow>{headers.map((h) => <TableHeader {...getHeaderProps({ header: h })} key={h.key}>{h.header}</TableHeader>)}</TableRow>
              </TableHead>
              <TableBody>
                {rs.map((row) => (
                  <TableRow {...getRowProps({ row })} key={row.id} className="row-click" onClick={() => setOpen(byId[row.id])}>
                    {row.cells.map((c) => (
                      <TableCell key={c.id}>
                        {c.info.header === 'action'
                          ? <Tag type={ACTION[c.value].type} size="md">{actionText(byId[row.id])}</Tag> : c.value}
                      </TableCell>))}
                  </TableRow>))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </DataTable>
      <Pagination page={page} pageSize={size} pageSizes={[10, 25, 50]} totalItems={filtered.length}
        onChange={({ page: p, pageSize }) => { setPage(p); setSize(pageSize); }} />
      <Modal open={!!open} passiveModal modalHeading={open ? `Customer ${open.customer_id}` : ''}
        modalLabel={open ? `Priority ${int(open.priority_rank)} of ${int(rows.length)}` : ''} onRequestClose={() => setOpen(null)}>
        {open && (
          <div className="detail">
            <p><Tag type={ACTION[open.action].type}>{actionText(open)}</Tag></p>
            {explain(open).map((t, i) => <p key={i}>{t}</p>)}
          </div>)}
      </Modal>
    </>
  );
}
