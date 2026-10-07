import React, { useEffect, useMemo, useState } from 'react';

const useWide = () => {
  const q = '(min-width: 1056px)';
  const [w, setW] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const f = () => setW(m.matches);
    m.addEventListener('change', f);
    return () => m.removeEventListener('change', f);
  }, []);
  return w;
};
import {
  Header, HeaderContainer, HeaderMenuButton, HeaderName, SideNav, SideNavItems, SideNavLink, Content,
  Loading, InlineNotification,
} from '@carbon/react';
import Overview from './pages/Overview.jsx';
import Churn from './pages/Churn.jsx';
import Fraud from './pages/Fraud.jsx';
import Value from './pages/Value.jsx';
import Stay from './pages/Stay.jsx';
import Recs from './pages/Recs.jsx';
import Customers from './pages/Customers.jsx';
import About from './pages/About.jsx';

const PAGES = [
  ['overview', 'Overview', Overview],
  ['churn', 'Who might leave', Churn],
  ['fraud', 'Suspicious payments', Fraud],
  ['value', 'What customers are worth', Value],
  ['stay', 'How long they stay', Stay],
  ['recs', 'What to offer next', Recs],
  ['customers', 'Customer list', Customers],
  ['about', 'About the data', About],
];

const fromHash = () => {
  const h = window.location.hash.replace('#', '');
  return PAGES.some((p) => p[0] === h) ? h : 'overview';
};

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(fromHash());
  const wide = useWide();

  useEffect(() => {
    fetch('./data.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setData).catch((e) => setError(String(e)));
    const on = () => { setPage(fromHash()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);

  const Page = useMemo(() => PAGES.find((p) => p[0] === page)[2], [page]);

  return (
    <HeaderContainer
      render={({ isSideNavExpanded, onClickSideNavExpand }) => (
        <>
          <Header aria-label="Bank Customer Insights">
            {!wide && <HeaderMenuButton aria-label="Open menu" isActive={isSideNavExpanded} onClick={onClickSideNavExpand} />}
            <HeaderName href="#overview" prefix="Fictional Bank">Customer Insights</HeaderName>
          </Header>
          <SideNav aria-label="Pages" expanded={wide || isSideNavExpanded} isFixedNav isChildOfHeader={!wide} onSideNavBlur={wide ? undefined : onClickSideNavExpand}>
            <SideNavItems>
              {PAGES.map(([id, label]) => (
                <SideNavLink key={id} href={`#${id}`} isActive={page === id}>{label}</SideNavLink>
              ))}
            </SideNavItems>
          </SideNav>
          <Content style={{ marginLeft: wide ? '16rem' : 0 }}>
            <div className="page">
              {error && <InlineNotification kind="error" title="Could not load data" subtitle={`${error}. Run the pipeline and src/export_dashboard.py first.`} hideCloseButton />}
              {!data && !error && <Loading withOverlay={false} description="Loading" />}
              {data && <Page data={data} />}
              <p className="footer-note">
                All figures come from an offline analysis of public and simulated datasets for a fictional bank. They illustrate a method and are not real customer data.
              </p>
            </div>
          </Content>
        </>
      )}
    />
  );
}
