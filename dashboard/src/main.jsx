import React from 'react';
import { createRoot } from 'react-dom/client';
import '@carbon/charts-react/styles.css';
import './styles.scss';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(<App />);
