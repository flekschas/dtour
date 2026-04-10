import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';

const root = document.getElementById('root');
if (!root) throw new Error('No #root element found');

// No StrictMode: the scatter engine uses transferControlToOffscreen() and
// ArrayBuffer transfers — one-shot ownership semantics that are fundamentally
// incompatible with StrictMode's mount→cleanup→remount cycle.
createRoot(root).render(<App />);
