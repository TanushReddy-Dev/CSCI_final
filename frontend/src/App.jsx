import React from 'react';
import ScreeningLab from './components/ScreeningLab';

function App() {
  return (
    <div className="app-container">
      {/* Premium Header */}
      <header className="glass-panel" style={{ margin: '1rem 2rem', padding: '1rem 2rem', position: 'sticky', top: '1rem', zIndex: 50 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <img src="/images/logo.jpg" alt="ConcreteGuard Logo" style={{ width: '40px', height: '40px', borderRadius: '10px' }} />
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0 }}>
                ConcreteGuard <span className="text-gradient">v2.4</span>
              </h1>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>Explainable AI Visual Inspection</p>
            </div>
          </div>
          
          <nav style={{ display: 'flex', gap: '2rem' }}>
            <a href="#" style={{ color: 'var(--text-primary)', textDecoration: 'none', fontWeight: 600, borderBottom: '2px solid var(--accent-primary)', paddingBottom: '4px' }}>Screening Lab</a>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Evidence Viewer</a>
            <a href="#" style={{ color: 'var(--text-secondary)', textDecoration: 'none' }}>Model Evaluation</a>
          </nav>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="badge badge-success">
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#34d399' }} className="animate-pulse"></span>
              Engine Ready
            </div>
            <img src="/images/profile.jpg" alt="Profile" style={{ width: '36px', height: '36px', borderRadius: '50%', border: '2px solid var(--glass-border)' }} />
          </div>
        </div>
      </header>

      <main className="container" style={{ padding: '2rem' }}>
        <div style={{ marginBottom: '3rem', textAlign: 'center', maxWidth: '800px', margin: '0 auto 3rem auto' }}>
          <div className="badge badge-neutral" style={{ marginBottom: '1rem' }}>Triage Stage 01 • ISO-13822</div>
          <h2 style={{ fontSize: '3rem', marginBottom: '1rem', fontWeight: 800 }}>
            Identify visual anomalies.<br/>
            <span className="text-gradient">Prioritize expert inspection.</span>
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            Upload concrete surface imagery to run transfer-learning CNN screening with Grad-CAM visual evidence, uncertainty estimation, and image quality verification.
          </p>
        </div>

        <ScreeningLab />
      </main>
      
      <footer style={{ marginTop: '4rem', padding: '2rem', borderTop: '1px solid var(--glass-border)', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        <p>© 2026 ConcreteGuard Consortium. Not a substitute for structural testing.</p>
      </footer>
    </div>
  );
}

export default App;
