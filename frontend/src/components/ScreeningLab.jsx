import React, { useState, useRef } from 'react';

export default function ScreeningLab() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const fileInputRef = useRef(null);

  const demoImages = [
    { path: '/images/demo-a104.jpg', label: 'Clear Crack (#A-104)' },
    { path: '/images/demo-b219.jpg', label: 'Hairline (#B-219)' },
    { path: '/images/demo-hn04.jpg', label: 'Stain (#HN-04)' },
    { path: '/images/demo-hn12.jpg', label: 'Joint (#HN-12)' },
    { path: '/images/demo-hn31.jpg', label: 'Shadow (#HN-31)' },
    { path: '/images/demo-dg08.jpg', label: 'Blurry (#DG-08)' },
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleDemoSelect = async (path) => {
    // Fetch the image as a Blob so we can send it
    setImagePreview(path);
    setResults(null);
    try {
      const res = await fetch(path);
      const blob = await res.blob();
      const file = new File([blob], path.split('/').pop(), { type: blob.type });
      setSelectedImage(file);
    } catch (e) {
      console.error("Failed to load demo image", e);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    setIsAnalyzing(true);
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('API Error');

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error(error);
      alert('Error connecting to backend.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
      {/* Left Column: Input */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <span className="material-symbols-outlined text-gradient" style={{ fontSize: '1.5rem' }}>file_upload</span>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Surface Capture Ingestion</h3>
          </div>
          
          <div 
            className="dropzone"
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
              accept="image/jpeg,image/png,image/webp"
            />
            {imagePreview ? (
              <img src={imagePreview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px', objectFit: 'contain' }} />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <span className="material-symbols-outlined" style={{ fontSize: '3rem', color: 'var(--accent-primary)' }}>add_photo_alternate</span>
                <p>Click to choose image or drag & drop here</p>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <span className="badge badge-neutral">.JPG</span>
                  <span className="badge badge-neutral">.PNG</span>
                </div>
              </div>
            )}
          </div>

          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem' }}>
            <button 
              className="btn-primary" 
              style={{ flex: 1, justifyContent: 'center' }}
              onClick={analyzeImage}
              disabled={!selectedImage || isAnalyzing}
            >
              {isAnalyzing ? (
                <><span className="animate-spin material-symbols-outlined">sync</span> Analyzing...</>
              ) : (
                <><span className="material-symbols-outlined">play_arrow</span> Analyze Image</>
              )}
            </button>
            <button className="btn-secondary" onClick={() => { setSelectedImage(null); setImagePreview(null); setResults(null); }}>
              <span className="material-symbols-outlined">restart_alt</span> Reset
            </button>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1rem' }}>Validation Library</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            {demoImages.map((img) => (
              <div 
                key={img.path} 
                className="glass-card" 
                style={{ padding: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}
                onClick={() => handleDemoSelect(img.path)}
              >
                <img src={img.path} alt={img.label} style={{ width: '48px', height: '48px', borderRadius: '6px', objectFit: 'cover' }} />
                <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>{img.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Column: Output */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="glass-panel animate-slide-in" style={{ padding: '2rem', minHeight: '600px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Screening Assessment</h2>
            {results && (
              <div className={`badge ${results.label.includes('crack') && !results.label.includes('no_') ? 'badge-error' : results.label.includes('no_') ? 'badge-success' : 'badge-warning'}`}>
                {results.label.replace(/_/g, ' ').toUpperCase()}
              </div>
            )}
          </div>

          {!results && !isAnalyzing && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-secondary)' }}>
              <span className="material-symbols-outlined" style={{ fontSize: '4rem', opacity: 0.5, marginBottom: '1rem' }}>analytics</span>
              <p>Awaiting image ingestion...</p>
            </div>
          )}

          {isAnalyzing && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
              <div className="animate-spin" style={{ width: '40px', height: '40px', border: '3px solid var(--glass-border)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%', marginBottom: '1rem' }}></div>
              <p className="font-mono text-gradient">Executing Dual ResNet-50 / ConvNeXt Backbone...</p>
            </div>
          )}

          {results && (
            <>
              {/* Confidence Meter */}
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Classification Confidence</span>
                  <span className="font-mono" style={{ fontWeight: 700, fontSize: '1.25rem' }}>{(results.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bg">
                  <div className="progress-fill" style={{ width: `${results.probability * 100}%` }}></div>
                </div>
              </div>

              {/* Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="glass-card" style={{ padding: '1rem' }}>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Visual Priority</div>
                  <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>{results.visual_priority.toUpperCase()}</div>
                </div>
                <div className="glass-card" style={{ padding: '1rem' }}>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Quality Status</div>
                  <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>{results.quality_status.toUpperCase()}</div>
                  <div style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>{results.quality_message}</div>
                </div>
              </div>

              {/* Action Banner */}
              <div style={{ background: 'rgba(59, 130, 246, 0.1)', borderLeft: '4px solid #3b82f6', padding: '1rem', borderRadius: '4px 8px 8px 4px' }}>
                <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Recommended Civil Action</div>
                <div style={{ fontSize: '0.9rem' }}>{results.review_recommendation}</div>
              </div>

              {/* Evidence Viewer */}
              {results.gradcam_base64 || results.gradcam_url ? (
                <div>
                  <h4 style={{ fontWeight: 600, marginBottom: '0.75rem' }}>Explainable Evidence Preview</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div style={{ position: 'relative' }}>
                      <img src={imagePreview} style={{ width: '100%', height: '160px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--glass-border)' }} />
                      <div className="badge badge-neutral" style={{ position: 'absolute', top: '8px', left: '8px' }}>RAW</div>
                    </div>
                    <div style={{ position: 'relative' }}>
                      <img src={results.gradcam_base64 ? `data:image/png;base64,${results.gradcam_base64}` : `http://localhost:8000${results.gradcam_url}`} style={{ width: '100%', height: '160px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--accent-primary)' }} />
                      <div className="badge" style={{ background: 'var(--accent-primary)', color: 'white', position: 'absolute', top: '8px', left: '8px' }}>GRAD-CAM</div>
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
