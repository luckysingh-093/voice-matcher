import streamlit as st
from speechbrain.inference import SpeakerRecognition
import torchaudio  
from pydub import AudioSegment
import plotly.graph_objects as go
import plotly.express as px
import io
import os
import time
import numpy as np
from datetime import datetime

def convert_to_wav(input_file, output_wav_path):

    if hasattr(input_file, "seek"):
        input_file.seek(0)
    audio = AudioSegment.from_file(input_file)
    audio.export(output_wav_path, format="wav")

st.set_page_config(
    page_title="Voice Matcher - Audio Forensics", 
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .upload-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        margin: 1rem 0;
        transition: all 0.3s ease;
        color: #333333 !important;
    }
    .upload-section:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    .upload-section h4 {
        color: #333333 !important;
        margin-bottom: 0.5rem !important;
    }
    .upload-section p {
        color: #666666 !important;
        margin-bottom: 0 !important;
    }
    .upload-section * {
        color: #333333 !important;
    }
    .success-banner {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-banner {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .danger-banner {
        background: linear-gradient(90deg, #fc466b 0%, #3f5efb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_verification_model():
    """Load the SpeechBrain model with caching"""
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )


st.markdown("""
<div class="main-header">
    <h1> Voice Matcher</h1>
    <p>Advanced Audio Forensics & Speaker Verification System</p>
    <small>Powered by SpeechBrain ECAPA-TDNN Neural Networks</small>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### System Information")
    st.info("""
    **Model**: ECAPA-TDNN  
    **Training Data**: VoxCeleb  
    **Accuracy**: 95%+ on test data  
    **Processing**: Real-time analysis  
    """)
    
    st.markdown("###  Confidence Levels")
    st.markdown("""
     **Strong Match** (≥0.80)  
    High confidence same speaker
    
     **Possible Match** (0.60-0.79)  
    Inconclusive, needs more data
    
     **No Match** (<0.60)  
    Likely different speakers
    """)
    
    st.markdown("### Best Practices")
    st.markdown("""
    • Clear audio, minimal noise
    • 3+ seconds of speech  
    • Same language preferred
    • Avoid music/overlaps
    • Multiple samples recommended
    """)
    
    st.markdown("---")
    st.markdown("### Session Stats")
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    st.metric("Analyses Performed", st.session_state.analysis_count)


with st.spinner(" Loading AI model..."):
    verification = load_verification_model()


st.markdown("##  Upload Audio Samples")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="upload-section">
        <h4> Suspect Voice Sample</h4>
        <p>Upload the suspect's voice recording</p>
    </div>
    """, unsafe_allow_html=True)
    
    suspect = st.file_uploader(
        "Choose suspect audio file", 
        type=["wav", "mp3", "m4a", "flac"], 
        key="suspect",
        help="Supported formats: WAV, MP3, M4A, FLAC"
    )
    
    if suspect:
        st.audio(suspect, format="audio/wav")
        file_size = suspect.size / 1024  # KB
        st.caption(f" {suspect.name} ({file_size:.1f} KB)")

with col2:
    st.markdown("""
    <div class="upload-section">
        <h4> Evidence Voice Sample</h4>
        <p>Upload the evidence voice recording</p>
    </div>
    """, unsafe_allow_html=True)
    
    evidence = st.file_uploader(
        "Choose evidence audio file", 
        type=["wav", "mp3", "m4a", "flac"], 
        key="evidence",
        help="Supported formats: WAV, MP3, M4A, FLAC"
    )
    
    if evidence:
        st.audio(evidence, format="audio/wav")
        file_size = evidence.size / 1024  # KB
        st.caption(f" {evidence.name} ({file_size:.1f} KB)")


if suspect and evidence:

    max_size_mb = 50
    suspect_size_mb = suspect.size / (1024 * 1024)
    evidence_size_mb = evidence.size / (1024 * 1024)
    
    if suspect_size_mb > max_size_mb or evidence_size_mb > max_size_mb:
        st.error(f" File too large! Maximum size: {max_size_mb}MB")
    else:
        
        if st.button(" **Start Voice Analysis**", type="primary", use_container_width=True):
            try:
              
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Loading files
                status_text.text(" Loading audio files...")
                progress_bar.progress(20)
                
                timestamp = str(int(time.time() * 1000))
                suspect_filename = f"suspect_{timestamp}.wav"
                evidence_filename = f"evidence_{timestamp}.wav"
                
                # Step 2: Processing audio
                status_text.text(" Processing audio data...")
                progress_bar.progress(40)
                
                suspect_io = io.BytesIO(suspect.read())
                evidence_io = io.BytesIO(evidence.read())
                
                convert_to_wav(suspect_io, suspect_filename)
                convert_to_wav(evidence_io, evidence_filename)

            
                
                # Step 3: Preparing for analysis
                status_text.text(" Preparing for neural network analysis...")
                progress_bar.progress(60)
                
                waveform1, sr1 = torchaudio.load(suspect_filename)
                waveform2, sr2 = torchaudio.load(evidence_filename)
                
                # Step 4: AI Analysis
                status_text.text(" Running AI voice comparison...")
                progress_bar.progress(80)
                
                score, prediction = verification.verify_files(suspect_filename, evidence_filename)
                
                # Step 5: Complete
                status_text.text(" Analysis complete!")
                progress_bar.progress(100)
                time.sleep(0.5)  # Brief pause for effect
                
               
                progress_bar.empty()
                status_text.empty()
                
                
                st.session_state.analysis_count += 1
                
                
                st.markdown("---")
                st.markdown("## Analysis Results")
                
                score_float = float(score)
                
               
                if score_float >= 0.80:
                    st.markdown(f"""
                    <div class="success-banner">
                        <h3>STRONG MATCH DETECTED</h3>
                        <p>High confidence that both voices belong to the same speaker</p>
                        <h2>{score_float:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    level_color = "green"
                    confidence = "Strong Match"
                    
                elif 0.60 <= score_float < 0.80:
                    st.markdown(f"""
                    <div class="warning-banner">
                        <h3>POSSIBLE MATCH</h3>
                        <p>Voices are somewhat similar, More samples needed..</p>
                        <h2>{score_float:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    level_color = "orange"
                    confidence = "Possible Match"
                    
                else:
                    st.markdown(f"""
                    <div class="danger-banner">
                        <h3>NO MATCH</h3>
                        <p>Voices likely belong to different speakers</p>
                        <h2>{score_float:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    level_color = "red"
                    confidence = "No Match"
                
             
                col1, col2, col3, col4 = st.columns(4)
                
                duration1 = waveform1.shape[1] / sr1
                duration2 = waveform2.shape[1] / sr2
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Similarity Score", f"{score_float:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Confidence", confidence)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Suspect Duration", f"{duration1:.1f}s")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Evidence Duration", f"{duration2:.1f}s")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                
                st.markdown("###  Visual Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                 
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=score_float,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"<b>Confidence Level</b><br><span style='font-size:0.8em;color:gray'>{confidence}</span>"},
                        delta={'reference': 0.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': level_color, 'thickness': 0.8},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.6], 'color': '#ffe6e6'},
                                {'range': [0.6, 0.8], 'color': '#fff8e1'},
                                {'range': [0.8, 1], 'color': '#e8f5e8'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.8
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        height=350,
                        font={'color': "darkblue", 'family': "Arial"},
                        paper_bgcolor="white"
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with viz_col2:
            
                    categories = ['No Match\n(<0.60)', 'Possible\n(0.60-0.79)', 'Strong Match\n(≥0.80)']
                    thresholds = [0.6, 0.8, 1.0]
                    colors = ['#ff6b6b' if score_float < 0.6 else '#ffcccc',
                             '#ffd93d' if 0.6 <= score_float < 0.8 else '#fff8e1',
                             '#6bcf7f' if score_float >= 0.8 else '#e8f5e8']
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(x=categories, y=thresholds, marker_color=colors, opacity=0.8)
                    ])
                    
                    fig_bar.add_hline(
                        y=score_float, 
                        line_dash="dash", 
                        line_color="blue", 
                        line_width=3,
                        annotation_text=f"Your Score: {score_float:.4f}",
                        annotation_position="top right"
                    )
                    
                    fig_bar.update_layout(
                        title="<b>Score Classification</b>",
                        yaxis_title="Threshold",
                        height=350,
                        showlegend=False,
                        plot_bgcolor="white"
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
           
                with st.expander(" **Detailed Technical Analysis**", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Suspect Audio Analysis")
                        st.write(f"**File Name**: {suspect.name}")
                        st.write(f"**Duration**: {duration1:.3f} seconds")
                        st.write(f"**Sample Rate**: {sr1:,} Hz")
                        st.write(f"**Channels**: {waveform1.shape[0]}")
                        st.write(f"**Total Samples**: {waveform1.shape[1]:,}")
                        st.write(f"**File Size**: {suspect.size:,} bytes")
                    
                    with col2:
                        st.markdown("#### Evidence Audio Analysis")
                        st.write(f"**File Name**: {evidence.name}")
                        st.write(f"**Duration**: {duration2:.3f} seconds")
                        st.write(f"**Sample Rate**: {sr2:,} Hz")
                        st.write(f"**Channels**: {waveform2.shape[0]}")
                        st.write(f"**Total Samples**: {waveform2.shape[1]:,}")
                        st.write(f"**File Size**: {evidence.size:,} bytes")
                    
                    st.markdown("---")
                    st.markdown("#### AI Model Information")
                    st.write(f"**Analysis Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Similarity Score (Raw)**: {score_float:.8f}")
                    st.write(f"**Binary Classification**: {'Same Speaker' if prediction else 'Different Speakers'}")
                    st.write(f"**Model Architecture**: ECAPA-TDNN (Emphasized Channel Attention)")
                    st.write(f"**Training Dataset**: VoxCeleb (1000+ speakers)")
                    st.write(f"**Embedding Dimension**: 192")
                    st.write(f"**Confidence Threshold**: 0.80")
                
                # Export option
                st.markdown("### Export Results")
                if st.button("Generate Analysis Report", type="secondary"):
                    report = f"""
VOICE MATCHER - ANALYSIS REPORT
=====================================

Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUSPECT AUDIO:
- File: {suspect.name}
- Duration: {duration1:.3f}s
- Sample Rate: {sr1} Hz
- Channels: {waveform1.shape[0]}

EVIDENCE AUDIO:
- File: {evidence.name}  
- Duration: {duration2:.3f}s
- Sample Rate: {sr2} Hz
- Channels: {waveform2.shape[0]}

ANALYSIS RESULTS:
- Similarity Score: {score_float:.6f}
- Classification: {confidence}
- Binary Prediction: {'Same Speaker' if prediction else 'Different Speakers'}

MODEL INFORMATION:
- Architecture: ECAPA-TDNN
- Training Data: VoxCeleb
- Threshold: 0.80

INTERPRETATION:
{
'Strong Match: High confidence same speaker' if score_float >= 0.80 
else 'Possible Match: Inconclusive, more data needed' if score_float >= 0.60 
else 'No Match: Likely different speakers'
}
                    """
                    
                    st.download_button(
                        label="⬇ Download Report (.txt)",
                        data=report,
                        file_name=f"voice_analysis_report_{timestamp}.txt",
                        mime="text/plain"
                    )
                
        
                try:
                    if os.path.exists(suspect_filename):
                        os.remove(suspect_filename)
                    if os.path.exists(evidence_filename):
                        os.remove(evidence_filename)
                except Exception:
                    pass
                    
            except Exception as e:
                st.error(f" Analysis failed: {str(e)}")
                st.info("Please ensure both files are valid audio formats.")
                
       
                try:
                    if 'suspect_filename' in locals() and os.path.exists(suspect_filename):
                        os.remove(suspect_filename)
                    if 'evidence_filename' in locals() and os.path.exists(evidence_filename):
                        os.remove(evidence_filename)
                except:
                    pass

else:

    st.markdown("## Welcome to Voice Matcher")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### How It Works
        1. **Upload** two voice samples
        2. **AI Analysis** using neural networks  
        3. **Get Results** with confidence scores
        4. **Download Report** for documentation
        """)
    
    with col2:
        st.markdown("""
        ### Supported Formats
        - **WAV** (Recommended)
        - **MP3** (Most common)
        - **M4A** (Apple format)
        - **FLAC** (High quality)
        """)
    
    with col3:
        st.markdown("""
        ### File Requirements  
        - **Size**: Max 50MB per file
        - **Duration**: 3+ seconds recommended
        - **Quality**: Clear speech, minimal noise
        - **Language**: Any (same language preferred)
        """)
    
    st.info(" **Get started by uploading two audio files above**")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        Voice Matcher | Built with SpeechBrain & Streamlit | 
        For forensic and research purposes only
    </small>
</div>
""", unsafe_allow_html=True)
