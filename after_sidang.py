import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Peramalan Harga Minyak Goreng Sawit di Indonesia",
    page_icon="🌴",
    layout="wide"
)

# CSS sederhana + styling untuk date inputs
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E8B57;
        color: #333;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Styling khusus untuk date inputs agar terlihat dapat diklik */
    .stDateInput > div > div {
        background-color: white;
        border: 2px solid #007bff;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,123,255,0.1);
        cursor: pointer;
    }
    .stDateInput > div > div:hover {
        border-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,123,255,0.2);
        transform: translateY(-1px);
    }
    .stDateInput label {
        font-weight: bold;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

def format_date_indonesian(date):
    months = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
        7: 'Jul', 8: 'Ags', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
    }
    if pd.isna(date):
        return ""
    return f"{months[date.month]} {date.year}"

def get_correlation_interpretation(corr_val):
    abs_corr = abs(corr_val)
    if abs_corr >= 0.7:
        return "Sangat Kuat"
    elif abs_corr >= 0.5:
        return "Kuat" 
    elif abs_corr >= 0.3:
        return "Sedang"
    else:
        return "Lemah"

def main():
    # Header
    st.markdown('<h1 class="main-title">🌴 Dashboard Peramalan Harga Minyak Goreng Sawit di Indonesia</h1>', unsafe_allow_html=True)
    
    # Data loader
    @st.cache_data
    def load_data():
        try:
            data = pd.read_excel("dataset_kemasan_baru.xlsx")
            data['Bulan'] = pd.to_datetime(data['Bulan'])
            data['Periode'] = data['Bulan'].apply(format_date_indonesian)
            return data
        except:
            st.warning("File tidak ditemukan. Menggunakan data contoh.")
            dates = pd.date_range(start='2012-01-01', end='2024-12-31', freq='M')
            np.random.seed(42)
            dummy_data = pd.DataFrame({
                'Bulan': dates,
                'Harga_Kemasan': np.random.normal(15000, 2000, len(dates)).astype(int),
                'Harga_CPO': np.random.normal(800, 150, len(dates)).astype(int),
                'Produksi_CPO': np.random.normal(4500, 500, len(dates)).astype(int),
                'IGT': np.random.randint(20, 100, len(dates))
            })
            dummy_data['Periode'] = dummy_data['Bulan'].apply(format_date_indonesian)
            return dummy_data
    
    data = load_data()
    
    # Navigation with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📈 Visualisasi", 
        "📋 Korelasi", 
        "🔮 Model", 
        "📊 Hasil Peramalan", 
        "🎯 Simulasi Peramalan"
    ])
    
    # Tab Overview
    with tab1:
        st.header("📊 Overview Data")
        
        if not data.empty:
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_records = len(data)
                st.markdown(f'''
                <div class="metric-box">
                    <h4>Total Records</h4>
                    <h3>{total_records:,}</h3>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if 'Harga_Kemasan' in data.columns:
                    avg_price = data['Harga_Kemasan'].mean()
                    st.markdown(f'''
                    <div class="metric-box">
                        <h4>Harga Rata-rata</h4>
                        <h3>Rp {avg_price:,.0f}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col3:
                missing_data = data.isnull().sum().sum()
                st.markdown(f'''
                <div class="metric-box">
                    <h4>Missing Values</h4>
                    <h3>{missing_data}</h3>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                if len(data) > 1:
                    first_period = data['Periode'].iloc[0]
                    last_period = data['Periode'].iloc[-1]
                    st.markdown(f'''
                    <div class="metric-box">
                        <h4>Periode Data</h4>
                        <h3 style="font-size: 1rem;">{first_period} - {last_period}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Date range selection for data table
            st.subheader("📋 Tabel Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_date_table = st.date_input(
                    "📅 Tanggal Mulai", 
                    value=data['Bulan'].min(),
                    min_value=data['Bulan'].min().date(),
                    max_value=data['Bulan'].max().date(),
                    key="table_start"
                )
            
            with col2:
                end_date_table = st.date_input(
                    "📅 Tanggal Akhir", 
                    value=data['Bulan'].max(),
                    min_value=data['Bulan'].min().date(),
                    max_value=data['Bulan'].max().date(),
                    key="table_end"
                )
            
            # Filter data based on date selection
            filtered_table_data = data[
                (data['Bulan'] >= pd.Timestamp(start_date_table)) & 
                (data['Bulan'] <= pd.Timestamp(end_date_table))
            ]
            
            display_cols = ['Periode'] + [col for col in filtered_table_data.columns if col not in ['Periode', 'Bulan']]
            st.dataframe(filtered_table_data[display_cols], use_container_width=True, hide_index=True)
            
            # Download
            csv = data.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # Tab Visualisasi
    with tab2:
        st.header("📈 Visualisasi Data")
        
        if not data.empty:
            variables = {
                'Harga_Kemasan': {'title': 'Harga Minyak Goreng Kemasan', 'unit': 'Rupiah/liter', 'color': '#2E8B57'},
                'Harga_CPO': {'title': 'Harga CPO Internasional', 'unit': 'USD/MT', 'color': '#FF6B35'},
                'Produksi_CPO': {'title': 'Produksi CPO Nasional', 'unit': 'Ton', 'color': '#4ECDC4'},
                'IGT': {'title': 'Indeks Google Trends', 'unit': 'Index (0-100)', 'color': '#45B7D1'}
            }
            
            # Date filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("📅 Dari", data['Bulan'].min(), key="viz_start")
            with col2:
                end_date = st.date_input("📅 Sampai", data['Bulan'].max(), key="viz_end")
            
            filtered_data = data[(data['Bulan'] >= pd.Timestamp(start_date)) & 
                               (data['Bulan'] <= pd.Timestamp(end_date))]
            
            # Individual plots for each variable (2 columns layout)
            variable_list = [(var_name, var_info) for var_name, var_info in variables.items() if var_name in filtered_data.columns]
            
            for i in range(0, len(variable_list), 2):
                col1, col2 = st.columns(2)
                
                # First variable in row
                with col1:
                    var_name, var_info = variable_list[i]
                    st.markdown(f"<h5 style='margin-bottom: 0.5rem;'>📈 {var_info['title']} ({var_info['unit']})</h5>", unsafe_allow_html=True)
                    
                    # Create individual plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=filtered_data['Bulan'],
                        y=filtered_data[var_name],
                        mode='lines',
                        name=var_info['title'],
                        line=dict(color=var_info['color'], width=2),
                        hovertemplate=f"<b>Periode</b>: %{{x}}<br><b>{var_info['title']}</b>: %{{y:,.0f}} {var_info['unit']}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Periode",
                        yaxis_title=f"{var_info['unit']}",
                        height=280,
                        plot_bgcolor='white',
                        hovermode='x unified',
                        margin=dict(t=20, b=30, l=40, r=20),
                        yaxis=dict(rangemode='tozero'),
                        font=dict(size=10),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics for first variable (mini boxes for visualization only)
                    min_val = filtered_data[var_name].min()
                    max_val = filtered_data[var_name].max()
                    avg_val = filtered_data[var_name].mean()
                    std_val = filtered_data[var_name].std()
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.markdown(f'''
                        <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                            <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Minimal</div>
                            <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{min_val:,.0f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f'''
                        <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                            <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Maksimal</div>
                            <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{max_val:,.0f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f'''
                        <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                            <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Rata-rata</div>
                            <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{avg_val:,.0f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col_d: # Menambahkan kolom untuk standar deviasi
                        st.markdown(f'''
                        <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                            <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Std Deviasi</div>
                            <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{std_val:,.0f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Second variable in row (if exists)
                if i + 1 < len(variable_list):
                    with col2:
                        var_name, var_info = variable_list[i + 1]
                        st.markdown(f"<h5 style='margin-bottom: 0.5rem;'>📈 {var_info['title']} ({var_info['unit']})</h5>", unsafe_allow_html=True)
                        
                        # Create individual plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=filtered_data['Bulan'],
                            y=filtered_data[var_name],
                            mode='lines',
                            name=var_info['title'],
                            line=dict(color=var_info['color'], width=2),
                            hovertemplate=f"<b>Periode</b>: %{{x}}<br><b>{var_info['title']}</b>: %{{y:,.0f}} {var_info['unit']}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Periode",
                            yaxis_title=f"{var_info['unit']}",
                            height=280,
                            plot_bgcolor='white',
                            hovermode='x unified',
                            margin=dict(t=20, b=30, l=40, r=20),
                            yaxis=dict(rangemode='tozero'),
                            font=dict(size=10),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics for second variable (mini boxes for visualization only)
                        min_val = filtered_data[var_name].min()
                        max_val = filtered_data[var_name].max()
                        avg_val = filtered_data[var_name].mean()
                        std_val = filtered_data[var_name].std()
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.markdown(f'''
                            <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                                <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Minimal</div>
                                <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{min_val:,.0f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f'''
                            <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                                <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Maksimal</div>
                                <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{max_val:,.0f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        with col_c:
                            st.markdown(f'''
                            <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                                <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Rata-rata</div>
                                <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{avg_val:,.0f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        with col_d: # Menambahkan kolom untuk standar deviasi
                            st.markdown(f'''
                            <div class="metric-box" style="padding: 0.25rem; margin: 0.1rem 0; line-height: 1.1;">
                                <div style="font-size: 0.75rem; margin: 0; line-height: 1; color: #666;">Std Deviasi</div>
                                <div style="font-weight: bold; font-size: 0.9rem; margin: 0; line-height: 1;">{std_val:,.0f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                
                # Remove separator between rows
                if i + 2 < len(variable_list):
                    st.markdown("---")
    
    # Tab Korelasi
    with tab3:
        st.header("📋 Analisis Korelasi")
        
        if not data.empty:
            st.markdown('''
            <div class="info-panel">
                <h4 style="margin-bottom: 0.5rem;">📊 Cross-Correlation Function (CCF)</h4>
                <p>
                    <b>Cross-Correlation Function (CCF)</b> adalah metode untuk mengukur korelasi antara dua data deret waktu (<i>time series</i>) pada pergeseran waktu yang berbeda (disebut <i>lag</i>). Pada dasarnya, CCF menerapkan perhitungan Korelasi Pearson pada setiap lag untuk melihat seberapa kuat hubungan satu variabel dengan variabel lain di masa lalu atau masa depan.
                </p>
                <p>Perhitungan korelasi pada setiap lag menggunakan <b>Rumus Korelasi Pearson</b>:</p>
            </div>
            ''', unsafe_allow_html=True)

            st.latex(r'''
            r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
            ''')

            st.markdown('''
            <div class="info-panel">
                <p style="margin-bottom: 0;">
                    <b>Tujuan Utama:</b> Tujuan dari analisis CCF ini adalah untuk mengidentifikasi <b>lag (pergeseran bulan)</b> di mana setiap variabel eksogen (Harga CPO, Produksi CPO, Indeks Google) memiliki korelasi tertinggi terhadap harga minyak goreng sawit. Lag dengan korelasi tertinggi inilah yang akan digunakan sebagai fitur dalam model peramalan untuk meningkatkan akurasi prediksi.
                </p>
            </div>
            ''', unsafe_allow_html=True)
            
            # CCF Analysis
            ccf_results = []
            exogenous_vars = {
                'Harga CPO Internasional': 'Harga_CPO',
                'Produksi CPO Nasional': 'Produksi_CPO', 
                'Indeks Google Trends': 'IGT'
            }
            
            target_col = 'Harga_Kemasan'
            
            for var_name, col_name in exogenous_vars.items():
                if col_name in data.columns and target_col in data.columns:
                    try:
                        x = data[col_name].dropna()
                        y = data[target_col].dropna()
                        
                        common_index = x.index.intersection(y.index)
                        x_aligned = x.loc[common_index]
                        y_aligned = y.loc[common_index]
                        
                        if len(x_aligned) > 24:
                            ccf_vals = ccf(x_aligned, y_aligned)
                            
                            for lag in range(1, 13):
                                if lag < len(ccf_vals):
                                    ccf_results.append({
                                        'Variabel': var_name,
                                        'Lag': lag,
                                        'Korelasi': ccf_vals[lag],
                                        'Interpretasi': get_correlation_interpretation(ccf_vals[lag])
                                    })
                    except:
                        continue
            
            if ccf_results:
                ccf_df = pd.DataFrame(ccf_results)
                ccf_pivot = ccf_df.pivot(index='Lag', columns='Variabel', values='Korelasi')
                
                # Urutkan kolom sesuai permintaan
                desired_order = ['Harga CPO Internasional', 'Produksi CPO Nasional', 'Indeks Google Trends']
                available_cols = [col for col in desired_order if col in ccf_pivot.columns]
                ccf_pivot = ccf_pivot[available_cols]
                ccf_pivot = ccf_pivot.round(4)
                
                # Pilihan tampilan
                st.subheader("📊 Visualisasi CCF")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    display_option = st.selectbox(
                        "Pilih tampilan:",
                        ["📊 Tabel", "🔥 Heatmap"]
                    )
                
                if display_option == "📊 Tabel":
                    # Table
                    st.markdown("##### Tabel CCF")
                    
                    # Find max absolute values for highlighting
                    max_positions = {}
                    for col in ccf_pivot.columns:
                        if not ccf_pivot[col].isna().all():
                            max_abs_idx = ccf_pivot[col].abs().idxmax()
                            max_positions[col] = max_abs_idx
                    
                    # Display table with custom formatting
                    col_headers = st.columns([1] + [2] * len(ccf_pivot.columns))
                    
                    # Header row
                    with col_headers[0]:
                        st.markdown("**Lag**")
                    for i, col in enumerate(ccf_pivot.columns):
                        with col_headers[i + 1]:
                            st.markdown(f"**{col}**")
                    
                    # Data rows
                    for lag in ccf_pivot.index:
                        row_cols = st.columns([1] + [2] * len(ccf_pivot.columns))
                        
                        with row_cols[0]:
                            st.markdown(f"**{lag}**")
                        
                        for i, col in enumerate(ccf_pivot.columns):
                            with row_cols[i + 1]:
                                value = ccf_pivot.loc[lag, col]
                                if pd.isna(value):
                                    st.markdown("-")
                                else:
                                    # Highlight max values
                                    if col in max_positions and lag == max_positions[col]:
                                        st.markdown(f'<div style="background-color: #D4EDDA; padding: 0.2rem; border-radius: 4px; font-weight: bold; text-align: center;">{value:.4f}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div style="text-align: center;">{value:.4f}</div>', unsafe_allow_html=True)
                    # Add legend below the table
                    st.markdown('''
                    <p style="font-weight: bold; margin-bottom: 0.25rem;">Legenda:</p>
                    <div style="margin-top: 1rem; display: flex; align-items: center; font-size: 0.9rem;">
                        <div style="width: 20px; height: 20px; background-color: #D4EDDA; border: 1px solid #ccc; margin-right: 10px;"></div>
                        <span><b>Lag dengan Korelasi tertinggi<b> per variabel Terhadap Harga Minyak Goreng Sawit</span>
                    </div>
                    ''', unsafe_allow_html=True)
                else:  # Heatmap
                    # Heatmap
                    st.markdown("##### Heatmap Korelasi")
                    
                    fig = px.imshow(
                        ccf_pivot.T,
                        title="Korelasi Lag Variabel Eksogen Terhadap Harga Minyak Goreng Sawit",
                        labels=dict(x="Lag (Bulan)", y="Variabel Eksogen", color="Korelasi"),
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        color_continuous_midpoint=0,  # Titik netral di 0
                        text_auto='.2f',
                        aspect="auto"
                    )
                    
                    fig.update_layout(
                        title_x=0.35,
                        width=800,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best lags
                st.subheader("💡 Lag yang Memiliki Korelasi Tertinggi")
                for col in ccf_pivot.columns:
                    if not ccf_pivot[col].isna().all():
                        max_abs_idx = ccf_pivot[col].abs().idxmax()
                        max_val = ccf_pivot[col].loc[max_abs_idx]
                        st.write(f"• **{col}**: Lag {max_abs_idx} bulan (korelasi: {max_val:.4f})")
    
    # Tab Model
    with tab4:
        st.header("🔮 Model Terbaik")
        
        # Model evaluation data - updated with new models
        models_data = {
            'Model': ['ARIMAX-GARCH', 'SVR', 'LSTM'],
            'RMSE': [1348.69, 6446.93, 4833.33],
            'MAPE': [5.36, 30.15, 22.77]
        }
        
        df_models = pd.DataFrame(models_data)
        
        # Best model is ARIMAX-GARCH (index 0)
        best_model_idx = 0
        best_model = df_models.loc[best_model_idx]
        
        # KPI Cards for best model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
            <div class="metric-box">
                <h4>Model Terbaik</h4>
                <h3>ARIMAX(2,1,2)-GARCH(1,1)</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-box">
                <h4>RMSE</h4>
                <h3>{best_model["RMSE"]:.2f}</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-box">
                <h4>MAPE</h4>
                <h3>{best_model["MAPE"]:.2f}%</h3>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model details
        st.subheader("⚙️ Detail Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **📈 Model yang Dibandingkan:**
            - **ARIMAX-GARCH**: AutoRegressive Integrated Moving Average with eXogenous variables dengan tambahan GARCH
            - **SVR**: Support Vector Regression
            - **LSTM**: Long Short-Term Memory
            """)
            
            st.markdown("""
            **🎛️ Hyperparameter Terbaik:**
            - **SVR**: kernel RBF, C=14, epsilon=0.13 , gamma=scale
            - **LSTM**: n_units=300, n_past=1, dropout_rate=0.1, learning_rate=0.01, batch_size=8
            """)
        
        with col2:
            st.markdown("**📋 Informasi Langkah-Langkah Pemodelan:**")
            
            # Buttons for step-by-step info
            if st.button("📊 ARIMAX-GARCH", use_container_width=True):
                st.session_state.show_arimax_steps = True
                st.session_state.show_svr_steps = False
                st.session_state.show_lstm_steps = False
            
            if st.button("🤖 SVR", use_container_width=True):
                st.session_state.show_svr_steps = True
                st.session_state.show_arimax_steps = False
                st.session_state.show_lstm_steps = False
            
            if st.button("🧠 LSTM", use_container_width=True):
                st.session_state.show_lstm_steps = True
                st.session_state.show_arimax_steps = False
                st.session_state.show_svr_steps = False
            
            if st.button("❌ Sembunyikan", use_container_width=True):
                st.session_state.show_arimax_steps = False
                st.session_state.show_svr_steps = False
                st.session_state.show_lstm_steps = False
        
        # Display step-by-step information based on button clicks
        if hasattr(st.session_state, 'show_arimax_steps') and st.session_state.show_arimax_steps:
            st.markdown("---")
            st.subheader("📊 Langkah-Langkah ARIMAX-GARCH")
            
            # Data Splitting
            st.markdown('''
            <div class="info-panel">
                <h5>🔄 Langkah 1: Data Splitting</h5>
                <p>Data dibagi menjadi data training (80%) untuk model belajar dan testing (20%) untuk evaluasi model.</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # ARIMAX Steps
            st.markdown('''
            <div class="info-panel">
                <h5>📈 Langkah 2: Pemodelan ARIMAX</h5>
                <ul>
                    <li><strong>Uji Stasioner:</strong> Melakukan uji stasioneritas pada data</li>
                    <li><strong>Differencing:</strong> Jika data tidak stasioner, lakukan differencing</li>
                    <li><strong>Identifikasi Model:</strong> Gunakan plot ACF & PACF untuk menentukan ordo</li>
                    <li><strong>Pemodelan:</strong> Coba beberapa kombinasi ordo potensial</li>
                    <li><strong>Pilih Model Terbaik:</strong> Berdasarkan nilai AIC terkecil</li>
                    <li><strong>Uji Ljung-Box:</strong> Pastikan residual adalah white noise</li>
                    <li><strong>Evaluasi:</strong> Jika residual white noise, lanjut ke GARCH</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
            
            # GARCH Steps  
            st.markdown('''
            <div class="info-panel">
                <h5>📊 Langkah 3: Pemodelan GARCH</h5>
                <ul>
                    <li><strong>Residual ARIMAX:</strong> Ambil residual dari model ARIMAX</li>
                    <li><strong>Uji ARCH-LM:</strong> Test heteroskedastisitas pada residual</li>
                    <li><strong>Pemodelan Varians Kondisional Redisual ARIMAX:</strong> Jika terdapat heteroskedastisitas, buat model GARCH</li>
                    <li><strong>Standarisasi Residual:</strong> Hitung v<sub>t</sub> = e<sub>t</sub>/(h<sub>t</sub>)<sup>0.5</sup></li>
                    <li><strong>Uji ARCH-LM:</strong> Test heteroskedastisitas pada residual yang distandarisasi</li>
                    <li><strong>Uji Normalitas:</strong> Cek apakah standarisasi residual berdistribusi normal</li>
                    <li><strong>Confidence Interval:</strong> Buat interval kepercayaan untuk peramalan</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'show_svr_steps') and st.session_state.show_svr_steps:
            st.markdown("---")
            st.subheader("🤖 Langkah-Langkah SVR")
            
            # Data Splitting
            st.markdown('''
            <div class="info-panel">
                <h5>🔄 Langkah 1: Data Splitting</h5>
                <p>Data dibagi menjadi data training (80%) untuk model belajar dan testing (20%) untuk evaluasi model.</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # SVR Steps
            st.markdown('''
            <div class="info-panel">
                <h5>🤖 Langkah 2-5: Pemodelan SVR</h5>
                <ul>
                    <li><strong>Langkah 2 - Standarisasi Data:</strong> Normalisasi menggunakan Min-Max Scaler untuk meningkatkan performa model</li>
                    <li><strong>Langkah 3 - Tuning Hyperparameter:</strong> Optimasi parameter menggunakan Grid Search:
                        <ul>
                            <li>Kernel: RBF, Polynomial, Linear</li>
                            <li>C: Mengatur seberapa besar model boleh membuat kesalahan saat belajar</li>
                            <li>Epsilon: Batas kesalahan yang masih dianggap wajar dalam prediksi</li>
                            <li>Gamma: Mengatur seberapa jauh pengaruh satu titik data dalam model (khusus untuk RBF)</li>
                            <li>Degree: Menentukan derajat pada kernel Polynomial</li>
                        </ul>
                    </li>
                    <li><strong>Langkah 3 - Pemodelan:</strong> Training model SVR dengan hyperparameter terbaik</li>
                    <li><strong>Langkah 4 - Transformasi Balik Data:</strong> Transformasi balik hasil peramalan ke skala asli</li>
                    <li><strong>Langkah 5 - Evaluasi Model:</strong> Hitung RMSE dan MAPE pada data testing</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'show_lstm_steps') and st.session_state.show_lstm_steps:
            st.markdown("---")
            st.subheader("🧠 Langkah-Langkah LSTM")
            
            # Data Splitting
            st.markdown('''
            <div class="info-panel">
                <h5>🔄 Langkah 1: Data Splitting</h5>
                <p>Data dibagi menjadi data training (80%) untuk model belajar dan testing (20%) untuk evaluasi model.</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # LSTM Steps
            st.markdown('''
            <div class="info-panel">
                <h5>🧠 Langkah 2-5: Pemodelan LSTM</h5>
                <ul>
                    <li><strong>Langkah 2 - Standarisasi Data:</strong> Normalisasi menggunakan Min-Max Scaler untuk stabilitas training</li>
                    <li><strong>Langkah 3 - Tuning Hyperparameter:</strong> Optimasi parameter menggunakan Grid Search:
                        <ul>
                            <li>n_units: Jumlah neuron dalam layer LSTM</li>
                            <li>timesteps: Jumlah data sebelumnya yang digunakan untuk memprediksi nilai berikutnya</li>
                            <li>dropout_rate: Persentase neuron yang dinonaktifkan sementara untuk mencegah overfitting</li>
                            <li>learning_rate: Seberapa cepat model belajar dari kesalahan</li>
                            <li>batch_size: Jumlah data yang diproses sekaligus dalam satu putaran pelatihan</li>
                        </ul>
                    </li>
                    <li><strong>Langkah 4 - Pemodelan:</strong> Training model LSTM dengan arsitektur neural network:
                        <ul>
                            <li>LSTM Layer dengan n_units neuron</li>
                            <li>Dropout Layer untuk mencegah overfitting</li>
                            <li>Dense Layer untuk output</li>
                        </ul>
                    </li>
                    <li><strong>Langkah 5 - Transformasi Balik Data:</strong> Transformasi balik hasil peramalan ke skala asli</li>
                    <li><strong>Langkah 6 - Evaluasi Model:</strong> Hitung RMSE dan MAPE pada data testing</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
        
        # Performance visualization options
        st.subheader("📊 Visualisasi Performa Model")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            viz_option = st.selectbox(
                "Pilih visualisasi:",
                ["📊 Tabel", "📈 Grafik"]
            )
        
        if viz_option == "📊 Tabel":
            # Table view
            st.markdown("##### Tabel Evaluasi Model")
            
            # Highlight best values
            def highlight_best_model(s):
                if s.name == 'Model':
                    return ['background-color: #D4EDDA; font-weight: bold' if v == 'ARIMAX-GARCH' else '' for v in s] 
                elif s.name == 'RMSE':
                    return ['background-color: #D4EDDA; font-weight: bold' if v == s.min() else '' for v in s]
                elif s.name == 'MAPE':
                    return ['background-color: #D4EDDA; font-weight: bold' if v == s.min() else '' for v in s]
                return ['' for _ in s]
            
            styled_df = df_models.style.apply(highlight_best_model).format({
                'RMSE': '{:.2f}',
                'MAPE': '{:.2f}%'
            })
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Legenda:**
            - 🟢 **Hijau**: Model Dengan Metrik Evaluasi Terbaik
            - **RMSE**: Root Mean Squared Error (semakin rendah semakin baik)
            - **MAPE**: Mean Absolute Percentage Error (semakin rendah semakin baik)
            """)
        
        else:  # Grafik
            # Chart view with 2 columns
            st.markdown("##### Grafik Perbandingan Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RMSE comparison
                fig_rmse = px.bar(
                    df_models, x='Model', y='RMSE',
                    title="Perbandingan RMSE",
                    color='RMSE',
                    color_continuous_scale='Reds_r'
                )
                fig_rmse.update_layout(height=350, xaxis_tickangle=-45)
                fig_rmse.update_traces(texttemplate='%{y:.2f}', textposition='auto')
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                # MAPE comparison
                fig_mape = px.bar(
                    df_models, x='Model', y='MAPE',
                    title="Perbandingan MAPE",
                    color='MAPE',
                    color_continuous_scale='Blues_r'
                )
                fig_mape.update_layout(height=350, xaxis_tickangle=-45)
                fig_mape.update_traces(texttemplate='%{y:.2f}%', textposition='auto')
                st.plotly_chart(fig_mape, use_container_width=True)
        
        st.info("🏆 **ARIMAX(2,1,2)-GARCH(1,1)** menjadi model terbaik dengan RMSE terendah 1348.69 dan MAPE terendah 5.36%. Model ini menggabungkan kemampuan ARIMAX dalam memodelkan pola temporal dengan GARCH untuk memodelkan volatilitas yang berubah-ubah.")
    
        
        # Equation button at the bottom
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📐 Lihat Persamaan ARIMAX(2,1,2)-GARCH(1,1)", use_container_width=True):
                st.session_state.show_arimax_equation = True
                st.session_state.show_arimax_steps = False
                st.session_state.show_svr_steps = False
                st.session_state.show_lstm_steps = False
        
        with col3:
            if st.button("❌ Sembunyikan Persamaan", use_container_width=True):
                st.session_state.show_arimax_equation = False
        
        # Display equation at the bottom
        if hasattr(st.session_state, 'show_arimax_equation') and st.session_state.show_arimax_equation:
            st.markdown('''
            <div class="info-panel">
            <h4 style="text-align: center; color: #2c3e50; margin-bottom: 1.5rem;">📈 Persamaan Model ARIMAX(2,1,2)-GARCH(1,1)</h4>
            
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h5 style="text-align: center; color: #34495e; margin-bottom: 1rem; font-weight: bold;">📊 Persamaan ARIMAX</h5>
                <div style="text-align: center; font-family: 'Courier New', monospace; font-size: 1rem; line-height: 2; background-color: white; padding: 1rem; border-radius: 8px; border: 2px solid #3498db;">
                    <strong>ΔHargaMiGor<sub>t</sub></strong> = 3,817 × Harga_CPO<sub>t-1</sub> - 0,075 × Produksi_CPO<sub>t-4</sub> + 51,295 × IGT<sub>t-1</sub><br>
                    + 0,817 × ΔHarga MiGor<sub>t-1</sub> - 0,711 × ΔHarga MiGor<sub>t-2</sub> - 1,217 × ε<sub>t-1</sub> + 0,777 × ε<sub>t-2</sub>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h5 style="text-align: center; color: #2d3436; margin-bottom: 1rem; font-weight: bold;">📈 Persamaan GARCH</h5>
                <div style="text-align: center; font-family: 'Courier New', monospace; font-size: 1rem; line-height: 2; background-color: white; padding: 1rem; border-radius: 8px; border: 2px solid #e17055;">
                    <strong>h<sub>t</sub></strong> = 44184 + 0,347 × ε<sub>t-1</sub><sup>2</sup> + 0,505 × h<sub>t-1</sub>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #dfe6e9 0%, #b2bec3 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h5 style="text-align: center; color: #2d3436; margin-bottom: 1rem; font-weight: bold;">📝 Keterangan</h5>
                <div style="background-color: white; padding: 1rem; border-radius: 8px; border: 2px solid #74b9ff;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                        <div>
                            <p><strong>ΔHargaMiGor<sub>t</sub></strong> = HargaMiGor<sub>t</sub> – HargaMiGor<sub>t–1</sub>
                            <p><strong>HargaCPOInter<sub>t-1</sub></strong> = Harga CPO internasional lag 1 bulan</p>
                            <p><strong>ProduksiCPO<sub>t-4</sub></strong> = Produksi CPO nasional lag 4 bulan</p>
                            <p><strong>IGT<sub>t-1</sub></strong> = Indeks Google Trends lag 1 bulan</p>
                        </div>
                        <div>
                            <p><strong>ε<sub>t</sub></strong> = Residual ARIMAX pada periode t</p>
                            <p><strong>h<sub>t</sub></strong> = Varians kondisional dari residual ARIMAX pada periode t</p>
                        </div>
                    </div>
                </div>
            </div>
            </div>
            ''', unsafe_allow_html=True)

    # Tab Hasil Peramalan
    with tab5:
        st.header("📊 Hasil Peramalan Model ARIMAX-GARCH")

        # Load forecast results data
        @st.cache_data
        def load_forecast_data():
            try:
                # Membaca data dari file excel
                data = pd.read_excel("hasil_plot_semua.xlsx")
                
                # --- PERUBAHAN 1: Nama kolom disesuaikan ---
                # Kolom 'Fitted_Train', 'Forecast_Test', 'Forecast_2025' digabung menjadi 'Peramalan'
                data.columns = ['Tanggal', 'Aktual', 'Peramalan', 'Batas_Atas', 'Batas_Bawah']
                
                # Mengonversi kolom Tanggal ke format datetime
                data['Tanggal'] = pd.to_datetime(data['Tanggal'])
                
                return data
            except FileNotFoundError:
                st.warning("File hasil_plot_semua.xlsx tidak ditemukan. Pastikan file berada di direktori yang sama.")
                return None
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat data: {e}")
                return None

        forecast_data = load_forecast_data()
        
        if forecast_data is not None and not forecast_data.empty:

            # Add dropdown for visualization choice
            col1, col2 = st.columns([1, 3])
            with col1:
                viz_option = st.selectbox(
                    "Pilih tampilan:",
                    ["📊 Visualisasi", "📋 Tabel Data"],
                    key="viz_option_tab5"
                )
            
            # --- PERUBAHAN 2: Logika Visualisasi Disesuaikan ---
            if viz_option == "📊 Visualisasi":
                fig = go.Figure()
                
                # Definisikan tanggal pemisah periode
                train_end = pd.to_datetime('2022-05-01')
                test_end = pd.to_datetime('2024-12-01')
                forecast_start = pd.to_datetime('2025-01-01')

                # Add actual data (tidak berubah)
                actual_mask = forecast_data['Aktual'].notna()
                if actual_mask.any():
                    fig.add_trace(go.Scatter(
                        x=forecast_data.loc[actual_mask, 'Tanggal'],
                        y=forecast_data.loc[actual_mask, 'Aktual'],
                        mode='lines',
                        name='Aktual',
                        line=dict(color='#000000', width=2),
                        hovertemplate="<b>Aktual</b><br>Tanggal: %{x|%b %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>"
                    ))
                
                # Buat mask untuk setiap periode
                train_mask = (forecast_data['Tanggal'] <= train_end) & forecast_data['Peramalan'].notna()
                test_mask = (forecast_data['Tanggal'] > train_end) & (forecast_data['Tanggal'] < forecast_start) & forecast_data['Peramalan'].notna()
                forecast_2025_mask = (forecast_data['Tanggal'] >= forecast_start) & forecast_data['Peramalan'].notna()

                peramalan_mask = forecast_data['Peramalan'].notna()
                if peramalan_mask.any():
                    fig.add_trace(go.Scatter(
                        x=forecast_data.loc[peramalan_mask, 'Tanggal'],
                        y=forecast_data.loc[peramalan_mask, 'Peramalan'],
                        mode='lines',
                        name='Hasil Peramalan',
                        line=dict(color='red', width=2, dash='dash'), # Garis putus-putus merah
                        hovertemplate="<b>Peramalan</b><br>Tanggal: %{x|%b %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>"
                    ))
                
                # Add confidence interval for 2025 forecast (tidak berubah)
                batas_mask = forecast_data['Batas_Atas'].notna() & forecast_data['Batas_Bawah'].notna()
                if batas_mask.any():
                    # Upper bound (invisible line)
                    fig.add_trace(go.Scatter(
                        x=forecast_data.loc[batas_mask, 'Tanggal'],
                        y=forecast_data.loc[batas_mask, 'Batas_Atas'],
                        mode='lines',
                        line=dict(color='rgba(128,128,128,0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    # Lower bound (visible line) with fill to the upper bound
                    fig.add_trace(go.Scatter(
                        x=forecast_data.loc[batas_mask, 'Tanggal'],
                        y=forecast_data.loc[batas_mask, 'Batas_Bawah'],
                        mode='lines',
                        line=dict(color='rgba(128,128,128,0)'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.2)',
                        name='Pita Keyakinan 95%',
                        hovertemplate="<b>Pita Keyakinan</b><br>Tanggal: %{x|%b %Y}<br>Batas Atas: Rp %{customdata:,.0f}<br>Batas Bawah: Rp %{y:,.0f}<extra></extra>",
                        customdata=forecast_data.loc[batas_mask, 'Batas_Atas']
                    ))

                # --- Latar belakang dan garis pemisah (tidak ada perubahan signifikan) ---
                fig.add_vrect(x0=forecast_data['Tanggal'].min(), x1=train_end, fillcolor="rgba(128,128,128,0.1)", layer="below", line_width=0, annotation_text="Training", annotation_position="top left")
                fig.add_vrect(x0=train_end, x1=test_end, fillcolor="rgba(0,0,255,0.1)", layer="below", line_width=0, annotation_text="Testing", annotation_position="top left")
                fig.add_vrect(x0=forecast_start, x1=forecast_data['Tanggal'].max(), fillcolor="rgba(255,165,0,0.1)", layer="below", line_width=0, annotation_text="Peramalan 2025", annotation_position="top left")
                fig.add_vline(x=train_end, line_dash="dot", line_color="red", line_width=2)
                fig.add_vline(x=forecast_start, line_dash="dot", line_color="red", line_width=2)
                
                # Update layout
                fig.update_layout(
                    title="Hasil Peramalan Model ARIMAX(2,1,2)-GARCH(1,1)",
                    xaxis_title="Waktu",
                    yaxis=dict(title="Harga Minyak Goreng (Rupiah)", range=[0, None]),
                    height=600,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Period explanations (tidak berubah)
                st.markdown("---")
                st.markdown("""
                **📋 Legenda:**
                - **Garis Hitam:** Data harga aktual.
                - **Garis Merah Putus-Putus:** Hasil peramalan model.
                - **Area Abu-abu Transparan:** Interval kepercayaan peramalan 95%.
                - **Area Latar Belakang (Abu-abu, Biru, Oranye):** Menandakan periode Training, Testing, dan Peramalan 2025.
                - **Garis Vertikal Merah Titik-Titik:** Garis pemisah antar periode.
                """)

                # Summary statistics
                st.subheader("📊 Ringkasan Hasil Peramalan")
                if forecast_2025_mask.any():
                    avg_forecast_2025 = forecast_data.loc[forecast_2025_mask, 'Peramalan'].mean()
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f'''
                        <div style="background-color: #f0f2f6; border-radius: 8px; padding: 1rem; text-align: center;">
                            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.2rem; line-height: 1;">Rata-rata Peramalan 2025</div>
                            <div style="font-size: 2.5rem; font-weight: bold; margin: 0.1rem 0; line-height: 0.9; color: #28a745;">Rp {avg_forecast_2025:,.0f}</div>
                            <div style="font-size: 1.1rem; color: #666; margin-top: 0.1rem; line-height: 1;">per liter per bulan</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'''
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; height: 100%;">
                            <p style="text-align: justify; line-height: 1.6; margin: 0;">
                            Berdasarkan hasil peramalan menggunakan model ARIMAX(2,1,2)-GARCH(1,1), 
                            prediksi harga minyak goreng sawit untuk tahun 2025 menunjukkan rata-rata sebesar 
                            <strong>Rp {avg_forecast_2025:,.0f} per liter per bulan</strong>. Model ini telah memperhitungkan 
                            pengaruh variabel eksogen dan volatilitas harga yang dinamis.
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)

            # --- PERUBAHAN 3: Logika Tabel Disederhanakan ---
            else:  # Tabel Data
                st.markdown("##### 📋 Tabel Data Hasil Peramalan")
                
                # Date filter for table
                col1, col2 = st.columns(2)
                with col1:
                    start_date_table = st.date_input(
                        "📅 Dari",
                        forecast_data['Tanggal'].min().date(),
                        min_value=forecast_data['Tanggal'].min().date(),
                        max_value=forecast_data['Tanggal'].max().date(),
                        key="table_start_tab5"
                    )
                with col2:
                    end_date_table = st.date_input(
                        "📅 Sampai",
                        forecast_data['Tanggal'].max().date(),
                        min_value=forecast_data['Tanggal'].min().date(),
                        max_value=forecast_data['Tanggal'].max().date(),
                        key="table_end_tab5"
                    )
                
                # Filter data based on date selection
                filtered_forecast_data = forecast_data[
                    (forecast_data['Tanggal'] >= pd.Timestamp(start_date_table)) & 
                    (forecast_data['Tanggal'] <= pd.Timestamp(end_date_table))
                ].copy()
                
                # Membuat tabel data untuk ditampilkan
                # Kolom 'Hasil Peramalan' sekarang langsung diambil dari 'Peramalan'
                display_data = pd.DataFrame({
                    'Periode': filtered_forecast_data['Tanggal'].dt.strftime('%b %Y'),
                    'Harga Aktual': filtered_forecast_data['Aktual'],
                    'Hasil Peramalan': filtered_forecast_data['Peramalan'],
                    'Batas Bawah (95%)': filtered_forecast_data['Batas_Bawah'],
                    'Batas Atas (95%)': filtered_forecast_data['Batas_Atas']
                })
                
                # Format angka untuk display
                for col in ['Harga Aktual', 'Hasil Peramalan', 'Batas Bawah (95%)', 'Batas Atas (95%)']:
                    display_data[col] = display_data[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                
                # Display table
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                
                # Download button
                csv = display_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Tabel CSV",
                    csv,
                    f"hasil_peramalan_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='download-csv'
                )
                
        else:
            st.error("❌ Data hasil peramalan tidak dapat dimuat atau kosong.")


    # Tab Simulasi Peramalan
    with tab6:
        st.header("🎯 Simulasi dengan Model Terbaik")
        st.markdown('<div class="info-panel"><h4>🔮 Lakukan Prediksi untuk Periode Berikutnya</h4><p>Masukkan nilai variabel eksogen sesuai dengan lag yang ditentukan untuk melakukan peramalan satu bulan ke depan.</p></div>', unsafe_allow_html=True)
        
        st.subheader("📊 Input Variabel Eksogen untuk Peramalan")
        with st.form(key='prediction_form'):
            col1, col2, col3 = st.columns(3)
            with col1:
                harga_cpo_input = st.number_input("Harga CPO Sebulan yang Lalu (USD/MT)", min_value=300.0, max_value=2500.0, value=900.0, step=10.0)
            with col2:
                produksi_cpo_input = st.number_input("Produksi CPO 4 Bulan yang Lalu (Ribu Ton)", min_value=500.0, max_value=8000.0, value=4000.0, step=200.0)
            with col3:
                igt_input = st.number_input("IGT Sebulan yang Lalu", min_value=0, max_value=100, value=20, step=1)
            
            submit_button = st.form_submit_button(label='🚀 Lakukan Peramalan', use_container_width=True)

        if submit_button:
            if data.empty or len(data) < 10:
                st.error("Data tidak cukup untuk melatih model. Diperlukan lebih banyak data historis.")
            else:
                with st.spinner("Sedang melatih model ARIMAX-GARCH dan melakukan peramalan..."):
                    try:
                        # Tetapkan periode peramalan menjadi 1
                        periode_peramalan = 1


                        # --- PERUBAHAN: Membagi data dengan skala yang ditentukan ---
                        df_model = data.copy().set_index('Bulan')
                        df_model['Harga_Kemasan'] = df_model['Harga_Kemasan'] / 10000
                        df_model['Harga_CPO'] = df_model['Harga_CPO'] / 1000
                        df_model['Produksi_CPO'] = df_model['Produksi_CPO'] / 1000000
                        df_model['IGT'] = df_model['IGT'] / 100

                        # 1. Persiapan data untuk pemodelan
                        df_model['Harga_CPO_lag1'] = df_model['Harga_CPO'].shift(1)
                        df_model['Produksi_CPO_lag4'] = df_model['Produksi_CPO'].shift(4)
                        df_model['IGT_lag1'] = df_model['IGT'].shift(1)
                        df_model = df_model.dropna()

                        y = df_model['Harga_Kemasan']
                        X = df_model[['Harga_CPO_lag1', 'Produksi_CPO_lag4', 'IGT_lag1']]

                        train_size = int(len(df_model) * 0.8)
                        y= y.iloc[:train_size]
                        X= X.iloc[:train_size]

                        # 2. Latih model ARIMAX(2,1,2) pada seluruh data
                        arimax_model = SARIMAX(endog=y, exog=X, order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
                        arimax_result = arimax_model.fit(disp=False)

                        # 3. Latih model GARCH(1,1) pada residual ARIMAX
                        residuals = arimax_result.resid[1:]
                        garch_model = arch_model(residuals, p=1, q=1, vol='GARCH', mean='Zero')
                        garch_result = garch_model.fit(disp="off")

                        # 4. Persiapan variabel eksogen untuk peramalan

                        # --- PERUBAHAN: Membagi input pengguna dengan skala yang ditentukan ---
                        scaled_harga_cpo = harga_cpo_input / 1000
                        scaled_produksi_cpo = produksi_cpo_input / 1000
                        scaled_igt = igt_input / 100


                        future_exog = pd.DataFrame({
                            'Harga_CPO_lag1': [scaled_harga_cpo],
                            'Produksi_CPO_lag4': [scaled_produksi_cpo],
                            'IGT_lag1': [scaled_igt]
                        })
                        
                        # 5. Lakukan peramalan
                        arimax_forecast = arimax_result.get_forecast(steps=periode_peramalan, exog=future_exog)
                        mean_forecast = arimax_forecast.predicted_mean

                        garch_forecast = garch_result.forecast(horizon=periode_peramalan, reindex=False)
                        variance_forecast = garch_forecast.variance.iloc[0]
                        std_dev_forecast = np.sqrt(variance_forecast)

                        # 6. Ekstrak nilai tunggal
                        predicted_price_scaled = mean_forecast.iloc[0]
                        std_dev = std_dev_forecast.iloc[0]
                        margin = 2 * std_dev
                        lower_bound_scaled = predicted_price_scaled - margin
                        upper_bound_scaled = predicted_price_scaled + margin

                        # --- PERUBAHAN: Mengembalikan hasil ke skala semula dengan dikali 10000 ---
                        predicted_price = predicted_price_scaled * 10000
                        lower_bound = lower_bound_scaled * 10000
                        upper_bound = upper_bound_scaled * 10000
                        
                        # 7. Tampilkan hasil dalam bentuk metrik
                        st.subheader(f"🔮 Hasil Peramalan untuk Periode Berikutnya")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prediksi Harga", f"Rp {predicted_price:,.0f}")
                        col2.metric("Batas Bawah (95%)", f"Rp {lower_bound:,.0f}")
                        col3.metric("Batas Atas (95%)", f"Rp {upper_bound:,.0f}")
                        
                        st.info(f"Berdasarkan model, harga minyak goreng untuk **periode berikutnya** diprediksi berada di angka **Rp {predicted_price:,.0f}**, dengan rentang kemungkinan antara **Rp {lower_bound:,.0f}** hingga **Rp {upper_bound:,.0f}** pada tingkat kepercayaan 95%.")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat melatih model atau melakukan peramalan: {e}")

if __name__ == "__main__":
    main()