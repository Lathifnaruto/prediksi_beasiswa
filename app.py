import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Beasiswa BAZNAS", layout="wide")

# Memuat dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('beasiswa.csv')
        data['Diterima'] = ((data['IPK'] >= 3.0) & 
                          (data['Pendapatan_Keluarga'] <= 3000000) & 
                          (data['Jumlah_Tanggungan'] >= 3) & 
                          (data['Skor_Motivasi'] >= 70)).astype(int)
        return data
    except Exception as e:
        st.error(f"Error memuat data: {str(e)}")
        return None

data = load_data()
if data is None:
    st.stop()

# Sidebar untuk informasi dataset
with st.sidebar:
    st.header("Informasi Dataset")
    st.write(f"Total Data: {len(data)} records")
    st.write("Kriteria Diterima:")
    st.write("- IPK ≥ 3.0")
    st.write("- Pendapatan ≤ 3 juta")
    st.write("- Tanggungan ≥ 3")
    st.write("- Skor Motivasi ≥ 70")
    
    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(data)

# Pelatihan model
@st.cache_resource
def train_model(data):
    features = ['Pendapatan_Keluarga', 'Jumlah_Tanggungan', 'IPK', 
               'Skor_Motivasi', 'Status_Orang_Tua', 'Aktivitas_Organisasi']
    X = pd.get_dummies(data[features])
    y = data['Diterima']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X, X_test, y_test

model, X, X_test, y_test = train_model(data)

# Fungsi untuk breakdown alasan
def generate_decision_reason(input_data):
    reasons = []
    
    # Kriteria dasar
    kriteria = {
        'IPK ≥ 3.0': input_data['IPK'][0] >= 3.0,
        'Pendapatan ≤ 3 juta': input_data['Pendapatan_Keluarga'][0] <= 3000000,
        'Jumlah Tanggungan ≥ 3': input_data['Jumlah_Tanggungan'][0] >= 3,
        'Skor Motivasi ≥ 70': input_data['Skor_Motivasi'][0] >= 70
    }
    
    # Analisis kriteria
    for k, v in kriteria.items():
        reasons.append(f"{'✓' if v else '✗'} {k}")
    
    # Analisis feature importance
    feature_importance = model.feature_importances_
    features = X.columns
    top_features = sorted(zip(features, feature_importance), 
                        key=lambda x: x[1], reverse=True)[:3]
    
    reasons.append("\nFaktor paling berpengaruh:")
    for feat, imp in top_features:
        value = input_data[feat][0] if feat in input_data.columns else "N/A"
        reasons.append(f"- {feat} (nilai Anda: {value})")
    
    return "\n".join(reasons)

# Antarmuka utama
st.title("🎓 Prediksi Beasiswa BAZNAS Indonesia")
st.write("Isi formulir berikut untuk memeriksa kelayakan beasiswa:")

col1, col2 = st.columns(2)

with col1:
    pendapatan = st.number_input("Pendapatan Keluarga (Rp)", 
                               min_value=0, 
                               step=100000,
                               value=2500000)
    tanggungan = st.number_input("Jumlah Tanggungan Keluarga", 
                               min_value=0, 
                               step=1,
                               value=3)
    ipk = st.number_input("IPK", 
                         min_value=0.0, 
                         max_value=4.0, 
                         step=0.01,
                         value=3.2)

with col2:
    motivasi = st.slider("Skor Motivasi", 
                       0, 100, 75)
    status_orang_tua = st.selectbox("Status Orang Tua", 
                                  data['Status_Orang_Tua'].unique())
    organisasi = st.selectbox("Aktivitas Organisasi", 
                            data['Aktivitas_Organisasi'].unique())

if st.button("🚀 Periksa Kelayakan", use_container_width=True):
    with st.spinner("Menganalisis data..."):
        try:
            # Persiapkan input data
            input_data = pd.DataFrame({
                'Pendapatan_Keluarga': [pendapatan],
                'Jumlah_Tanggungan': [tanggungan],
                'IPK': [ipk],
                'Skor_Motivasi': [motivasi],
                'Status_Orang_Tua': [status_orang_tua],
                'Aktivitas_Organisasi': [organisasi]
            })
            
            # Encoding konsisten dengan training
            input_encoded = pd.get_dummies(input_data)
            missing_cols = set(X.columns) - set(input_encoded.columns)
            for col in missing_cols:
                input_encoded[col] = 0
            input_encoded = input_encoded[X.columns]
            
            # Prediksi
            prediction = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0][1]
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi")
            if prediction == 1:
                st.success("## DITERIMA ✅")
                st.balloons()
            else:
                st.error("## TIDAK DITERIMA ❌")
            
            st.metric("Probabilitas Diterima", f"{proba*100:.1f}%")
            
            # Breakdown alasan
            st.subheader("Analisis Keputusan")
            st.write(generate_decision_reason(input_data))
            
            # Visualisasi kriteria
            st.subheader("Pencapaian Kriteria")
            kriteria = {
                'IPK ≥ 3.0': ipk >= 3.0,
                'Pendapatan ≤ 3jt': pendapatan <= 3000000,
                'Tanggungan ≥ 3': tanggungan >= 3,
                'Motivasi ≥ 70': motivasi >= 70
            }
            
            cols = st.columns(4)
            for i, (k, v) in enumerate(kriteria.items()):
                cols[i].metric(k, "✓" if v else "✗")
            
            # Grafik probabilitas
            fig, ax = plt.subplots()
            ax.bar(['Tidak Diterima', 'Diterima'], 
                  [1-proba, proba], 
                  color=['red', 'green'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")

# Evaluasi model di expander
with st.expander("📊 Evaluasi Model"):
    st.write(f"Akurasi model: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    
    # Feature importance
    st.subheader("Faktor Penentu Keputusan")
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feat_importances.nlargest(10))

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi Prediksi Beasiswa BAZNAS - © 2023")