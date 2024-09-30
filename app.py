import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('penguin_model.pkl')

# Judul aplikasi
st.title("Penguins Species Classification")

# Input dari pengguna
st.write("Masukkan fitur penguins:")

# culmen_length_mm	culmen_depth_mm	flipper_length_mm	body_mass_g	sex

culmenlength = st.slider('Culmen Length', 32.1, 59.6)
culmendepth = st.slider('Culmen depth', 13.1, 21.5)
flipperlength = st.slider('flipper length', 172.0, 231.0)
bodymass = st.slider('body mass', 2700.0, 6300.0)
sex = st.slider('sex', 0, 1)


# Buat prediksi jika tombol ditekan
if st.button('Predict'):
    input_data = np.array([[culmenlength, culmendepth, flipperlength, bodymass, sex]])
    prediction = model.predict(input_data)
    
    # Map hasil prediksi ke nama spesies
    species = ['Adelie', 'Chinstrap', 'Gentoo']
    st.write(f"Predicted penguins Species: {species[prediction[0]]}")

# Menjalankan aplikasi dengan: streamlit run app.py