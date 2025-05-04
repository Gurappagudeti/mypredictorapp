import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------
# ✅ Streamlit page setup
# --------------------------------------------
st.set_page_config(page_title="AutoML Prediction App", layout="centered")


# --------------------------------------------
# 🔐 Simple password protection
# --------------------------------------------
PASSWORD = "abc123"  # 👉 Change this to your desired password

def password_protection():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        with st.form("login_form"):
            st.markdown("### 🔐 Please enter the password to access the app")
            password = st.text_input("Password", type="password", placeholder="")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if password == PASSWORD:
                    st.session_state.password_correct = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password, try again.")

        return False
    return True

if not password_protection():
    st.stop()  # Prevent access until password is correct

# --------------------------------------------
# ✅ Load model and scaler (cached)
# --------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("cleaned_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# --------------------------------------------
# 📊 App UI and logic
# --------------------------------------------
st.title("📊 AutoML Equipment Condition Predictor")
st.markdown("Upload an Excel file to predict `% Condition`.")

uploaded_file = st.file_uploader("📁 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Map categorical fields
        df['Purchase Condition'] = df['Purchase Condition'].map({'Brand New': 1, 'Second Hand': 2})
        df['Keep or Sell'] = df['Keep or Sell'].map({'Keep': 1, 'Sell': 2})

        # Prepare features
        X_test = df.drop(columns=[col for col in ['% Condition', 'Plant Number'] if col in df.columns])
        X_scaled = scaler.transform(X_test)

        # Predict
        df['Predicted % Condition'] = model.predict(X_scaled)

        # ✅ Show metrics if actual `% Condition` values exist
        if '% Condition' in df.columns:
            r2 = r2_score(df['% Condition'], df['Predicted % Condition'])
            mse = mean_squared_error(df['% Condition'], df['Predicted % Condition'])

            st.subheader("📈 Model Performance Metrics")
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("Mean Squared Error", f"{mse:.2f}")
        else:
            st.warning("⚠️ No actual `% Condition` values found to evaluate model accuracy.")

        # Output
        st.success("✅ Predictions generated successfully.")
        st.dataframe(df)

        # Excel export
        output = BytesIO()
        df.to_excel(output, index=False)
        st.download_button("📥 Download Result", output.getvalue(), file_name="predictions.xlsx")

    except Exception as e:
        st.error(f"❌ Error: {e}")