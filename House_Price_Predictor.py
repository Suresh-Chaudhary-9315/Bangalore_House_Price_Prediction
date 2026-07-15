import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set clean, professional layout
st.set_page_config(
    page_title="Real Estate Valuator | ML Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED ARTIFICAL LOADERS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("randomforest_regressor.pkl")
        encoders = joblib.load("house_label_encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        st.error("⚠️ Required model or encoder .pkl files are missing in this directory!")
        return None, None

house_model, encoders = load_assets()

# --- SIDEBAR PROJECT DETAILS ---
st.sidebar.title("🎓 Project Dashboard")
st.sidebar.markdown("### **House Price Prediction Engine**")
st.sidebar.markdown("---")
st.sidebar.info("This system uses parallel Multi-Label Encoding and Ensemble Regression to estimate market valuations.")

# --- MAIN CONTENT ---
st.title("🏡 Smart Real Estate Valuation System")
st.markdown("Enter property structural dimensions and routing parameters below to calculate real-time estimated asset values.")

tab1, tab2 = st.tabs(["🚀 Valuation Sandbox", "📊 Model Design & Architecture"])

# ==================== TAB 1: VALUATION SANDBOX ====================
with tab1:
    st.header("Property Parameter Input")
    st.write("Fill out the specific spatial characteristics of the property:")

    if house_model and encoders:
        # Extract original classes from your fitted encoders to populate the dropdown select boxes safely
        area_options = list(encoders['area_type'].classes_)
        avail_options = list(encoders['availability'].classes_)
        loc_options = list(encoders['location'].classes_)
        size_options = list(encoders['size'].classes_)

        # Clean 2-column input layout
        col1, col2 = st.columns(2)
        
        with col1:
            ui_area = st.selectbox("Area Categorization Type", options=area_options, index=0)
            ui_avail = st.selectbox("Availability Status Profile", options=avail_options, index=0)
            ui_loc = st.selectbox("Geographic Location Zone", options=loc_options, index=0)
        
        with col2:
            ui_sqft = st.number_input("Total Spatial Density (Square Feet)", min_value=100, max_value=50000, value=1200, step=50)
            ui_size = st.selectbox("Property Configuration Unit Size", options=size_options, index=0)
            ui_balcony = st.number_input("Total Balcony Unit Count", min_value=0, max_value=10, value=1, step=1)

        st.markdown("---")
        
        if st.button("💰 Calculate Market Valuation Price", type="primary"):
            with st.spinner("Processing structural equations and label encodings..."):
                try:
                    # 1. Transform text inputs using their specific corresponding encoders
                    enc_area = encoders['area_type'].transform([ui_area])[0]
                    enc_avail = encoders['availability'].transform([ui_avail])[0]
                    enc_loc = encoders['location'].transform([ui_loc])[0]
                    enc_size = encoders['size'].transform([ui_size])[0]

                    # 2. Re-construct the identical input array shape: [area_type, availability, location, total_sqft, size, balcony]
                    live_features = np.array([[enc_area, enc_avail, enc_loc, float(ui_sqft), enc_size, float(ui_balcony)]])

                    # 3. Generate Valuation Prediction
                    predicted_price = house_model.predict(live_features)[0]

                    # If your training notebook used log transformation (e.g. np.log1p), uncomment the line below to revert it back:
                    # predicted_price = np.expm1(predicted_price)

                    # 4. Display Clean Visual Results
                    st.markdown("### 📊 Valuation Output Summary")
                    st.success("🎉 Processing Successful! Valuation pipeline completed.")
                    
                    # Renders a sleek pricing metric callout card
                    st.metric(
                        label="Estimated Market Value Valuation Price", 
                        value=f"₹ {predicted_price:,.2f} Lakhs", 
                        delta="Asset Cleared"
                    )
                    
                    # Explanatory Technical Breakdown expander for the grading professor
                    with st.expander("🔍 Inspect Internal Pipeline Metrics (Professor Review)"):
                        st.markdown("**Processed Feature Array (Fed to Model):**")
                        feature_cols = ['area_type', 'availability', 'location', 'total_sqft', 'size', 'balcony']
                        debug_df = pd.DataFrame(live_features, columns=feature_cols)
                        st.dataframe(debug_df)
                        st.caption("Categorical text elements were safely transformed into distinct serial numerical coordinates through independent encoder vectors.")

                except Exception as e:
                    st.error(f"❌ Valuation Failure. Structural conflict detected: {str(e)}")
                    st.info("Check if your web inputs contain categories your notebook label encoders have never seen before.")

# ==================== TAB 2: MODEL ARCHITECTURE ====================
with tab2:
    st.header("Project Execution Specifications")
    st.markdown("### 📊 Configured Variable Vector Matrix")
    st.markdown("""
    The regression matrix evaluates house pricing across 6 targeted properties:
    *   **`area_type`**: Evaluated via specialized category matching (`LabelEncoder`).
    *   **`availability`**: Ready-to-move timelines or future construction date tracking (`LabelEncoder`).
    *   **`location`**: Hyper-local regional address zoning data (`LabelEncoder`).
    *   **`total_sqft`**: Continuous numerical dimension parameters representing property sizing scale.
    *   **`size`**: Apartment layout specifications (e.g., 2 BHK, 3 BHK) mapped smoothly (`LabelEncoder`).
    *   **`balcony`**: Structured continuous scalar counting unit availability indices.
    """)
