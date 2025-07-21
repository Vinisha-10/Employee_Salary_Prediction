import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
salary_anim = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_isbiybfh.json')  # Updated URL
data_anim = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_isbiybfh.json')

# Load model
model = joblib.load("bestmodel.pkl")

# Load dataset and clean column names
df = pd.read_csv("jobs_in_data.csv")
df.columns = df.columns.str.strip()

# Helper to get unique values for form fields
def get_unique_values(column):
    return sorted(df[column].dropna().unique().tolist())

# Function to get currency conversion rate
def get_conversion_rate(from_currency, to_currency):
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url)
    data = response.json()
    return data['rates'].get(to_currency, 1)

# App layout
st.set_page_config(page_title="Salary Predictor", layout="centered", page_icon="💼")

# Sidebar for additional controls
with st.sidebar:
    st_lottie(data_anim, height=150, key="data")
    st.markdown("### Salary Prediction Controls")
    show_details = st.checkbox("Show detailed analysis", value=True)
    anim_speed = st.slider("Animation speed", 0.5, 2.0, 1.0)

# Main content
st.title("💼 Salary Predictor")
st.markdown("### Fill out the job details to get an estimated salary prediction")

# Animation header
st_lottie(salary_anim, height=200, key="salary", speed=anim_speed)

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", get_unique_values("job_title"), 
                            help="Select the job title for prediction")
    
    with st.expander("Experience Details"):
        experience = st.selectbox("Experience Level", get_unique_values("experience_level"))
        work_year = st.select_slider("Work Year", 
                                   options=sorted(df["work_year"].dropna().unique().astype(str)),
                                   value="2023")

with col2:
    with st.expander("Employment Details"):
        employment_type = st.selectbox("Employment Type", get_unique_values("employment_type"))
        company_size = st.selectbox("Company Size", get_unique_values("company_size"))
    
    with st.expander("Location Details"):
        company_location = st.selectbox("Company Location", get_unique_values("company_location"))
        employee_residence = st.selectbox("Employee Residence", get_unique_values("employee_residence"))

# Additional form elements
with st.expander("Advanced Options"):
    salary_currency = st.selectbox("Salary Currency", get_unique_values("salary_currency"))
    job_category = st.selectbox("Job Category", get_unique_values("job_category"))
    work_setting = st.radio("Work Setting", get_unique_values("work_setting"))

# Prediction button with animation
if st.button("🚀 Predict Salary", use_container_width=True):
    with st.spinner("Crunching numbers..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        # Build the input row
        input_data = pd.DataFrame([{
            "job_title": job_title,
            "experience_level": experience,
            "employment_type": employment_type,
            "company_size": company_size,
            "work_setting": work_setting,
            "company_location": company_location,
            "salary_currency": salary_currency,
            "job_category": job_category,
            "employee_residence": employee_residence,
            "work_year": int(work_year)
        }])

        try:
            prediction = model.predict(input_data)
            
            # Convert prediction to desired currency
            conversion_rate = get_conversion_rate("USD", salary_currency)
            converted_salary = prediction[0] * conversion_rate
            
            # Success animation
            st.balloons()
            
            # Display result with visualization
            st.success(f"💰 Estimated Salary: **{salary_currency} {converted_salary:,.2f}** per year")
            st.markdown("### Additional Information")
            st.markdown("This prediction is based on the current market trends and the data provided. Factors such as location, experience, and company size can significantly influence salary.")

            if show_details:
                # Comparison visualization
                st.markdown("### Salary Comparison")
                
                # Create sample comparison data
                comparison_df = pd.DataFrame({
                    'Category': ['Entry-Level', 'Mid-Level', 'Senior', 'Your Prediction'],
                    'Salary': [
                        df[df['experience_level'] == 'Entry']['salary_in_usd'].median(),
                        df[df['experience_level'] == 'Mid']['salary_in_usd'].median(),
                        df[df['experience_level'] == 'Senior']['salary_in_usd'].median(),
                        converted_salary
                    ]
                })
                
                fig = px.bar(comparison_df, x='Category', y='Salary', 
                            title="How Your Salary Compares",
                            color='Category',
                            text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Footer with additional info
st.markdown("---")
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
<p>Salary Prediction Model • Updated 2023 • <a href="#" style="color:white;">About</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
