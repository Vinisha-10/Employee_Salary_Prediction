# Salary Predictor

A Streamlit web application that predicts data professional salaries based on various job attributes and provides insightful data visualizations.

## Table of Contents

* [About the Project](#about-the-project)
    * [Features](#features)
    * [Built With](#built-with)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [Model Training & Evaluation](#model-training-evaluation)
    * [Data Preprocessing](#data-preprocessing)
    * [Models Evaluated](#models-evaluated)
    * [Performance Metrics](#performance-metrics)
    * [Best Model](#best-model)
* [Data Insights](#data-insights)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgments](#acknowledgments)

## About The Project

This project provides a user-friendly web interface for predicting salaries in the data industry. It leverages a machine learning model trained on a comprehensive dataset of jobs in data. The application allows users to input various job characteristics and receive an estimated salary, along with visualizations to understand how their predicted salary compares to market trends.

The core of the project involves a machine learning pipeline that handles data preprocessing, model training, and prediction. The accompanying Jupyter Notebook (`model training.ipynb`) details the entire process from data loading and exploration to model selection and evaluation.

### Features

* **Interactive Salary Prediction:** Input job title, experience level, employment type, company size, work setting, company and employee location, work year, and desired salary currency to get an estimated annual salary.
* **Currency Conversion:** Predicted salaries can be converted from USD to various other currencies using real-time exchange rates.
* **Salary Comparison Visualizations:** See how your predicted salary compares to median salaries for different experience levels.
* **Detailed Data Insights:** Explore various visualizations generated during model training, including:
    * Distribution of Salary in USD
    * Salary by Experience Level (Box Plot)
    * Salary by Top 10 Job Categories (Box Plot)
    * Salary by Company Size (Box Plot)
    * Average Salary over Work Year (Bar Plot)
* **Responsive User Interface:** Built with Streamlit for a clean and interactive experience.

### Built With

* **Python 3.x**
* **Streamlit:** For creating the interactive web application.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For building and evaluating machine learning models (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor).
* **Joblib:** For efficient saving and loading of the trained machine learning model.
* **Plotly Express:** For interactive and aesthetically pleasing data visualizations within the Streamlit app.
* **Matplotlib:** For generating static data insight and model comparison plots during the training phase.
* **Requests:** For fetching external API data (currency exchange rates).
* **Streamlit Lottie:** For engaging Lottie animations in the UI.

## Getting Started

To get a local copy of this project up and running, follow these steps.

### Prerequisites

Ensure you have Python 3.x installed on your system. You will also need `pip` for package installation.

```sh
# Check Python version
python --version
```

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/salary-predictor.git](https://github.com/your_username/salary-predictor.git)
    ```
2.  **Navigate into the project directory:**
    ```sh
    cd salary-predictor
    ```
3.  **Install the required Python packages:**
    It's recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
    Then install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file would typically be generated, but based on the provided notebooks, you'd need to install `streamlit`, `pandas`, `joblib`, `plotly-express`, `scikit-learn`, `matplotlib`, `streamlit-lottie`, and `requests`.)*

## Usage

To run the Streamlit web application:

1.  **Ensure you are in the project directory** and your virtual environment is activated.
2.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

## Model Training & Evaluation

The `model training.ipynb` Jupyter Notebook contains the complete workflow for training and evaluating the machine learning models.

### Data Preprocessing

* The `jobs_in_data.csv` dataset is loaded.
* Column names are cleaned (stripped of whitespace).
* Categorical features (`job_title`, `job_category`, `salary_currency`, `employee_residence`, `experience_level`, `employment_type`, `work_setting`, `company_location`, `company_size`) are identified for one-hot encoding.
* Numerical features (`work_year`) are identified for passthrough.
* A `ColumnTransformer` with `OneHotEncoder` (handling unknown categories) is used for preprocessing.
* The data is split into 80% training and 20% testing sets.

### Models Evaluated

The following regression models were trained and evaluated:

* **Linear Regression**
* **Random Forest Regressor**
* **Gradient Boosting Regressor**

### Performance Metrics

Each model was evaluated using:

* **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors.
* **Root Mean Squared Error (RMSE)**: Measures the square root of the average of the squared errors.
* **R-squared ($R^2$)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

| Model                       | MAE     | RMSE    | R-squared |
|:----------------------------|:--------|:--------|:----------|
| Linear Regression           | 38951.4 | 51346.2 | 0.36      |
| Random Forest Regressor     | 38693.5 | 51136.6 | 0.37      |
| Gradient Boosting Regressor | 38981.8 | 51306.6 | 0.37      |

### Best Model

Based on the R-squared score (higher is better), the **Random Forest Regressor** was selected as the best model for salary prediction. This model is saved as `bestmodel.pkl` (or `salary_prediction_random_forest_model.joblib`) and loaded by the `app.py` for predictions.

## Data Insights

The `model training.ipynb` notebook also generates several plots to provide insights into the `jobs_in_data.csv` dataset:

* **Distribution of Salary in USD:** A histogram showing the frequency of different salary ranges.
* **Salary in USD by Experience Level:** Box plots illustrating salary distributions across various experience levels.
* **Salary in USD by Top 10 Job Categories:** Box plots comparing salaries for the most frequent job categories.
* **Salary in USD by Company Size:** Box plots showing salary variations based on company size (S, M, L).
* **Average Salary in USD Over Work Year:** A bar plot depicting the trend of average salaries over different work years.

These plots are saved as `.png` files during the notebook execution.

## Roadmap

* [ ] Implement advanced feature engineering techniques.
* [ ] Explore deep learning models for improved accuracy.
* [ ] Add more interactive visualizations to the Streamlit app.
* [ ] Allow users to upload their own datasets for custom model training (if feasible).
* [ ] Improve UI/UX based on user feedback.

See the [open issues](https://github.com/your_username/salary-predictor/issues) for a full list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## Contact

Project Link: [https://github.com/your_username/salary-predictor](https://github.com/your_username/salary-predictor)

## Acknowledgments

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
* [LottieFiles](https://lottiefiles.com/)
