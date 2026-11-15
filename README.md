markdown
# Used Motorcycle Selling Price Prediction

## 1. Problem Description

The objective of this project is to model and predict the selling price of various brands of used motorcycles using machine learning algorithms and deploy the solution as a web service utilizing **FastAPI**, **Docker**, and **Fly.io**.

By employing a regression model, this service forecasts the resale value of different motorcycle brands based on key vehicle attributes.

* **Data Source:** The motorcycle dataset is available on [OpenML (Listing ID: 43801)](https://openml.org) and was downloaded using the `fetch_openml` function from `sklearn.datasets`.
* **Dataset Shape:** 1,061 rows and 7 columns.
* **Features:** `name`, `year`, `seller_type`, `owner`, `km_driven`, and `ex_showroom_price`.
* **Target Variable:** `selling_price`.

---

## 2. Dataset & Exploratory Data Analysis (EDA)

Key findings and preprocessing steps from the exploratory data analysis include:

* **Unique Values:** `selling_price` (130), `seller_type` (2: individual or commercial), `owner` (4: 1st, 2nd, 3rd, or 4th owner), `km_driven` (304), `ex_showroom_price` (231), and `age` (28).
* **Missing Values:** Found 435 missing values exclusively in `ex_showroom_price`, which were imputed using the mean price grouped by motorcycle model.
* **Statistical Summary:** Conducted numerical summary statistics using `df.describe()`.
* **Feature Engineering:** Created a new `age` feature by subtracting the motorcycle's manufacturing `year` from the current year.
* **Feature Selection:** Dropped the `name` column as it was unnecessary for modeling after imputation.
* **Correlation Analysis:** Numerical correlation analysis revealed the strongest positive relationship between `selling_price` and `ex_showroom_price` (r = 0.75).
* **Target Normalization:** The histogram of `selling_price` showed right-skewness with outliers, so a logarithmic transformation (`np.log`) was applied to normalize the distribution for modeling.

---

## 3. Modeling & Training

### Model Selection & Tuning
The dataset was split using `train_test_split` from `sklearn.model_selection`:
1. Initial split into `df_full_train` (80%) and `df_test` (20%).
2. Secondary split of `df_full_train` into `df_train` (60% of total) and `df_val` (20% of total / 25% of full train).

Features were converted into dictionaries and vectorized into a feature matrix using `DictVectorizer`. Three regression algorithms were trained and tuned using `GridSearchCV` (5-fold cross-validation) evaluated via Root Mean Squared Error (**RMSE**):

| Model | Best Hyperparameters | Best CV RMSE | Test Set RMSE |
| :--- | :--- | :---: | :---: |
| **Decision Tree Regressor** | `max_depth=5`, `max_features=None`, `min_samples_leaf=1`, `min_samples_split=2` | 37,858.58 | — |
| **Random Forest Regressor** | `max_depth=40`, `min_samples_leaf=2`, `n_estimators=30` | 34,874.73 | — |
| **XGBoost Regressor** *(Selected)* | `subsample=0.7`, `reg_lambda=1.0`, `reg_alpha=0`, `n_estimators=140`, `min_child_weight=1`, `max_depth=10`, `learning_rate=0.1`, `colsample_bytree=0.9` | **34,265.00** | **28,385.11** |

**Conclusion:** **XGBoost** achieved the lowest RMSE on cross-validation and strong generalization on the test set (`28,385.11`), making it the selected model for production.

### Development Workflow
* **`training.ipynb` & `train.py`:** Modularized data loading (`load_data()`), pipeline training (`DictVectorizer` + `XGBRegressor`), and model serialization (`save_model()` using `pickle.dump()` to output `model.bin`). Converted to `train.py` via `jupyter nbconvert --to script train.ipynb`.
* **`predict.ipynb` & `predict.py`:** Loaded `model.bin` with `pickle.load()`, defined a Pydantic input schema (`Motorcycle_Features`), initialized a FastAPI app, and exposed a `POST /predict` endpoint (`predict_selling_price()`). Converted to `predict.py` via `jupyter nbconvert --to script predict.ipynb`.

---

## 4. Repository Structure

```text
├── notebook.ipynb        # Initial EDA, feature engineering, and model experimentation
├── training.ipynb        # Modularized training notebook
├── train.py              # Production training script that generates model.bin
├── predict.ipynb         # API service development notebook
├── predict.py            # FastAPI application serving the /predict endpoint
├── model.bin             # Serialized DictVectorizer and trained XGBoost pipeline
├── requirements.txt      # Python dependencies with pinned versions
├── Dockerfile            # Container configuration for packaging the FastAPI service
└── README.md             # Project documentation
```

---

## 5. Reproducibility & Environment Setup

### Dependencies
The project uses the following core packages (listed in `requirements.txt`):

| Package | Version | Role |
| :--- | :--- | :--- |
| `fastapi` | `0.162.5` | Builds the `/predict` REST API web service and Swagger UI (`/docs`). |
| `uvicorn` | `0.54.0` | ASGI web server that runs the FastAPI application. |
| `pydantic` | `2.14.3` | Validates incoming JSON payloads against the `Motorcycle_Features` schema. |
| `scikit-learn` | `1.9.1` | Provides `DictVectorizer` and pipeline utilities saved inside `model.bin`. |
| `xgboost` | Latest | Provides the `XGBRegressor` model saved inside `model.bin`. |
| `pandas` | Latest | Handles dataframe manipulation and preprocessing. |

### Local Setup Commands

#### 1. Clone the Repository
```bash
git clone https://github.com<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

#### 2. Create and Activate a Virtual Environment
**On Linux / macOS / GitHub Codespaces:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Train the Model
Run the training script to preprocess the dataset, train the XGBoost pipeline, and generate `model.bin`:
```bash
python train.py
```

#### 5. Start the FastAPI Server
Launch the local prediction service on port `9696`:
```bash
python predict.py
```
Once running, open `http://localhost:9696/docs` in your browser to test live predictions via the interactive Swagger UI.

---

## 6. Deployment & Containerization

This project includes a `Dockerfile` to package the FastAPI prediction service and trained XGBoost model (`model.bin`) into a standalone container.

### Dockerfile
```dockerfile
# Paste your Dockerfile contents here
```

### 1. Run Locally with Docker

**Build the Docker image:**
```bash
docker build -t motorcycle-price-predictor .
```

**Run the container:**
```bash
docker run -it --rm -p 9696:9696 motorcycle-price-predictor
```
Once started, the API is accessible at `http://localhost:9696` and the Swagger UI documentation is available at `http://localhost:9696/docs`.

### 2. Test the Live API Endpoint

Send a sample JSON `POST` request using `curl` to verify predictions:

```bash
curl -X POST "http://localhost:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "seller_type": "Individual",
    "owner": "1st owner",
    "km_driven": 25000,
    "ex_showroom_price": 85000.0,
    "age": 4
  }'
```

**Example JSON Response:**
```json
{
  "predicted_selling_price": 62450.75
}
```

### 3. Deploy to Cloud (Fly.io)

Because the repository contains a `Dockerfile`, it can be deployed directly to Fly.io:

```bash
# Install the Fly.io CLI (Linux / macOS)
curl -L https://fly.io | sh

# Authenticate your account
fly auth login

# Launch and deploy the containerized service
fly launch
```
*(Note: Ensure `internal_port = 9696` is configured inside the generated `fly.toml` file before deploying.)*
Use code with caution.










```
