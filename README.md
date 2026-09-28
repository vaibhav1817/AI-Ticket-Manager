# AI Support Ticket Manager & Auto-Routing System

An end-to-end Machine Learning intelligence and auto-routing system for customer support tickets, featuring NLP text preprocessing, TF-IDF feature extraction, multiclass category classification, separate urgency prediction, F1/precision/recall evaluation, a human-in-the-loop low-confidence review queue, and a web dashboard + REST API.

---

## 📋 Requirement Compliance Checklist

| # | Requirement | Implementation Details | Status |
|---|---|---|:---:|
| **1** | **Collect historical ticket text, category, priority, and resolution data** | Loaded from `ticket.zip` / `customer_support_tickets.csv` (8,469 records) containing subject, description, category (`Ticket Type`), urgency (`Ticket Priority`), and resolution data. | ✅ **100% Complete** |
| **2** | **Clean text and create labels (billing, technical, account, product, etc.)** | Implemented regex-based NLP cleaning in `clean_text()` (lowercasing, bracket stripping, punctuation normalization, whitespace cleaning). Target classes: `Billing inquiry`, `Technical issue`, `Cancellation request`, `Product inquiry`, `Refund request`. | ✅ **100% Complete** |
| **3** | **Create text features using TF-IDF or sentence embeddings** | `TfidfVectorizer` configured with 10,000 maximum features, n-gram range `(1, 2)`, sublinear TF scaling, and stop-word filtering. | ✅ **100% Complete** |
| **4** | **Train a multiclass classifier and a separate urgency model** | 1. **Category Model**: Multiclass balanced `LogisticRegression` for ticket category.<br>2. **Urgency Model**: Separate multiclass balanced `LogisticRegression` for urgency levels (`Critical`, `High`, `Medium`, `Low`). | ✅ **100% Complete** |
| **5** | **Evaluate with F1-score, confusion matrix, and precision/recall by class** | Both models evaluated on an unseen 20% stratified test set. Accuracy, Macro F1, Weighted F1, full `classification_report` (precision, recall, f1 per class), and full confusion matrices are generated and saved to `models/metrics.json`. | ✅ **100% Complete** |
| **6** | **Expose the model through a small API/dashboard and log low-confidence predictions for human review** | 1. **REST API** (`/api/predict`, `/api/reviews`, `/api/metrics`, `/api/stats`).<br>2. **Web Dashboard** with real-time ticket playground, candidate probabilities, and review queue.<br>3. **Low-Confidence Logging**: Any prediction below confidence threshold is automatically logged to `data/low_confidence_reviews.csv` for human review. | ✅ **100% Complete** |

---

## 🏗️ Project Architecture

```
internship projects/
│
├── AI-TICKET-MANAGER            # Full EDA, Data Quality & Model Training script
├── model_pipeline.py            # NLP cleaning, TF-IDF vectorizer & dual-model pipeline
├── review_logger.py             # Human review queue manager & CSV logging service
├── app.py                       # Flask REST API & Modern Web Dashboard
│
├── models/                      # Serialized ML Artifacts
│   ├── tfidf_vectorizer.joblib  # Trained TF-IDF vectorizer (10k features)
│   ├── category_model.joblib    # Trained Category multiclass classifier
│   ├── urgency_model.joblib     # Trained Urgency multiclass classifier
│   └── metrics.json             # F1, precision, recall & confusion matrices
│
├── data/                        # Live Operations & Review Queue Data
│   ├── low_confidence_reviews.csv # Audit log of flagged tickets for human review
│   └── reviews.json             # JSON review queue storage
│
├── ticket.zip                   # Raw source dataset (8,469 records)
└── customer_support_cleaned.csv # Processed dataset with engineered features
```

---

## 🚀 Running the Project

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard & REST API
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser:
- **Tab 1: Live Classifier & Router**: Paste any ticket text, test sample tickets, adjust confidence threshold, and see instant auto-routing decisions with probability breakdowns.
- **Tab 2: Human Review Queue**: Inspect low-confidence tickets, update predicted categories/urgencies, and resolve them.
- **Tab 3: Model Evaluation**: View live F1-scores, precision, recall, and interactive confusion matrices.

### 3. Run the Batch EDA & Training Script
```powershell
python AI-TICKET-MANAGER
```

---

## 📡 REST API Reference

### 1. Predict & Auto-Route Ticket
- **URL**: `POST /api/predict`
- **Body**:
  ```json
  {
    "subject": "Double charged on monthly bill",
    "text": "I was billed twice on September 1st. Please issue a refund.",
    "threshold": 0.50
  }
  ```
- **Response**:
  ```json
  {
    "category": {
      "prediction": "Billing inquiry",
      "confidence": 0.78,
      "all_probabilities": { ... }
    },
    "urgency": {
      "prediction": "High",
      "confidence": 0.65,
      "all_probabilities": { ... }
    },
    "overall_confidence": 0.65,
    "needs_human_review": false,
    "review_id": "TCK-A1B2C3D4",
    "status": "AUTO_ROUTED"
  }
  ```

### 2. Fetch Review Queue
- **URL**: `GET /api/reviews?filter=pending`

### 3. Resolve a Review Ticket
- **URL**: `POST /api/reviews/<review_id>/resolve`
- **Body**:
  ```json
  {
    "final_category": "Billing inquiry",
    "final_urgency": "High",
    "notes": "Confirmed double charge, routed to billing specialist."
  }
  ```

### 4. Fetch Model Metrics
- **URL**: `GET /api/metrics`
