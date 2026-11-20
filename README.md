# Customer Value & Repeat-Purchase Prediction for E-Commerce (Olist)
# DEPI-E-Commerce-Customer-Value-Prediction
DEPI Data Science Graduation Project: E-Commerce Customer Value Analysis and Purchase Prediction using Machine Learning.

> Data Scientist Track – Graduation Project (Project 7)
> Team: _[AAMMO]_ · Members:[Assem Abdelgawad ,Ahmed Mostafa ,Mohamed Essam ,Menna Mohamed ,Omar Bastawisy]

---

## 1. Problem Statement

An online marketplace processes tens of thousands of orders, but management has no clear view of customer behavior. It is unclear which customers and product categories drive revenue, how delivery experience affects satisfaction, and which first-time buyers are likely to return. As a result, marketing and logistics spending is not targeted. This project builds a SQL-backed analytics system with a machine-learning model that predicts whether a customer will place a repeat order, and an interactive dashboard that supports these decisions.

## 2. Project Objectives

1. Build a relational database (SQL) from the Olist data and write queries that compute the business KPIs.
2. Clean the data and engineer customer-level features (order value, delivery delay, review score, payment behavior, region, category).
3. Train and compare at least three ML models to predict repeat purchase, and handle class imbalance.
4. Deliver an interactive dashboard showing revenue, customers, products, customer segments, and model predictions.
5. Document, version (GitHub), and deploy the work so it can be reproduced.

## 3. Business Questions

1. How do revenue and number of orders change over time (monthly / seasonally)?
2. Which product categories and sellers generate the most revenue?
3. What share of customers are repeat buyers, and how do they differ from one-time buyers?
4. How does delivery delay (actual vs. estimated date) affect review scores?
5. Which regions (states/cities) have the most customers and the highest average order value?
6. Which payment types and installment counts are most common, and do they relate to order value?
7. Which customers are likely to buy again, and what factors drive that prediction?

## 4. Chosen Dataset

- **Name:** Brazilian E-Commerce Public Dataset by Olist
- **Link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Period:** 2016 – 2018 · **Files:** 9 CSV files
- **Size (verify with `df.shape` and update):**

| File | Rows (approx.) | Columns |
|---|---|---|
| olist_customers_dataset | 99,441 | 5 |
| olist_orders_dataset | 99,441 | 8 |
| olist_order_items_dataset | 112,650 | 7 |
| olist_order_payments_dataset | 103,886 | 5 |
| olist_order_reviews_dataset | 99,224 | 7 |
| olist_products_dataset | 32,951 | 9 |
| olist_sellers_dataset | 3,095 | 4 |
| olist_geolocation_dataset | ~1,000,000 | 5 |
| product_category_name_translation | 71 | 2 |

> **Important:** `customer_id` is unique per order. The real customer identifier is `customer_unique_id`.

## 5. Column Descriptions

**customers**
- `customer_id` – key to the orders table (changes with every order)
- `customer_unique_id` – the real, unique customer identifier
- `customer_zip_code_prefix` – first five digits of the customer's zip code
- `customer_city`, `customer_state` – customer location

**orders**
- `order_id` – unique order identifier
- `customer_id` – links the order to a customer
- `order_status` – delivered, shipped, canceled, etc.
- `order_purchase_timestamp` – when the purchase was made
- `order_approved_at` – when payment was approved
- `order_delivered_carrier_date` – when the order was handed to the carrier
- `order_delivered_customer_date` – actual delivery date to the customer
- `order_estimated_delivery_date` – estimated delivery date shown at purchase

**order_items**
- `order_id`, `order_item_id` – order and item sequence number within it
- `product_id`, `seller_id` – the product sold and who sold it
- `shipping_limit_date` – seller's deadline to hand the item to the carrier
- `price` – item price
- `freight_value` – shipping cost for the item

**order_payments**
- `order_id` – the order paid for
- `payment_sequential` – sequence number when multiple payment methods are used
- `payment_type` – credit card, boleto, voucher, debit card
- `payment_installments` – number of installments chosen
- `payment_value` – amount paid

**order_reviews**
- `review_id`, `order_id` – review and its order
- `review_score` – satisfaction rating from 1 to 5
- `review_comment_title`, `review_comment_message` – optional comment text
- `review_creation_date` – when the survey was sent
- `review_answer_timestamp` – when the customer answered

**products**
- `product_id` – unique product identifier
- `product_category_name` – category (Portuguese)
- `product_name_lenght`, `product_description_lenght` – character counts of name and description
- `product_photos_qty` – number of product photos
- `product_weight_g` – weight in grams
- `product_length_cm`, `product_height_cm`, `product_width_cm` – dimensions in cm

**sellers**
- `seller_id` – unique seller identifier
- `seller_zip_code_prefix`, `seller_city`, `seller_state` – seller location

**geolocation**
- `geolocation_zip_code_prefix` – zip code prefix
- `geolocation_lat`, `geolocation_lng` – coordinates
- `geolocation_city`, `geolocation_state` – location names

**category_translation**
- `product_category_name` – Portuguese name
- `product_category_name_english` – English name

## 6. Database Tables

| Table | Main columns |
|---|---|
| `customers` | **customer_id (PK)**, customer_unique_id, customer_city, customer_state, customer_zip_code_prefix |
| `orders` | **order_id (PK)**, customer_id (FK), order_status, order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date |
| `order_items` | **(order_id, order_item_id) (PK)**, order_id (FK), product_id (FK), seller_id (FK), price, freight_value |
| `order_payments` | **(order_id, payment_sequential) (PK)**, order_id (FK), payment_type, payment_installments, payment_value |
| `order_reviews` | **review_id (PK)**, order_id (FK), review_score, review_creation_date |
| `products` | **product_id (PK)**, product_category_name (FK), product_weight_g, product_photos_qty |
| `sellers` | **seller_id (PK)**, seller_city, seller_state |
| `category_translation` | **product_category_name (PK)**, product_category_name_english |
| `customer_features` _(derived, used by the ML model)_ | **customer_unique_id (PK)**, first_order_value, delivery_delay_days, first_review_score, payment_type, installments, state, top_category, is_repeat_buyer (target) |

## 7. Machine-Learning Task

- **Task:** Repeat-Purchase Prediction (binary classification) – will a customer place another order after their first one?
- **Features:** taken from the customer's first order only (to avoid data leakage).
- **Challenge:** only about 3% of customers are repeat buyers → class imbalance (handled with class weights / SMOTE; evaluated with Recall, F1, ROC-AUC, not Accuracy).
- **Models:** Logistic Regression (baseline), Random Forest, XGBoost / LightGBM.

## 8. Dashboard Plan

**KPI cards**
- Total revenue · Total orders · Number of unique customers · Average order value · Repeat-customer rate · Average review score

**Charts**
- Line chart: monthly revenue and orders
- Bar chart: top 10 categories by revenue
- Map / bar chart: customers and revenue by state
- Bar chart: average review score by delivery-delay bucket
- Pie / bar chart: payment types and installments
- Customer segments chart (e.g., RFM or repeat vs. one-time)
- Model section: feature importance, confusion matrix, ROC curve, and a table of customers with their repeat-purchase probability
- Filters: date range, state, category

## 9. Tools & Technologies

| Purpose | Tools |
|---|---|
| Database | PostgreSQL (or SQLite) + SQL |
| Data processing & analysis | Python, pandas, numpy |
| Visualization | matplotlib, seaborn, Plotly |
| Machine learning | scikit-learn, XGBoost / LightGBM, imbalanced-learn |
| Dashboard | Plotly Dash |
| Experiment tracking _(optional)_ | MLflow |
| Model serving _(optional)_ | FastAPI |
| Version control | Git & GitHub |

## 10. Repository Structure

```
├── README.md
├── data/            # raw and processed data (or download instructions)
├── sql/             # schema + KPI queries
├── notebooks/       # cleaning, EDA, modeling
├── src/             # reusable Python code
├── dashboard/       # Dash app
├── docs/            # diagrams, reports, presentations
└── requirements.txt
```

## 11. Team Roles

| Member | Role |
|---|---|
| _[Ahmed Mostafa]_ | Database & SQL |
| _[Mohamed Essam]_ | Data cleaning |
|_[Omar Bastawisy]_ |EDA|
| _[Assem Abdelgawad]_ | Machine learning |
| _[Menna Mohamed]_ | Dashboard |
| _[all_Team]_ | Documentation & presentation |

## 12. How to Run

_To be completed at the implementation stage: installation steps, database setup, running notebooks, and starting the dashboard._
