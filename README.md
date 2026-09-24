# E-Commerce Sales Analysis Dashboard with AI

## Project Overview
This project analyzes e-commerce sales data and converts it into an interactive, decision-support dashboard. It combines traditional data analytics with AI/ML techniques for customer segmentation and sales forecasting.

## Business Problem
E-commerce businesses generate large volumes of order, product, customer and payment data. Raw data makes it difficult to identify sales trends, high-value customer groups, category performance and future revenue expectations.

## Solution
The project:
1. Cleans and combines e-commerce datasets.
2. Calculates sales KPIs and business metrics.
3. Performs exploratory data analysis.
4. Creates dashboard-ready datasets.
5. Uses **K-Means clustering** for customer segmentation.
6. Uses **Random Forest Regression** for monthly sales forecasting.
7. Exports outputs for Power BI or another BI dashboard.

## Dataset
**Brazilian E-Commerce Public Dataset by Olist (Kaggle):**  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset contains information about orders, order items, customers, products, payments, sellers and delivery dates.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Power BI (recommended for final dashboard)

## AI/ML Features
### 1. Customer Segmentation
RFM features:
- Recency
- Frequency
- Monetary value

K-Means creates four behavior-based customer clusters.

### 2. Sales Forecasting
A Random Forest regression model uses:
- Month
- Year
- Time index
- Previous-month revenue
- Two-month lag revenue
- Three-month rolling average

The final 20% of historical months are used as a time-based test set.

## Dashboard Pages
### Executive Overview
- Total Revenue
- Total Orders
- Unique Customers
- Average Order Value
- Late Delivery Rate
- Monthly Revenue Trend
- Top Categories
- Revenue by State

### Customer Analytics
- Customer segments
- RFM scatter plot
- Cluster profiles
- Customer value distribution

### AI Forecast
- Actual vs predicted sales
- Forecast metrics
- Next 3-month revenue forecast

### Operations
- Delivery performance
- Average delivery days
- Category/state analysis

## Project Structure
```text
Nagendra_ECommerce_Sales_AI_Project/
│
├── Nagendra_ECommerce_Sales_AI_Project.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   └── product_category_name_translation.csv
└── outputs/
    ├── ecommerce_transaction_data.csv
    ├── monthly_sales.csv
    ├── category_sales.csv
    ├── state_sales.csv
    ├── customer_segments.csv
    └── sales_forecast.csv
```

## Setup Instructions
1. Install Python 3.10+.
2. Download the Olist dataset from Kaggle.
3. Create a `data` folder beside the notebook.
4. Put the required CSV files inside `data/`.
5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Start Jupyter:

```bash
jupyter notebook
```

7. Open `Nagendra_ECommerce_Sales_AI_Project.ipynb`.
8. Run the notebook from top to bottom.
9. Import the generated `outputs/` CSV files into Power BI if you want a separate interactive dashboard.

## How to Add the Project to GitHub

### 1. Create a repository
Create a new GitHub repository, for example:

`ecommerce-sales-analysis-ai-dashboard`

### 2. Clone it
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-sales-analysis-ai-dashboard.git
cd ecommerce-sales-analysis-ai-dashboard
```

### 3. Copy project files
Add:
- Notebook
- requirements.txt
- README.md
- Power BI `.pbix` file if created
- screenshots (optional)

