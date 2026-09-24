{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# E-Commerce Sales Analysis Dashboard with AI\n",
    "\n",
    "**Project type:** Data Analytics + AI/ML  \n",
    "**Dataset:** Brazilian E-Commerce Public Dataset by Olist  \n",
    "**Main tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Statsmodels  \n",
    "**Goal:** Analyze e-commerce sales performance, customer behavior and product/category trends, then add AI-based customer segmentation and sales forecasting.\n",
    "\n",
    "> Dataset source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Install/import libraries\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "print(\"Libraries loaded successfully.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset setup\n",
    "\n",
    "Download the Olist CSV files from Kaggle and place them in a folder named `data/` next to this notebook.\n",
    "\n",
    "Required files:\n",
    "- `olist_orders_dataset.csv`\n",
    "- `olist_order_items_dataset.csv`\n",
    "- `olist_products_dataset.csv`\n",
    "- `olist_customers_dataset.csv`\n",
    "- `olist_order_payments_dataset.csv`\n",
    "- `product_category_name_translation.csv` (optional)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Load data\n",
    "DATA_DIR = Path(\"data\")\n",
    "\n",
    "required = [\n",
    "    \"olist_orders_dataset.csv\",\n",
    "    \"olist_order_items_dataset.csv\",\n",
    "    \"olist_products_dataset.csv\",\n",
    "    \"olist_customers_dataset.csv\",\n",
    "    \"olist_order_payments_dataset.csv\"\n",
    "]\n",
    "\n",
    "missing = [f for f in required if not (DATA_DIR / f).exists()]\n",
    "if missing:\n",
    "    raise FileNotFoundError(\n",
    "        \"Missing files: \" + \", \".join(missing) +\n",
    "        \". Download the Olist dataset and place the CSV files inside ./data/\"\n",
    "    )\n",
    "\n",
    "orders = pd.read_csv(DATA_DIR / \"olist_orders_dataset.csv\")\n",
    "items = pd.read_csv(DATA_DIR / \"olist_order_items_dataset.csv\")\n",
    "products = pd.read_csv(DATA_DIR / \"olist_products_dataset.csv\")\n",
    "customers = pd.read_csv(DATA_DIR / \"olist_customers_dataset.csv\")\n",
    "payments = pd.read_csv(DATA_DIR / \"olist_order_payments_dataset.csv\")\n",
    "\n",
    "translation_path = DATA_DIR / \"product_category_name_translation.csv\"\n",
    "translation = pd.read_csv(translation_path) if translation_path.exists() else None\n",
    "\n",
    "for df_name, df in {\n",
    "    \"orders\": orders, \"items\": items, \"products\": products,\n",
    "    \"customers\": customers, \"payments\": payments\n",
    "}.items():\n",
    "    print(df_name, df.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Data cleaning and preparation\n",
    "date_cols = [\n",
    "    \"order_purchase_timestamp\", \"order_approved_at\",\n",
    "    \"order_delivered_carrier_date\", \"order_delivered_customer_date\",\n",
    "    \"order_estimated_delivery_date\"\n",
    "]\n",
    "for c in date_cols:\n",
    "    orders[c] = pd.to_datetime(orders[c], errors=\"coerce\")\n",
    "\n",
    "# Remove duplicate rows\n",
    "orders = orders.drop_duplicates()\n",
    "items = items.drop_duplicates()\n",
    "products = products.drop_duplicates()\n",
    "customers = customers.drop_duplicates()\n",
    "payments = payments.drop_duplicates()\n",
    "\n",
    "# Fill product category with a readable label\n",
    "products[\"product_category_name\"] = products[\"product_category_name\"].fillna(\"unknown\")\n",
    "\n",
    "# Translate categories when translation file is available\n",
    "if translation is not None:\n",
    "    products = products.merge(translation, on=\"product_category_name\", how=\"left\")\n",
    "    products[\"category\"] = products[\"product_category_name_english\"].fillna(\n",
    "        products[\"product_category_name\"]\n",
    "    )\n",
    "else:\n",
    "    products[\"category\"] = products[\"product_category_name\"]\n",
    "\n",
    "print(\"Cleaning completed.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Build the analytical dataset\n",
    "df = (\n",
    "    items.merge(orders, on=\"order_id\", how=\"left\")\n",
    "         .merge(products[[\"product_id\", \"category\"]], on=\"product_id\", how=\"left\")\n",
    "         .merge(customers[[\"customer_id\", \"customer_unique_id\", \"customer_state\"]], on=\"customer_id\", how=\"left\")\n",
    ")\n",
    "\n",
    "# Payment totals by order\n",
    "payment_summary = payments.groupby(\"order_id\", as_index=False).agg(\n",
    "    payment_value=(\"payment_value\", \"sum\"),\n",
    "    payment_installments=(\"payment_installments\", \"max\")\n",
    ")\n",
    "\n",
    "df = df.merge(payment_summary, on=\"order_id\", how=\"left\")\n",
    "df[\"revenue\"] = df[\"price\"].fillna(0) + df[\"freight_value\"].fillna(0)\n",
    "df[\"purchase_date\"] = df[\"order_purchase_timestamp\"].dt.date\n",
    "df[\"year_month\"] = df[\"order_purchase_timestamp\"].dt.to_period(\"M\").astype(str)\n",
    "df[\"delivery_days\"] = (\n",
    "    df[\"order_delivered_customer_date\"] - df[\"order_purchase_timestamp\"]\n",
    ").dt.total_seconds() / 86400\n",
    "\n",
    "df[\"is_late\"] = (\n",
    "    df[\"order_delivered_customer_date\"] > df[\"order_estimated_delivery_date\"]\n",
    ").fillna(False)\n",
    "\n",
    "print(\"Analytical dataset:\", df.shape)\n",
    "display(df.head())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Executive KPIs\n",
    "\n",
    "The dashboard focuses on:\n",
    "- Total revenue\n",
    "- Total orders\n",
    "- Unique customers\n",
    "- Average order value\n",
    "- Average delivery time\n",
    "- Late-delivery rate\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. KPI calculations\n",
    "total_revenue = df[\"revenue\"].sum()\n",
    "total_orders = df[\"order_id\"].nunique()\n",
    "unique_customers = df[\"customer_unique_id\"].nunique()\n",
    "aov = total_revenue / total_orders if total_orders else 0\n",
    "avg_delivery = df[\"delivery_days\"].mean()\n",
    "late_rate = df[\"is_late\"].mean() * 100\n",
    "\n",
    "kpis = pd.DataFrame({\n",
    "    \"KPI\": [\"Total Revenue\", \"Total Orders\", \"Unique Customers\",\n",
    "            \"Average Order Value\", \"Avg Delivery Days\", \"Late Delivery Rate\"],\n",
    "    \"Value\": [total_revenue, total_orders, unique_customers,\n",
    "              aov, avg_delivery, late_rate]\n",
    "})\n",
    "display(kpis)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis\n",
    "\n",
    "The following visualizations identify monthly sales trends, top categories, state-level sales and order delivery performance.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. Monthly revenue trend\n",
    "monthly_sales = df.groupby(\"year_month\", as_index=False).agg(\n",
    "    revenue=(\"revenue\", \"sum\"),\n",
    "    orders=(\"order_id\", \"nunique\")\n",
    ")\n",
    "\n",
    "fig = px.line(\n",
    "    monthly_sales, x=\"year_month\", y=\"revenue\",\n",
    "    markers=True, title=\"Monthly Revenue Trend\"\n",
    ")\n",
    "fig.update_layout(xaxis_title=\"Month\", yaxis_title=\"Revenue\")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8. Top product categories\n",
    "category_sales = (\n",
    "    df.groupby(\"category\", as_index=False)\n",
    "      .agg(revenue=(\"revenue\", \"sum\"), orders=(\"order_id\", \"nunique\"))\n",
    "      .sort_values(\"revenue\", ascending=False)\n",
    "      .head(15)\n",
    ")\n",
    "\n",
    "fig = px.bar(\n",
    "    category_sales.sort_values(\"revenue\"),\n",
    "    x=\"revenue\", y=\"category\", orientation=\"h\",\n",
    "    title=\"Top 15 Product Categories by Revenue\"\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9. Revenue by customer state\n",
    "state_sales = (\n",
    "    df.groupby(\"customer_state\", as_index=False)\n",
    "      .agg(revenue=(\"revenue\", \"sum\"), orders=(\"order_id\", \"nunique\"))\n",
    "      .sort_values(\"revenue\", ascending=False)\n",
    ")\n",
    "\n",
    "fig = px.bar(\n",
    "    state_sales.head(15),\n",
    "    x=\"customer_state\", y=\"revenue\",\n",
    "    title=\"Top States by Revenue\"\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10. Delivery performance\n",
    "delivery_summary = df.groupby(\"is_late\", as_index=False).agg(\n",
    "    orders=(\"order_id\", \"nunique\")\n",
    ")\n",
    "delivery_summary[\"status\"] = delivery_summary[\"is_late\"].map(\n",
    "    {False: \"On Time\", True: \"Late\"}\n",
    ")\n",
    "\n",
    "fig = px.pie(\n",
    "    delivery_summary, names=\"status\", values=\"orders\",\n",
    "    title=\"Order Delivery Performance\"\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. AI/ML Module 1 \u2014 Customer Segmentation\n",
    "\n",
    "RFM-style features are created for each customer:\n",
    "- **Recency:** days since the customer's latest purchase\n",
    "- **Frequency:** number of unique orders\n",
    "- **Monetary:** total revenue\n",
    "\n",
    "K-Means clustering groups customers with similar purchasing behavior.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 11. Prepare customer-level RFM data\n",
    "reference_date = df[\"order_purchase_timestamp\"].max() + pd.Timedelta(days=1)\n",
    "\n",
    "rfm = df.groupby(\"customer_unique_id\").agg(\n",
    "    last_purchase=(\"order_purchase_timestamp\", \"max\"),\n",
    "    frequency=(\"order_id\", \"nunique\"),\n",
    "    monetary=(\"revenue\", \"sum\")\n",
    ").reset_index()\n",
    "\n",
    "rfm[\"recency\"] = (reference_date - rfm[\"last_purchase\"]).dt.days\n",
    "rfm = rfm[[\"customer_unique_id\", \"recency\", \"frequency\", \"monetary\"]].copy()\n",
    "\n",
    "# Log transformation reduces the effect of extreme values\n",
    "rfm_model = rfm[[\"recency\", \"frequency\", \"monetary\"]].copy()\n",
    "rfm_model = np.log1p(rfm_model)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(rfm_model)\n",
    "\n",
    "kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)\n",
    "rfm[\"cluster\"] = kmeans.fit_predict(X)\n",
    "\n",
    "cluster_profile = rfm.groupby(\"cluster\").agg(\n",
    "    customers=(\"customer_unique_id\", \"count\"),\n",
    "    avg_recency=(\"recency\", \"mean\"),\n",
    "    avg_frequency=(\"frequency\", \"mean\"),\n",
    "    avg_monetary=(\"monetary\", \"mean\")\n",
    ").reset_index()\n",
    "\n",
    "display(cluster_profile)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12. Customer segment visualization\n",
    "fig = px.scatter(\n",
    "    rfm, x=\"recency\", y=\"monetary\",\n",
    "    color=rfm[\"cluster\"].astype(str),\n",
    "    size=\"frequency\",\n",
    "    hover_data=[\"frequency\"],\n",
    "    title=\"AI Customer Segmentation \u2014 Recency vs Monetary Value\",\n",
    "    labels={\"color\": \"Cluster\"}\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Interpreting the clusters\n",
    "\n",
    "The cluster numbers are machine-generated labels and do not inherently mean \"good\" or \"bad\".\n",
    "Interpret each cluster from its average recency, frequency and monetary values. For example:\n",
    "- Low recency + high frequency + high monetary value \u2192 highly active/high-value behavior.\n",
    "- High recency + low frequency + low monetary value \u2192 low-engagement behavior.\n",
    "- Intermediate profiles \u2192 occasional or emerging customers.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. AI/ML Module 2 \u2014 Monthly Sales Forecasting\n",
    "\n",
    "A Random Forest regression model forecasts monthly revenue using calendar features and lagged revenue. The final months are held out as a time-based test set to avoid random leakage.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 13. Create forecasting dataset\n",
    "forecast = monthly_sales.copy()\n",
    "forecast[\"date\"] = pd.to_datetime(forecast[\"year_month\"])\n",
    "forecast = forecast.sort_values(\"date\").reset_index(drop=True)\n",
    "\n",
    "forecast[\"month\"] = forecast[\"date\"].dt.month\n",
    "forecast[\"year\"] = forecast[\"date\"].dt.year\n",
    "forecast[\"time_index\"] = np.arange(len(forecast))\n",
    "forecast[\"lag_1\"] = forecast[\"revenue\"].shift(1)\n",
    "forecast[\"lag_2\"] = forecast[\"revenue\"].shift(2)\n",
    "forecast[\"rolling_3\"] = forecast[\"revenue\"].shift(1).rolling(3).mean()\n",
    "\n",
    "model_df = forecast.dropna().copy()\n",
    "\n",
    "features = [\"month\", \"year\", \"time_index\", \"lag_1\", \"lag_2\", \"rolling_3\"]\n",
    "X_all = model_df[features]\n",
    "y_all = model_df[\"revenue\"]\n",
    "\n",
    "# Last 20% as time-based test set\n",
    "split = max(1, int(len(model_df) * 0.8))\n",
    "X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]\n",
    "y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]\n",
    "\n",
    "rf_model = RandomForestRegressor(\n",
    "    n_estimators=300, random_state=42, max_depth=8, min_samples_leaf=2\n",
    ")\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "pred = rf_model.predict(X_test)\n",
    "\n",
    "mae = mean_absolute_error(y_test, pred)\n",
    "rmse = mean_squared_error(y_test, pred) ** 0.5\n",
    "r2 = r2_score(y_test, pred)\n",
    "\n",
    "print(f\"MAE: {mae:,.2f}\")\n",
    "print(f\"RMSE: {rmse:,.2f}\")\n",
    "print(f\"R\u00b2: {r2:,.3f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 14. Actual vs predicted revenue\n",
    "test_results = model_df.iloc[split:][[\"date\", \"revenue\"]].copy()\n",
    "test_results[\"predicted_revenue\"] = pred\n",
    "\n",
    "fig = go.Figure()\n",
    "fig.add_trace(go.Scatter(\n",
    "    x=test_results[\"date\"], y=test_results[\"revenue\"],\n",
    "    mode=\"lines+markers\", name=\"Actual\"\n",
    "))\n",
    "fig.add_trace(go.Scatter(\n",
    "    x=test_results[\"date\"], y=test_results[\"predicted_revenue\"],\n",
    "    mode=\"lines+markers\", name=\"Predicted\"\n",
    "))\n",
    "fig.update_layout(\n",
    "    title=\"AI Sales Forecast \u2014 Actual vs Predicted\",\n",
    "    xaxis_title=\"Month\", yaxis_title=\"Revenue\"\n",
    ")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 15. Forecast the next 3 months recursively\n",
    "history = forecast[[\"date\", \"revenue\"]].copy()\n",
    "future_rows = []\n",
    "\n",
    "for _ in range(3):\n",
    "    next_date = history[\"date\"].max() + pd.offsets.MonthBegin(1)\n",
    "    lag_1 = history[\"revenue\"].iloc[-1]\n",
    "    lag_2 = history[\"revenue\"].iloc[-2]\n",
    "    rolling_3 = history[\"revenue\"].iloc[-3:].mean()\n",
    "\n",
    "    row = pd.DataFrame([{\n",
    "        \"month\": next_date.month,\n",
    "        \"year\": next_date.year,\n",
    "        \"time_index\": len(history),\n",
    "        \"lag_1\": lag_1,\n",
    "        \"lag_2\": lag_2,\n",
    "        \"rolling_3\": rolling_3\n",
    "    }])\n",
    "\n",
    "    predicted_revenue = rf_model.predict(row[features])[0]\n",
    "    future_rows.append({\"date\": next_date, \"forecast_revenue\": predicted_revenue})\n",
    "    history = pd.concat([\n",
    "        history,\n",
    "        pd.DataFrame([{\"date\": next_date, \"revenue\": predicted_revenue}])\n",
    "    ], ignore_index=True)\n",
    "\n",
    "future_forecast = pd.DataFrame(future_rows)\n",
    "display(future_forecast)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. AI-assisted business insights\n",
    "\n",
    "The project converts model outputs into interpretable business observations rather than treating the model as a black box.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 16. Generate simple data-driven insights\n",
    "top_category = category_sales.iloc[0]\n",
    "top_state = state_sales.iloc[0]\n",
    "best_cluster = cluster_profile.sort_values(\n",
    "    [\"avg_monetary\", \"avg_frequency\"], ascending=False\n",
    ").iloc[0]\n",
    "\n",
    "print(\"KEY INSIGHTS\")\n",
    "print(f\"1. Highest-revenue category: {top_category['category']} \"\n",
    "      f\"with revenue of {top_category['revenue']:,.2f}.\")\n",
    "print(f\"2. Highest-revenue customer state: {top_state['customer_state']} \"\n",
    "      f\"with revenue of {top_state['revenue']:,.2f}.\")\n",
    "print(f\"3. Cluster {int(best_cluster['cluster'])} has the highest average monetary value \"\n",
    "      f\"({best_cluster['avg_monetary']:,.2f}).\")\n",
    "print(f\"4. Overall late-delivery rate: {late_rate:.2f}%.\")\n",
    "print(f\"5. Three-month model forecast total: \"\n",
    "      f\"{future_forecast['forecast_revenue'].sum():,.2f}.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Dashboard-ready dataset export\n",
    "\n",
    "The following files can be imported into Power BI or used by a Streamlit dashboard.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 17. Export dashboard datasets\n",
    "OUTPUT_DIR = Path(\"outputs\")\n",
    "OUTPUT_DIR.mkdir(exist_ok=True)\n",
    "\n",
    "df.to_csv(OUTPUT_DIR / \"ecommerce_transaction_data.csv\", index=False)\n",
    "monthly_sales.to_csv(OUTPUT_DIR / \"monthly_sales.csv\", index=False)\n",
    "category_sales.to_csv(OUTPUT_DIR / \"category_sales.csv\", index=False)\n",
    "state_sales.to_csv(OUTPUT_DIR / \"state_sales.csv\", index=False)\n",
    "rfm.to_csv(OUTPUT_DIR / \"customer_segments.csv\", index=False)\n",
    "future_forecast.to_csv(OUTPUT_DIR / \"sales_forecast.csv\", index=False)\n",
    "\n",
    "print(f\"Exported dashboard datasets to {OUTPUT_DIR.resolve()}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Suggested dashboard layout\n",
    "\n",
    "**Page 1 \u2014 Executive Overview**\n",
    "- KPI cards: Revenue, Orders, Customers, AOV, Late Delivery %\n",
    "- Monthly Revenue Trend\n",
    "- Top 10 Categories\n",
    "- Revenue by State\n",
    "\n",
    "**Page 2 \u2014 Customer Analytics**\n",
    "- Customer segment distribution\n",
    "- RFM scatter plot\n",
    "- Segment-level revenue and order frequency\n",
    "\n",
    "**Page 3 \u2014 AI Forecast**\n",
    "- Actual vs predicted revenue\n",
    "- Next 3-month forecast\n",
    "- Forecast error metrics (MAE, RMSE, R\u00b2)\n",
    "\n",
    "**Page 4 \u2014 Operations**\n",
    "- On-time vs late deliveries\n",
    "- Average delivery days\n",
    "- Category/state filters\n",
    "\n",
    "Recommended Power BI slicers: Year/Month, Category, Customer State, Order Status.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Conclusion\n",
    "\n",
    "This project combines descriptive analytics with machine learning. The analytics layer explains what happened in the e-commerce business, while K-Means segmentation identifies customer behavior groups and Random Forest forecasting estimates future monthly revenue. The exported tables are ready for an interactive Power BI dashboard.\n",
    "\n",
    "**Important:** Forecast quality depends on the available historical period and data quality. The model is a demonstration/decision-support tool, not a guarantee of future sales.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
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

**Do not upload the raw Kaggle dataset if its license/size makes that inappropriate.** Instead, keep the Kaggle link in README.md and provide setup instructions.

### 4. Commit and push
```bash
git add .
git commit -m "Add e-commerce sales analysis AI project"
git push origin main
```

## Suggested GitHub Description
> E-commerce sales analytics dashboard using Python, Power BI and machine learning for customer segmentation and sales forecasting.

## Resume Project Entry
**E-Commerce Sales Analysis Dashboard with AI | Python, Power BI, Machine Learning**
- Analyzed e-commerce transactions to identify revenue, category, customer and delivery trends.
- Built KPI-driven dashboard datasets for revenue, orders, AOV and delivery performance.
- Applied K-Means clustering to segment customers using RFM features.
- Built a Random Forest model for monthly revenue forecasting and evaluated it using MAE, RMSE and R².
- Exported cleaned and modeled datasets for interactive Power BI visualization.

## Limitations
- The forecasting model is intended for demonstration and decision support.
- Forecast accuracy depends on the historical data period and data quality.
- Customer cluster numbers are labels; business meaning must be interpreted from cluster profiles.

## License / Dataset Note
Review the dataset provider's terms before redistributing the raw CSV files. The notebook and project code can be version-controlled separately from the raw data.
