
import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import json
import calendar

# --------------------------------------------
# Human-friendly labels (fix for NameError)
# --------------------------------------------
MONTH_NAMES = list(calendar.month_name[1:])   # ['January', ..., 'December']
DAY_NAMES   = list(calendar.day_name)         # ['Monday', ..., 'Sunday']

# --------------------------------------------
# Page config & styles
# --------------------------------------------
st.set_page_config(
    page_title="Olist Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# State mappings for better UX
# --------------------------------------------
STATE_MAPPING = {
    'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas', 'BA': 'Bahia',
    'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás',
    'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
    'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco', 'PI': 'Piauí',
    'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo',
    'SE': 'Sergipe', 'TO': 'Tocantins'
}
REVERSE_STATE_MAPPING = {v: k for k, v in STATE_MAPPING.items()}

# --------------------------------------------
# Data / model loaders
# --------------------------------------------
@st.cache_data
def load_data():
    data_path = 'data/'
    datasets = {}
    files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'category_translation': 'product_category_name_translation.csv'
    }
    for name, filename in files.items():
        try:
            datasets[name] = pd.read_csv(os.path.join(data_path, filename))
        except FileNotFoundError:
            st.error(f"File not found: {filename}")
            return None
    return datasets

@st.cache_data
def load_models():
    try:
        artifacts = joblib.load('models/olist_model_artifacts.pkl')
        return artifacts
    except ImportError:
        st.error("❌ joblib module not found. Please run: pip install joblib")
        return None
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please ensure olist_model_artifacts.pkl is in the models/ folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# --------------------------------------------
# App router
# --------------------------------------------
def main():
    datasets = load_data()
    model_artifacts = load_models()
    if datasets is None or model_artifacts is None:
        st.stop()

    st.markdown('<h1 class="main-header">🛒 Olist Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced E-commerce Analytics & Predictive Intelligence")

    st.sidebar.title("📊 Navigation")
    pages = {
        "🏠 Executive Dashboard": "dashboard",
        "🚚 Delivery Prediction": "delivery",
        "👥 Customer Intelligence": "customers",
        "📈 Sales Analytics": "sales",
        "🎯 Business Insights": "insights"
    }
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    current_page = pages[selected_page]

    if current_page == "dashboard":
        show_dashboard(datasets, model_artifacts)
    elif current_page == "delivery":
        show_delivery_prediction(datasets, model_artifacts)
    elif current_page == "customers":
        show_customer_intelligence(datasets, model_artifacts)
    elif current_page == "sales":
        show_sales_analytics(datasets, model_artifacts)
    elif current_page == "insights":
        show_business_insights(datasets, model_artifacts)

# --------------------------------------------
# Pages
# --------------------------------------------
def show_dashboard(datasets, model_artifacts):
    st.header("🏠 Executive Dashboard")
    st.markdown("#### Key Business Metrics & Performance Overview")

    orders_df = datasets['orders'].copy()
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])

    items_df = datasets['order_items']
    revenue_data = orders_df.merge(
        items_df.groupby('order_id')['price'].sum().reset_index().rename(columns={'price': 'total_revenue'}),
        on='order_id', how='left'
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{len(orders_df):,}")
    with col2:
        st.metric("Total Revenue", f"R$ {revenue_data['total_revenue'].sum():,.0f}")
    with col3:
        st.metric("Average Order Value", f"R$ {revenue_data['total_revenue'].mean():.2f}")
    with col4:
        delivered_orders = len(orders_df[orders_df['order_status'] == 'delivered'])
        delivery_rate = (delivered_orders / len(orders_df)) * 100
        st.metric("Delivery Rate", f"{delivery_rate:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Order Trends")
        monthly_data = revenue_data.groupby(revenue_data['order_purchase_timestamp'].dt.to_period('M')).agg({
            'order_id': 'count',
            'total_revenue': 'sum'
        }).reset_index()
        monthly_data['order_purchase_timestamp'] = monthly_data['order_purchase_timestamp'].astype(str)
        fig = px.line(monthly_data, x='order_purchase_timestamp', y='order_id',
                      title='Orders Over Time', labels={'order_id': 'Number of Orders'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Order Status Distribution")
        status_counts = orders_df['order_status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title='Order Status Breakdown')
        st.plotly_chart(fig, use_container_width=True)

    if 'customer_segments' in model_artifacts:
        st.subheader("Customer Segmentation Insights")
        customer_data = model_artifacts['customer_segments'].copy()
        # Safety: ensure required columns exist
        if 'customer_segment' not in customer_data:
            customer_data['customer_segment'] = 'General'
        if 'total_monetary_value' not in customer_data:
            customer_data['total_monetary_value'] = 0.0

        col1, col2 = st.columns(2)
        with col1:
            segment_counts = customer_data.groupby('customer_segment').size()
            fig = px.bar(x=segment_counts.index, y=segment_counts.values,
                         title="Customer Segments", labels={'y': 'Number of Customers'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            segment_revenue = customer_data.groupby('customer_segment')['total_monetary_value'].sum()
            fig = px.bar(x=segment_revenue.index, y=segment_revenue.values,
                         title="Revenue by Segment", labels={'y': 'Total Revenue (R$)'})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Key Business Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="insight-box"><h4>📈 Growth Trajectory</h4>
        <p>Strong order volume growth with consistent month-over-month increases. Peak performance in Q2-Q3.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="insight-box"><h4>👥 Customer Base</h4>
        <p>Diverse customer segments identified. Focus on converting "At Risk" customers to loyal buyers.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="insight-box"><h4>🚚 Operations</h4>
        <p>High delivery success rate. Opportunity to optimize cross-state logistics for faster delivery.</p></div>""", unsafe_allow_html=True)

def show_delivery_prediction(datasets, model_artifacts):
    st.header("🚚 Advanced Delivery Prediction & Analytics")
    st.markdown("#### ML-Powered Logistics Intelligence with Interactive Analytics")

    if model_artifacts is None:
        st.error("Model artifacts not loaded")
        return

    delivery_model = model_artifacts['delivery_model']
    late_classifier = model_artifacts['late_classifier']
    label_encoders = model_artifacts['label_encoders']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Prediction", "📊 Batch Analysis", "🔄 Scenario Comparison", "🗺️ Geographic Insights", "📈 Historical Analysis"
    ])

    # ---------------- Single Prediction ----------------
    with tab1:
        st.subheader("Interactive Order Prediction")

        col1, col2 = st.columns([2, 1])
        with col1:
            # --- Geographic Configuration (clean version: uses selectboxes) ---
            st.markdown("**Geographic Configuration**")

            # Options like "SP — São Paulo"
            cust_opts = sorted(
                [(s, f"{s} — {STATE_MAPPING.get(s, s)}") for s in datasets['customers']['customer_state'].unique()],
                key=lambda x: x[1]
            )
            sell_opts = sorted(
                [(s, f"{s} — {STATE_MAPPING.get(s, s)}") for s in datasets['sellers']['seller_state'].unique()],
                key=lambda x: x[1]
            )

            # Defaults from session (fallback to 'SP')
            default_cust = next((i for i, (code, _) in enumerate(cust_opts) if code == st.session_state.get('customer_state', 'SP')), 0)
            default_sell = next((i for i, (code, _) in enumerate(sell_opts) if code == st.session_state.get('seller_state', 'SP')), 0)

            cust_choice = st.selectbox("Customer State", cust_opts, index=default_cust, format_func=lambda t: t[1])
            sell_choice = st.selectbox("Seller State", sell_opts, index=default_sell, format_func=lambda t: t[1])

            customer_state = cust_choice[0]
            seller_state = sell_choice[0]
            st.session_state.customer_state = customer_state
            st.session_state.seller_state = seller_state

            same_state = int(customer_state == seller_state)
            if same_state:
                st.success("✅ Same-state delivery - Faster shipping expected")
            else:
                st.warning("⚠️ Cross-state delivery - Extended shipping time expected")


            st.markdown("**Order Details**")
            col_items1, col_items2 = st.columns(2)
            with col_items1:
                n_items_slider = st.slider("Number of Items", 1, 10, 1, key="items_slider")
            with col_items2:
                n_items_manual = st.number_input("Or enter manually:", min_value=1, max_value=50, value=n_items_slider, key="items_manual")
            n_items = n_items_manual

            col_price1, col_price2 = st.columns(2)
            with col_price1:
                total_price_slider = st.slider("Total Order Value (R$)", 10, 1000, 100, key="price_slider")
            with col_price2:
                total_price_manual = st.number_input("Or enter manually (R$):", min_value=10.0, max_value=10000.0, value=float(total_price_slider), key="price_manual")
            total_price = total_price_manual

            col_freight1, col_freight2 = st.columns(2)
            with col_freight1:
                total_freight_slider = st.slider("Freight Cost (R$)", 5, 100, 20, key="freight_slider")
            with col_freight2:
                total_freight_manual = st.number_input("Or enter manually (R$):", min_value=5.0, max_value=500.0, value=float(total_freight_slider), key="freight_manual")
            total_freight = total_freight_manual

            st.markdown("**Timing Configuration**")
            col_month, col_day = st.columns(2)
            with col_month:
                month_selection = st.selectbox(
                    "Purchase Month",
                    [(i + 1, MONTH_NAMES[i]) for i in range(12)],
                    format_func=lambda x: x[1],
                    index=4
                )
                purchase_month = month_selection[0]
            with col_day:
                day_selection = st.selectbox(
                    "Day of Week",
                    [(i, DAY_NAMES[i]) for i in range(7)],
                    format_func=lambda x: x[1],
                    index=2
                )
                purchase_dow = day_selection[0]

        with col2:
            st.markdown("**Live Prediction Results**")
            try:
                customer_state_encoded = label_encoders['customer_state'].transform([customer_state])[0]
                seller_state_encoded = label_encoders['seller_state'].transform([seller_state])[0]
            except ValueError:
                customer_state_encoded = 0
                seller_state_encoded = 0
                st.warning("State not in training data - using default encoding")

            input_data = pd.DataFrame({
                'same_state': [same_state],
                'n_items': [n_items],
                'total_price': [total_price],
                'total_freight': [total_freight],
                'purchase_month': [purchase_month],
                'purchase_dow': [purchase_dow],
                'customer_state_encoded': [customer_state_encoded],
                'seller_state_encoded': [seller_state_encoded]
            })

            delivery_days = float(delivery_model.predict(input_data)[0])
            late_probability = float(late_classifier.predict_proba(input_data)[0][1])

            confidence_lower = max(1, delivery_days - 2.5)
            confidence_upper = delivery_days + 2.5

            st.markdown("---")
            st.metric("Predicted Delivery", f"{delivery_days:.1f} days",
                      delta=f"Range: {confidence_lower:.1f}-{confidence_upper:.1f}")
            st.metric("Late Risk", f"{late_probability:.1%}",
                      delta="High" if late_probability > 0.3 else "Medium" if late_probability > 0.15 else "Low")

            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=late_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Late Risk %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 30], 'color': "yellow"},
                        {'range': [30, 100], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}
                }
            ))
            fig_risk.update_layout(height=250)
            st.plotly_chart(fig_risk, use_container_width=True)

            st.markdown("**Historical Context**")
            st.info("📊 Same-state average: 7.9 days" if same_state else "📊 Cross-state average: 15.2 days")

            st.markdown("**Optimization Suggestions**")
            if not same_state and late_probability > 0.3:
                st.error("💡 Consider expedited shipping (+R$15) to reduce risk")
            elif total_freight > 30:
                st.warning("💡 High freight cost - consider bulk shipping")
            else:
                st.success("💡 Current configuration looks optimal")

    # ---------------- Batch Analysis ----------------
    with tab2:
        st.subheader("Batch Order Analysis")
        st.markdown("Upload multiple orders for bulk prediction analysis")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            st.markdown("**Expected CSV format:**")
            sample_df = pd.DataFrame({
                'customer_state': ['SP', 'RJ', 'MG'],
                'seller_state': ['SP', 'SP', 'RJ'],
                'n_items': [1, 2, 1],
                'total_price': [100, 250, 150],
                'total_freight': [15, 25, 20],
                'purchase_month': [5, 6, 7],
                'purchase_dow': [1, 3, 5]
            })
            st.dataframe(sample_df)

            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} orders for analysis")
                if st.button("🚀 Process Batch Predictions"):
                    batch_results = process_batch_predictions(batch_df, delivery_model, late_classifier, label_encoders)
                    st.subheader("Batch Results")
                    st.dataframe(batch_results)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Average Delivery", f"{batch_results['predicted_delivery'].mean():.1f} days")
                    with c2:
                        st.metric("High Risk Orders", f"{(batch_results['late_risk'] > 0.3).sum()}")
                    with c3:
                        st.metric("Same-State Orders", f"{batch_results['same_state'].sum()}")

                    csv = batch_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # ---------------- Scenario Comparison ----------------
    with tab3:
        st.subheader("Scenario Comparison")
        st.markdown("Compare different order configurations side-by-side")

        customer_states = sorted(datasets['customers']['customer_state'].unique())
        seller_states = sorted(datasets['sellers']['seller_state'].unique())
        num_scenarios = st.slider("Number of scenarios to compare", 2, 4, 2)

        scenario_results = []
        cols = st.columns(num_scenarios)

        for i in range(num_scenarios):
            with cols[i]:
                st.markdown(f"**Scenario {i+1}**")
                template = st.selectbox(f"Template {i+1}",
                                        ["Custom", "Same-State Standard", "Cross-State Express", "High-Value Order"],
                                        key=f"template_{i}")
                if template == "Same-State Standard":
                    sc_customer = sc_seller = 'SP'; sc_items, sc_price, sc_freight = 1, 100, 15; sc_month, sc_dow = 5, 2
                elif template == "Cross-State Express":
                    sc_customer, sc_seller = 'AM', 'SP'; sc_items, sc_price, sc_freight = 2, 300, 40; sc_month, sc_dow = 11, 4
                elif template == "High-Value Order":
                    sc_customer, sc_seller = 'RJ', 'RJ'; sc_items, sc_price, sc_freight = 5, 800, 60; sc_month, sc_dow = 12, 6
                else:
                    sc_customer = st.selectbox(f"Customer {i+1}", customer_states, key=f"sc_cust_{i}")
                    sc_seller = st.selectbox(f"Seller {i+1}", seller_states, key=f"sc_sell_{i}")
                    sc_items = st.number_input(f"Items {i+1}", 1, 10, 1, key=f"sc_items_{i}")
                    sc_price = st.number_input(f"Price {i+1}", 10, 1000, 100, key=f"sc_price_{i}")
                    sc_freight = st.number_input(f"Freight {i+1}", 5, 100, 20, key=f"sc_freight_{i}")
                    sc_month = st.selectbox(f"Month {i+1}", range(1, 13), key=f"sc_month_{i}")
                    sc_dow = st.selectbox(f"Day {i+1}", range(7), key=f"sc_dow_{i}")

                try:
                    sc_customer_encoded = label_encoders['customer_state'].transform([sc_customer])[0]
                    sc_seller_encoded = label_encoders['seller_state'].transform([sc_seller])[0]
                except Exception:
                    sc_customer_encoded = sc_seller_encoded = 0

                sc_same_state = 1 if sc_customer == sc_seller else 0
                sc_input = pd.DataFrame({
                    'same_state': [sc_same_state],
                    'n_items': [sc_items],
                    'total_price': [sc_price],
                    'total_freight': [sc_freight],
                    'purchase_month': [sc_month],
                    'purchase_dow': [sc_dow],
                    'customer_state_encoded': [sc_customer_encoded],
                    'seller_state_encoded': [sc_seller_encoded]
                })
                sc_delivery = float(delivery_model.predict(sc_input)[0])
                sc_risk = float(late_classifier.predict_proba(sc_input)[0][1])

                st.metric("Delivery", f"{sc_delivery:.1f} days")
                st.metric("Risk", f"{sc_risk:.1%}")

                scenario_results.append({
                    'scenario': f'Scenario {i+1}',
                    'delivery_days': sc_delivery,
                    'late_risk': sc_risk,
                    'customer_state': STATE_MAPPING.get(sc_customer, sc_customer),
                    'seller_state': STATE_MAPPING.get(sc_seller, sc_seller),
                    'same_state': sc_same_state
                })

        if len(scenario_results) > 1:
            comparison_df = pd.DataFrame(scenario_results)
            fig_comparison = make_subplots(rows=1, cols=2,
                                           subplot_titles=['Delivery Time Comparison', 'Risk Comparison'])
            fig_comparison.add_trace(go.Bar(x=comparison_df['scenario'], y=comparison_df['delivery_days'], name='Delivery Days'), row=1, col=1)
            fig_comparison.add_trace(go.Bar(x=comparison_df['scenario'], y=comparison_df['late_risk'], name='Late Risk'), row=1, col=2)
            fig_comparison.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comparison, use_container_width=True)

    # ---------------- Geographic Insights ----------------
    with tab4:
        st.subheader("Geographic Delivery Intelligence")
        create_delivery_map(datasets, delivery_model, label_encoders)

    # ---------------- Historical Analysis ----------------
    with tab5:
        st.subheader("Historical Performance Analysis")
        show_historical_analysis(datasets, model_artifacts)

def process_batch_predictions(batch_df, delivery_model, late_classifier, label_encoders):
    results = []
    for _, row in batch_df.iterrows():
        try:
            customer_encoded = label_encoders['customer_state'].transform([row['customer_state']])[0]
            seller_encoded = label_encoders['seller_state'].transform([row['seller_state']])[0]
        except Exception:
            customer_encoded = seller_encoded = 0
        same_state = 1 if row['customer_state'] == row['seller_state'] else 0
        input_data = pd.DataFrame({
            'same_state': [same_state],
            'n_items': [row['n_items']],
            'total_price': [row['total_price']],
            'total_freight': [row['total_freight']],
            'purchase_month': [row['purchase_month']],
            'purchase_dow': [row['purchase_dow']],
            'customer_state_encoded': [customer_encoded],
            'seller_state_encoded': [seller_encoded]
        })
        delivery_pred = float(delivery_model.predict(input_data)[0])
        risk_pred = float(late_classifier.predict_proba(input_data)[0][1])
        result = row.to_dict()
        result.update({
            'predicted_delivery': delivery_pred,
            'late_risk': risk_pred,
            'same_state': same_state,
            'risk_level': 'HIGH' if risk_pred > 0.3 else 'MEDIUM' if risk_pred > 0.15 else 'LOW'
        })
        results.append(result)
    return pd.DataFrame(results)

def create_delivery_map(datasets, delivery_model, label_encoders):
    st.markdown("Interactive map showing average delivery times between states")
    state_coords = {
        'SP': [-23.55, -46.64], 'RJ': [-22.91, -43.17], 'MG': [-19.92, -43.94],
        'RS': [-30.03, -51.23], 'PR': [-25.43, -49.27], 'SC': [-27.60, -48.55],
        'BA': [-12.97, -38.48], 'GO': [-16.69, -49.25], 'DF': [-15.78, -47.93],
        'PE': [-8.05, -34.88], 'CE': [-3.72, -38.54], 'PA': [-1.46, -48.50],
        'AM': [-3.12, -60.02], 'RO': [-8.76, -63.90], 'AC': [-9.97, -67.81]
    }
    m = folium.Map(location=[-14.235, -51.925], zoom_start=4)
    for state_code, coords in state_coords.items():
        if state_code in datasets['customers']['customer_state'].values:
            sample_input = pd.DataFrame({
                'same_state': [1], 'n_items': [1], 'total_price': [100],
                'total_freight': [15], 'purchase_month': [6], 'purchase_dow': [2],
                'customer_state_encoded': [0], 'seller_state_encoded': [0]
            })
            avg_delivery = float(delivery_model.predict(sample_input)[0])
            folium.Marker(
                coords,
                popup=f"{STATE_MAPPING.get(state_code, state_code)}<br>Avg Delivery: {avg_delivery:.1f} days",
                tooltip=STATE_MAPPING.get(state_code, state_code),
                icon=folium.Icon(color='green' if avg_delivery < 10 else 'orange' if avg_delivery < 15 else 'red')
            ).add_to(m)
    folium_static(m)

    st.markdown("**State-to-State Delivery Time Matrix**")
    common_states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA']
    matrix_data = []
    for customer_state in common_states:
        row = []
        for seller_state in common_states:
            same_state = 1 if customer_state == seller_state else 0
            sample_input = pd.DataFrame({
                'same_state': [same_state], 'n_items': [1], 'total_price': [100],
                'total_freight': [15], 'purchase_month': [6], 'purchase_dow': [2],
                'customer_state_encoded': [0], 'seller_state_encoded': [0]
            })
            pred_delivery = float(delivery_model.predict(sample_input)[0])
            row.append(pred_delivery)
        matrix_data.append(row)
    matrix_df = pd.DataFrame(matrix_data,
                             index=[STATE_MAPPING.get(s, s) for s in common_states],
                             columns=[STATE_MAPPING.get(s, s) for s in common_states])
    fig_heatmap = px.imshow(matrix_df, title="Predicted Delivery Days Matrix", color_continuous_scale="RdYlGn_r")
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_historical_analysis(datasets, model_artifacts):
    orders_df = datasets['orders'].copy()
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])

    delivered_orders = orders_df[orders_df['order_delivered_customer_date'].notna()].copy()
    delivered_orders['actual_delivery_days'] = (
        delivered_orders['order_delivered_customer_date'] - delivered_orders['order_purchase_timestamp']
    ).dt.total_seconds() / (24 * 3600)

    monthly_performance = delivered_orders.groupby(
        delivered_orders['order_purchase_timestamp'].dt.to_period('M')
    )['actual_delivery_days'].agg(['mean', 'std', 'count']).reset_index()
    monthly_performance['period'] = monthly_performance['order_purchase_timestamp'].astype(str)

    fig_trends = px.line(monthly_performance, x='period', y='mean',
                         error_y='std',
                         title='Historical Delivery Performance Trends',
                         labels={'mean': 'Average Delivery Days', 'period': 'Month'})
    fig_trends.update_layout(height=400)
    st.plotly_chart(fig_trends, use_container_width=True)

    seasonal_performance = delivered_orders.groupby(
        delivered_orders['order_purchase_timestamp'].dt.month
    )['actual_delivery_days'].mean().reset_index()
    seasonal_performance['month_name'] = seasonal_performance['order_purchase_timestamp'].map(
        lambda x: MONTH_NAMES[x - 1]
    )

    fig_seasonal = px.bar(seasonal_performance, x='month_name', y='actual_delivery_days',
                          title='Seasonal Delivery Performance',
                          labels={'actual_delivery_days': 'Average Delivery Days'})
    st.plotly_chart(fig_seasonal, use_container_width=True)

    st.markdown("**Historical Insights**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Historical Average", f"{delivered_orders['actual_delivery_days'].mean():.1f} days")
    with c2:
        best_month = seasonal_performance.loc[seasonal_performance['actual_delivery_days'].idxmin(), 'month_name']
        st.metric("Best Month", best_month)
    with c3:
        worst_month = seasonal_performance.loc[seasonal_performance['actual_delivery_days'].idxmax(), 'month_name']
        st.metric("Challenging Month", worst_month)

def show_customer_intelligence(datasets, model_artifacts):
    st.header("Customer Intelligence & Segmentation Analytics")
    st.markdown("#### Advanced RFM Analysis with Predictive Customer Insights")

    if model_artifacts is None or 'customer_segments' not in model_artifacts:
        st.error("Customer segmentation data not loaded")
        return

    customer_data = model_artifacts['customer_segments'].copy()
    # Safety: ensure required columns for charts exist
    for col, default in [('customer_id', np.arange(len(customer_data))),
                         ('total_monetary_value', 0.0),
                         ('order_frequency', 1),
                         ('recency_days', 90),
                         ('customer_segment', 'General')]:
        if col not in customer_data:
            customer_data[col] = default

    tab1, tab2, tab3, tab4 = st.tabs(["Customer Segments", "RFM Analysis", "Customer Lifetime Value", "Churn Risk Analysis"])

    with tab1:
        st.subheader("Interactive Customer Segmentation Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Customers", f"{len(customer_data):,}")
        with c2:
            st.metric("Total CLV", f"R$ {customer_data['total_monetary_value'].sum():,.0f}")
        with c3:
            st.metric("Average CLV", f"R$ {customer_data['total_monetary_value'].mean():.2f}")
        with c4:
            repeat_rate = (customer_data['order_frequency'] > 1).sum() / len(customer_data) * 100
            st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")

        col1, col2 = st.columns([2, 1])
        with col1:
            segment_counts = customer_data.groupby('customer_segment').agg({
                'customer_id': 'count',
                'total_monetary_value': ['sum', 'mean'],
                'order_frequency': 'mean',
                'recency_days': 'mean'
            }).round(2)
            segment_counts.columns = ['customer_count', 'total_revenue', 'avg_revenue', 'avg_frequency', 'avg_recency']
            segment_counts = segment_counts.reset_index()
            segment_counts['revenue_percent'] = (segment_counts['total_revenue'] / segment_counts['total_revenue'].sum() * 100).round(1)

            selected_segments = st.multiselect(
                "Select segments to analyze:",
                segment_counts['customer_segment'].tolist(),
                default=segment_counts['customer_segment'].tolist()
            )
            if selected_segments:
                filtered_data = segment_counts[segment_counts['customer_segment'].isin(selected_segments)]
                fig_segments = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Customer Count', 'Total Revenue', 'Average CLV', 'Revenue Share'],
                    specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'pie'}]]
                )
                fig_segments.add_trace(go.Bar(x=filtered_data['customer_segment'], y=filtered_data['customer_count']), row=1, col=1)
                fig_segments.add_trace(go.Bar(x=filtered_data['customer_segment'], y=filtered_data['total_revenue']), row=1, col=2)
                fig_segments.add_trace(go.Bar(x=filtered_data['customer_segment'], y=filtered_data['avg_revenue']), row=2, col=1)
                fig_segments.add_trace(go.Pie(labels=filtered_data['customer_segment'], values=filtered_data['revenue_percent']), row=2, col=2)
                fig_segments.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_segments, use_container_width=True)

        with col2:
            st.markdown("**Segment Insights**")
            top_revenue_segment = segment_counts.loc[segment_counts['total_revenue'].idxmax()]
            st.info(f"**Top Revenue Segment**\n{top_revenue_segment['customer_segment']}\nR$ {top_revenue_segment['total_revenue']:,.0f}")
            high_value_threshold = customer_data['total_monetary_value'].quantile(0.9)
            high_value_count = (customer_data['total_monetary_value'] > high_value_threshold).sum()
            st.success(f"**High-Value Customers**\n{high_value_count} customers\n(Top 10% by CLV)")
            at_risk_count = len(customer_data[customer_data['customer_segment'] == 'At Risk'])
            st.warning(f"**At-Risk Customers**\n{at_risk_count} customers\nNeed immediate attention")
            st.markdown("**Recommended Actions**")
            if at_risk_count > 0:
                st.error(f"• Re-engage {at_risk_count} at-risk customers with personalized offers")

    with tab2:
        st.subheader("Advanced RFM Analysis")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**RFM Configuration**")
            recency_threshold = st.slider("Recency Threshold (days)", 30, 365, 90)
            frequency_threshold = st.slider("Frequency Threshold (orders)", 1, 10, 2)
            monetary_threshold = st.slider("Monetary Threshold (R$)", 50, 1000, 200)
            st.markdown("**Analysis Filters**")
            min_clv = st.number_input("Minimum CLV (R$)", 0.0, 1000.0, 0.0)
            max_recency = st.number_input("Maximum Recency (days)", 0, 500, 500)
        with col2:
            filtered_customers = customer_data[
                (customer_data['total_monetary_value'] >= min_clv) &
                (customer_data['recency_days'] <= max_recency)
            ].copy()
            fig_rfm = px.scatter_3d(filtered_customers, x='recency_days', y='order_frequency', z='total_monetary_value',
                                    color='customer_segment', title='3D RFM Analysis',
                                    labels={'recency_days': 'Recency (Days)', 'order_frequency': 'Frequency (Orders)', 'total_monetary_value': 'Monetary (R$)'},
                                    height=500)
            st.plotly_chart(fig_rfm, use_container_width=True)

            fig_rfm_dist = make_subplots(rows=1, cols=3,
                                         subplot_titles=['Recency Distribution', 'Frequency Distribution', 'Monetary Distribution'])
            fig_rfm_dist.add_trace(go.Histogram(x=filtered_customers['recency_days'], nbinsx=20), row=1, col=1)
            fig_rfm_dist.add_trace(go.Histogram(x=filtered_customers['order_frequency'], nbinsx=20), row=1, col=2)
            fig_rfm_dist.add_trace(go.Histogram(x=filtered_customers['total_monetary_value'], nbinsx=20), row=1, col=3)
            fig_rfm_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_rfm_dist, use_container_width=True)

    with tab3:
        st.subheader("Customer Lifetime Value Prediction")
        col1, col2 = st.columns(2)
        with col1:
            fig_clv = px.histogram(customer_data, x='total_monetary_value', nbins=30, title='Customer Lifetime Value Distribution')
            fig_clv.update_layout(height=400)
            st.plotly_chart(fig_clv, use_container_width=True)
            st.markdown("**Top 10 Customers by CLV**")
            top_customers = customer_data.nlargest(10, 'total_monetary_value')[
                ['customer_id', 'total_monetary_value', 'order_frequency', 'recency_days', 'customer_segment']
            ]
            st.dataframe(top_customers, use_container_width=True)
        with col2:
            clv_by_segment = customer_data.groupby('customer_segment').agg({
                'total_monetary_value': ['mean', 'median', 'sum', 'count']
            }).round(2)
            clv_by_segment.columns = ['Mean CLV', 'Median CLV', 'Total CLV', 'Customer Count']
            clv_by_segment = clv_by_segment.reset_index()
            fig_clv_segment = px.box(customer_data, x='customer_segment', y='total_monetary_value', title='CLV Distribution by Segment')
            fig_clv_segment.update_layout(height=400)
            st.plotly_chart(fig_clv_segment, use_container_width=True)
            st.markdown("**CLV Insights**")
            st.info(f"**Average CLV**: R$ {customer_data['total_monetary_value'].mean():.2f}")
            st.info(f"**Median CLV**: R$ {customer_data['total_monetary_value'].median():.2f}")
            total_revenue = customer_data['total_monetary_value'].sum()
            customer_data_sorted = customer_data.sort_values('total_monetary_value', ascending=False)
            customer_data_sorted['cumulative_revenue'] = customer_data_sorted['total_monetary_value'].cumsum()
            customer_data_sorted['cumulative_percent'] = customer_data_sorted['cumulative_revenue'] / total_revenue * 100
            top_20_percent_customers = int(len(customer_data_sorted) * 0.2)
            revenue_from_top_20 = customer_data_sorted.iloc[:top_20_percent_customers]['cumulative_percent'].iloc[-1]
            st.success(f"**Pareto Principle**: Top 20% customers generate {revenue_from_top_20:.1f}% of revenue")

    with tab4:
        st.subheader("Churn Risk Analysis & Retention Strategies")
        customer_data_churn = customer_data.copy()
        customer_data_churn['churn_risk'] = pd.cut(
            customer_data_churn['recency_days'],
            bins=[0, 90, 180, 365, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        col1, col2 = st.columns(2)
        with col1:
            churn_dist = customer_data_churn['churn_risk'].value_counts()
            fig_churn = px.pie(values=churn_dist.values, names=churn_dist.index, title='Customer Churn Risk Distribution')
            st.plotly_chart(fig_churn, use_container_width=True)
            risk_by_segment = pd.crosstab(customer_data_churn['customer_segment'], customer_data_churn['churn_risk'], normalize='index') * 100
            fig_risk_segment = px.bar(risk_by_segment, title='Churn Risk by Customer Segment (%)')
            fig_risk_segment.update_layout(height=400)
            st.plotly_chart(fig_risk_segment, use_container_width=True)
        with col2:
            st.markdown("**Retention Strategy Recommendations**")
            high_risk_customers = len(customer_data_churn[customer_data_churn['churn_risk'].isin(['High', 'Critical'])])
            high_value_at_risk = len(customer_data_churn[
                (customer_data_churn['churn_risk'].isin(['High', 'Critical'])) &
                (customer_data_churn['total_monetary_value'] > customer_data_churn['total_monetary_value'].quantile(0.7))
            ])
            st.error(f"**Urgent Action Required**\n{high_risk_customers} customers at high churn risk")
            st.warning(f"**High-Value at Risk**\n{high_value_at_risk} valuable customers need immediate attention")
            st.markdown("**Recommended Retention Tactics**")
            critical_customers = len(customer_data_churn[customer_data_churn['churn_risk'] == 'Critical'])
            if critical_customers > 0:
                st.error(f"**Critical ({critical_customers} customers)**\n• Personal outreach calls\n• Exclusive win-back offers\n• Account manager assignment")
            high_only = len(customer_data_churn[customer_data_churn['churn_risk'] == 'High'])
            if high_only > 0:
                st.warning(f"**High Risk ({high_only} customers)**\n• Email re-engagement campaigns\n• Discount incentives\n• Product recommendations")
            medium_risk = len(customer_data_churn[customer_data_churn['churn_risk'] == 'Medium'])
            if medium_risk > 0:
                st.info(f"**Medium Risk ({medium_risk} customers)**\n• Newsletter engagement\n• Loyalty program enrollment\n• Seasonal promotions")
            churn_report = customer_data_churn[['customer_id', 'customer_segment', 'churn_risk',
                                               'total_monetary_value', 'recency_days', 'order_frequency']]
            csv_churn = churn_report.to_csv(index=False)
            st.download_button(
                label="Download Churn Risk Report",
                data=csv_churn,
                file_name=f'churn_risk_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )

def show_sales_analytics(datasets, model_artifacts):
    st.header("Sales Analytics & Forecasting")
    st.markdown("#### Advanced Time Series Analysis and Revenue Intelligence")

    orders_df = datasets['orders'].copy()
    items_df = datasets['order_items'].copy()
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])

    revenue_data = orders_df.merge(
        items_df.groupby('order_id').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index()
               .rename(columns={'price': 'total_revenue', 'freight_value': 'total_freight'}),
        on='order_id', how='left'
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Revenue Trends", "Sales Forecasting", "Product Analytics", "Performance Dashboard", "Advanced Insights"])

    # ---- Revenue Trends ----
    with tab1:
        st.subheader("Interactive Revenue Trend Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            analysis_level = st.selectbox("Analysis Granularity", ["Daily", "Weekly", "Monthly", "Quarterly"])
        with c2:
            date_range = st.date_input(
                "Date Range",
                value=[revenue_data['order_purchase_timestamp'].min().date(),
                       revenue_data['order_purchase_timestamp'].max().date()],
                min_value=revenue_data['order_purchase_timestamp'].min().date(),
                max_value=revenue_data['order_purchase_timestamp'].max().date()
            )
        with c3:
            metrics_to_show = st.multiselect("Metrics to Display", ["Revenue", "Orders", "AOV", "Freight"], default=["Revenue", "Orders"])

        if len(date_range) == 2:
            filtered_data = revenue_data[
                (revenue_data['order_purchase_timestamp'].dt.date >= date_range[0]) &
                (revenue_data['order_purchase_timestamp'].dt.date <= date_range[1])
            ]
        else:
            filtered_data = revenue_data

        if analysis_level == "Daily":
            time_grouper = filtered_data['order_purchase_timestamp'].dt.date
        elif analysis_level == "Weekly":
            time_grouper = filtered_data['order_purchase_timestamp'].dt.to_period('W')
        elif analysis_level == "Monthly":
            time_grouper = filtered_data['order_purchase_timestamp'].dt.to_period('M')
        else:
            time_grouper = filtered_data['order_purchase_timestamp'].dt.to_period('Q')

        trend_data = filtered_data.groupby(time_grouper).agg({
            'total_revenue': ['sum', 'count', 'mean'],
            'total_freight': 'sum'
        }).round(2)
        trend_data.columns = ['total_revenue', 'order_count', 'aov', 'total_freight']
        trend_data = trend_data.reset_index()
        trend_data['period'] = trend_data['order_purchase_timestamp'].astype(str)

        fig_trends = make_subplots(rows=2, cols=2,
                                   subplot_titles=['Revenue Trend', 'Order Volume', 'Average Order Value', 'Freight Analysis'],
                                   specs=[[{'secondary_y': False}, {'secondary_y': False}],
                                          [{'secondary_y': False}, {'secondary_y': False}]])
        if "Revenue" in metrics_to_show:
            fig_trends.add_trace(go.Scatter(x=trend_data['period'], y=trend_data['total_revenue'], mode='lines+markers', name='Revenue'), row=1, col=1)
        if "Orders" in metrics_to_show:
            fig_trends.add_trace(go.Scatter(x=trend_data['period'], y=trend_data['order_count'], mode='lines+markers', name='Orders'), row=1, col=2)
        if "AOV" in metrics_to_show:
            fig_trends.add_trace(go.Scatter(x=trend_data['period'], y=trend_data['aov'], mode='lines+markers', name='AOV'), row=2, col=1)
        if "Freight" in metrics_to_show:
            fig_trends.add_trace(go.Scatter(x=trend_data['period'], y=trend_data['total_freight'], mode='lines+markers', name='Freight'), row=2, col=2)
        fig_trends.update_layout(height=600, showlegend=False)
        fig_trends.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trends, use_container_width=True)

        st.markdown("**Trend Analysis Insights**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            revenue_growth = ((trend_data['total_revenue'].iloc[-1] / trend_data['total_revenue'].iloc[0]) - 1) * 100 if len(trend_data) > 1 else 0
            st.metric("Revenue Growth", f"{revenue_growth:.1f}%")
        with c2:
            order_growth = ((trend_data['order_count'].iloc[-1] / trend_data['order_count'].iloc[0]) - 1) * 100 if len(trend_data) > 1 else 0
            st.metric("Order Growth", f"{order_growth:.1f}%")
        with c3:
            st.metric("Average AOV", f"R$ {trend_data['aov'].mean():.2f}")
        with c4:
            peak_period = trend_data.loc[trend_data['total_revenue'].idxmax(), 'period'] if len(trend_data) else "-"
            st.metric("Peak Period", peak_period)

    # ---- Sales Forecasting ----
    with tab2:
        st.subheader("Advanced Sales Forecasting")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Forecasting Parameters**")
            forecast_periods = st.slider("Forecast Periods", 1, 12, 3)
            forecast_metric = st.selectbox("Metric to Forecast", ["Revenue", "Orders", "AOV"])
            confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
            include_seasonality = st.checkbox("Include Seasonality", value=True)
            include_trend = st.checkbox("Include Trend", value=True)
            if st.button("Generate Forecast", type="primary"):
                st.session_state.generate_forecast = True
        with c2:
            if st.session_state.get('generate_forecast', False):
                monthly_data = revenue_data.groupby(
                    revenue_data['order_purchase_timestamp'].dt.to_period('M')
                ).agg({'total_revenue': 'sum', 'order_id': 'count'}).reset_index()
                monthly_data['period'] = monthly_data['order_purchase_timestamp'].astype(str)
                monthly_data['aov'] = monthly_data['total_revenue'] / monthly_data['order_id']
                if forecast_metric == "Revenue":
                    historical_values = monthly_data['total_revenue'].values
                elif forecast_metric == "Orders":
                    historical_values = monthly_data['order_id'].values
                else:
                    historical_values = monthly_data['aov'].values

                alpha = 0.3
                forecast_values, last_value = [], historical_values[-1]
                for i in range(forecast_periods):
                    next_value = last_value
                    if include_trend and len(historical_values) > 1:
                        trend = np.mean(np.diff(historical_values[-3:]))
                        next_value = last_value + trend
                    if include_seasonality and len(historical_values) >= 12:
                        seasonal_factor = historical_values[-(12 - i % 12)] / np.mean(historical_values[-12:])
                        next_value *= seasonal_factor
                    forecast_values.append(next_value)
                    last_value = next_value

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=monthly_data['period'], y=historical_values, mode='lines+markers', name='Historical'))
                last_period = pd.Period(monthly_data['period'].iloc[-1])
                future_periods = [str(last_period + i) for i in range(1, forecast_periods + 1)]
                fig_forecast.add_trace(go.Scatter(x=future_periods, y=forecast_values, mode='lines+markers', name='Forecast', line=dict(dash='dash')))

                forecast_std = np.std(historical_values[-6:]) if len(historical_values) > 6 else np.std(historical_values)
                confidence_multiplier = 1.96 if confidence_level == 95 else 2.58
                upper_bound = [v + confidence_multiplier * forecast_std for v in forecast_values]
                lower_bound = [v - confidence_multiplier * forecast_std for v in forecast_values]
                fig_forecast.add_trace(go.Scatter(x=future_periods, y=upper_bound, fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
                fig_forecast.add_trace(go.Scatter(x=future_periods, y=lower_bound, fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', name=f'{confidence_level}% Confidence', fillcolor='rgba(255,0,0,0.2)'))
                fig_forecast.update_layout(title=f'{forecast_metric} Forecast - Next {forecast_periods} Periods', xaxis_title='Period', yaxis_title=forecast_metric, height=500)
                st.plotly_chart(fig_forecast, use_container_width=True)

                forecast_df = pd.DataFrame({'Period': future_periods, 'Forecasted_Value': forecast_values, 'Lower_Bound': lower_bound, 'Upper_Bound': upper_bound})
                st.markdown("**Forecast Summary**")
                st.dataframe(forecast_df.round(2), use_container_width=True)
                st.download_button(
                    label="Download Forecast Data",
                    data=forecast_df.to_csv(index=False),
                    file_name=f'sales_forecast_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

    # ---- Product Analytics ----
    with tab3:
        st.subheader("Product Performance Analytics")
        products_df = datasets['products'].copy()
        category_df = datasets['category_translation'].copy()
        items_df = datasets['order_items'].copy()
        product_sales = items_df.merge(products_df, on='product_id', how='left').merge(category_df, on='product_category_name', how='left')

        col1, col2 = st.columns(2)
        with col1:
            category_performance = product_sales.groupby('product_category_name_english').agg({'price': ['sum', 'count', 'mean']}).round(2)
            category_performance.columns = ['total_revenue', 'total_orders', 'avg_price']
            category_performance = category_performance.reset_index().sort_values('total_revenue', ascending=False)
            top_categories = category_performance.head(10)
            fig_top_categories = px.bar(top_categories, x='total_revenue', y='product_category_name_english',
                                        orientation='h', title='Top 10 Categories by Revenue',
                                        labels={'total_revenue': 'Revenue (R$)', 'product_category_name_english': 'Category'})
            fig_top_categories.update_layout(height=500)
            st.plotly_chart(fig_top_categories, use_container_width=True)
        with col2:
            fig_matrix = px.scatter(category_performance, x='total_orders', y='avg_price', size='total_revenue',
                                    hover_name='product_category_name_english',
                                    title='Category Performance Matrix',
                                    labels={'total_orders': 'Total Orders', 'avg_price': 'Average Price (R$)', 'total_revenue': 'Total Revenue'})
            fig_matrix.update_layout(height=500)
            st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("**Product Performance Insights**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if len(top_categories):
                st.metric("Top Category", (top_categories.iloc[0]['product_category_name_english'] or "")[:15] + "...")
            else:
                st.metric("Top Category", "-")
        with c2:
            st.metric("Active Categories", f"{len(category_performance)}")
        with c3:
            if len(category_performance):
                highest_aov_category = category_performance.loc[category_performance['avg_price'].idxmax()]
                st.metric("Highest AOV Category", f"R$ {highest_aov_category['avg_price']:.0f}")
            else:
                st.metric("Highest AOV Category", "-")
        with c4:
            st.metric("Total Products", f"{product_sales['product_id'].nunique():,}")

    # ---- Performance Dashboard ----
    with tab4:
        st.subheader("Real-time Performance Dashboard")
        total_revenue = revenue_data['total_revenue'].sum()
        total_orders = len(revenue_data)
        avg_aov = revenue_data['total_revenue'].mean()

        recent_date = revenue_data['order_purchase_timestamp'].max()
        last_30_days = revenue_data[revenue_data['order_purchase_timestamp'] >= (recent_date - timedelta(days=30))]
        previous_30_days = revenue_data[
            (revenue_data['order_purchase_timestamp'] >= (recent_date - timedelta(days=60))) &
            (revenue_data['order_purchase_timestamp'] < (recent_date - timedelta(days=30)))
        ]
        revenue_growth = ((last_30_days['total_revenue'].sum() / previous_30_days['total_revenue'].sum()) - 1) * 100 if len(previous_30_days) > 0 else 0
        order_growth = ((len(last_30_days) / len(previous_30_days)) - 1) * 100 if len(previous_30_days) > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Revenue", f"R$ {total_revenue:,.0f}", f"{revenue_growth:+.1f}% (30d)")
        with c2:
            st.metric("Total Orders", f"{total_orders:,}", f"{order_growth:+.1f}% (30d)")
        with c3:
            st.metric("Average AOV", f"R$ {avg_aov:.2f}")
        with c4:
            conversion_rate = (len(revenue_data[revenue_data['order_status'] == 'delivered']) / total_orders) * 100
            st.metric("Delivery Rate", f"{conversion_rate:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            revenue_target = total_revenue * 1.2
            revenue_progress = (total_revenue / revenue_target) * 100
            fig_revenue_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=revenue_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Revenue Target Progress (%)"},
                delta={'reference': 100},
                gauge={'axis': {'range': [None, 120]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                 {'range': [50, 80], 'color': "yellow"},
                                 {'range': [80, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}
            ))
            fig_revenue_gauge.update_layout(height=400)
            st.plotly_chart(fig_revenue_gauge, use_container_width=True)
        with c2:
            order_target = total_orders * 1.15
            order_progress = (total_orders / order_target) * 100
            fig_order_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=order_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Order Target Progress (%)"},
                delta={'reference': 100},
                gauge={
                    'axis': {'range': [None, 120]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}
                }
            ))
            fig_order_gauge.update_layout(height=400)
            st.plotly_chart(fig_order_gauge, use_container_width=True)

    # ---- Advanced Insights ----
    with tab5:
        st.subheader("Advanced Business Intelligence")

        # Cohort analysis (simplified acquisition by month)
        st.markdown("**Customer Cohort Analysis**")
        cohort_data = revenue_data.copy()
        cohort_data['order_month'] = cohort_data['order_purchase_timestamp'].dt.to_period('M')

        customer_cohorts = cohort_data.groupby(['customer_id', 'order_month']).size().reset_index()
        customer_acq = customer_cohorts.groupby('order_month').size().reset_index()
        customer_acq.columns = ['cohort_month', 'new_customers']
        customer_acq['cohort_month'] = customer_acq['cohort_month'].astype(str)

        fig_cohorts = px.bar(customer_acq, x='cohort_month', y='new_customers',
                             title='Customer Acquisition by Month')
        fig_cohorts.update_layout(height=400)
        fig_cohorts.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cohorts, use_container_width=True)

        # Revenue concentration (Pareto)
        st.markdown("**Revenue Concentration Analysis**")
        customer_revenue = revenue_data.groupby('customer_id')['total_revenue'].sum().sort_values(ascending=False)

        total_rev_all = customer_revenue.sum()
        total_customers = len(customer_revenue)
        cumulative_customers = np.arange(1, total_customers + 1)
        cumulative_revenue = customer_revenue.cumsum()
        revenue_percentage = (cumulative_revenue / total_rev_all) * 100 if total_rev_all > 0 else np.zeros_like(cumulative_revenue)
        customer_percentage = (cumulative_customers / total_customers) * 100 if total_customers > 0 else np.zeros_like(cumulative_customers)

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(x=customer_percentage, y=revenue_percentage,
                                        mode='lines', name='Revenue Concentration',
                                        line=dict(width=3)))
        fig_pareto.add_trace(go.Scatter(x=[0, 20, 100], y=[0, 80, 100],
                                        mode='lines', name='80-20 Reference',
                                        line=dict(dash='dash')))
        fig_pareto.update_layout(title='Customer Revenue Concentration (Pareto Analysis)',
                                 xaxis_title='Customer Percentage', yaxis_title='Revenue Percentage', height=500)
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Strategic insights
        st.markdown("**Strategic Business Insights**")
        top_20_n = int(max(1, np.floor(total_customers * 0.2)))
        revenue_from_top_20_percent = (customer_revenue.head(top_20_n).sum() / total_rev_all * 100) if total_rev_all > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Revenue Concentration**\nTop 20% of customers generate {revenue_from_top_20_percent:.1f}% of total revenue")
        with col2:
            monthly_rev = revenue_data.groupby(revenue_data['order_purchase_timestamp'].dt.month)['total_revenue'].sum()
            seasonal_coefficient = (monthly_rev.std() / monthly_rev.mean()) if monthly_rev.mean() > 0 else 0.0
            st.warning(f"**Seasonality Impact**\nRevenue variability coefficient: {seasonal_coefficient:.2f}")
        with col3:
            order_counts = revenue_data.groupby('customer_id').size()
            repeat_customers = (order_counts > 1).sum()
            repeat_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0.0
            st.error(f"**Customer Retention**\nRepeat customer rate: {repeat_rate:.1f}%")

def show_business_insights(datasets, model_artifacts):
    # ---------- Hero ----------
    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#e6f3ff, #f4f9ff);
        padding: 24px 28px; border-radius: 16px; border: 1px solid #d9e8ff;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:36px;">🎯</div>
        <div>
          <div style="font-size:32px;font-weight:800;color:#0f3d65;">Business Intelligence Insights</div>
          <div style="font-size:16px;color:#375e7a;opacity:.9;">
            Advanced E-commerce Analytics & Predictive Intelligence
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------- Derive quick KPIs from data ----------
    orders = datasets['orders'].copy()
    items  = datasets['order_items'].copy()
    products = datasets['products'].copy()
    cats = datasets['category_translation'].copy()

    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    revenue_df = orders.merge(
        items.groupby('order_id')
             .agg(price=('price','sum'), freight=('freight_value','sum'))
             .reset_index(),
        on='order_id', how='left'
    )
    total_rev = float(revenue_df['price'].sum())
    total_orders = len(revenue_df)
    delivered_rate = (revenue_df['order_status'].eq('delivered').mean() * 100.0) if total_orders else 0.0
    aov = float(revenue_df['price'].mean()) if total_orders else 0.0

    # Top category by revenue
    prod_sales = items.merge(products, on='product_id', how='left').merge(cats, on='product_category_name', how='left')
    top_cat = "-"
    if len(prod_sales):
        cat_rev = prod_sales.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False)
        if len(cat_rev):
            top_cat = cat_rev.index[0]

    # At-risk customers from model_artifacts (if available)
    at_risk = None
    if model_artifacts and 'customer_segments' in model_artifacts:
        seg = model_artifacts['customer_segments']
        if 'customer_segment' in seg:
            at_risk = int((seg['customer_segment'] == 'At Risk').sum())

    # ---------- KPI strip ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue", f"R$ {total_rev:,.0f}")
        st.caption("Sum of realized order value")
    with c2:
        st.metric("Delivery Rate", f"{delivered_rate:.1f}%")
        st.caption("Delivered / All orders")
    with c3:
        st.metric("Average Order Value", f"R$ {aov:,.2f}")
    with c4:
        st.metric("Top Category", top_cat if top_cat else "-")
        if at_risk is not None:
            st.caption(f"At-Risk Customers: {at_risk:,}")

    st.divider()

    # ---------- Impact vs Effort matrix ----------
    st.subheader("📌 Prioritization: Impact vs Effort")
    st.caption("Place initiatives where they belong—focus top-right first (high impact, low effort).")

    initiatives = pd.DataFrame([
        # name, impact(1-10), effort(1-10), note
        ["Accelerate cross-state logistics", 9, 7, "Predictive SLA routing + carrier SLAs"],
        ["VIP shipping for top-CLV",        7, 5, "Premium promise for top decile CLV"],
        ["Freight bundling nudges",         6, 3, "Lower freight-to-basket ratio"],
        ["Win-back (High/Critical churn)",  8, 4, "Triggered offers & journeys"],
    ], columns=["Initiative","Impact","Effort","Notes"])

    fig_ie = go.Figure()
    fig_ie.add_trace(go.Scatter(
        x=initiatives["Effort"], y=initiatives["Impact"],
        mode="markers+text", text=initiatives["Initiative"],
        textposition="top center", marker=dict(size=16)
    ))

    # Quadrant lines & labels
    fig_ie.add_shape(type="line", x0=5.5, x1=5.5, y0=1, y1=10, line=dict(dash="dot"))
    fig_ie.add_shape(type="line", x0=1, x1=10, y0=5.5, y1=5.5, line=dict(dash="dot"))
    fig_ie.add_annotation(x=3.2, y=8.8, text="Quick Wins", showarrow=False)
    fig_ie.add_annotation(x=8.0, y=8.8, text="Major Projects", showarrow=False)
    fig_ie.add_annotation(x=3.0, y=3.0, text="Fill-ins", showarrow=False)
    fig_ie.add_annotation(x=8.2, y=3.0, text="Reconsider", showarrow=False)

    fig_ie.update_layout(
        xaxis=dict(range=[1,10], title="Effort"),
        yaxis=dict(range=[1,10], title="Impact"),
        height=440, margin=dict(l=20,r=20,t=30,b=20)
    )
    st.plotly_chart(fig_ie, use_container_width=True)

    # ---------- Roadmap / Action plan ----------
    st.subheader("🛠️ 90-Day Action Plan")
    plan = pd.DataFrame([
        ["Freight bundling nudges",     "Product + Data", "Now–30d",  "Low",   "R$ ↑ / Freight% ↓"],
        ["Win-back journeys",           "CRM",            "Now–45d",  "Medium","Retention ↑"],
        ["VIP shipping (top 10% CLV)",  "Ops + CRM",      "30–60d",   "Medium","NPS ↑ / Repeat ↑"],
        ["Carrier SLA optimization",    "Ops + DS",       "45–90d",   "High",  "Late risk ↓"],
    ], columns=["Initiative","Owner","Window","Effort","Primary KPI"])
    st.dataframe(plan, use_container_width=True, hide_index=True)

    # ---------- Narrative bullets (polished copy) ----------
    st.subheader("💡 Executive Recommendations")
    st.markdown("""
- **Accelerate cross-state logistics:** Prioritize lanes with high predicted lateness; negotiate SLA tiers and dynamic carrier selection.
- **Protect high-value segments:** Offer **VIP shipping** and proactive comms for top-CLV customers during seasonal peaks.
- **Freight optimization:** Add **bundling nudges** at checkout when freight-to-basket ratio > threshold.
- **Retention focus:** Trigger **win-back** flows for customers with *High*/*Critical* churn risk; measure 30-/60-day reactivation.
    """)

    # ---------- Download one-pager ----------
    one_pager = initiatives.copy()
    one_pager["AOV"] = round(aov,2)
    one_pager["Delivered_Rate_%"] = round(delivered_rate,1)
    one_pager_csv = one_pager.to_csv(index=False)
    st.download_button(
        "📥 Download Insight One-Pager (CSV)",
        data=one_pager_csv,
        file_name=f"olist_business_insights_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # ---------- Quick navigation hints ----------
    st.info("Jump to **Delivery Prediction** to simulate lane changes, or **Customer Intelligence** to target VIPs & win-backs.")


# --------------------------------------------
# Entrypoint
# --------------------------------------------
if __name__ == "__main__":
    main()

