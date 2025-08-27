# 🛒 Olist E-Commerce Analytics Platform

> **Advanced Business Intelligence & Predictive Analytics Dashboard**

A comprehensive data science and business analytics platform built with **Machine Learning predictions**, **interactive dashboards**, and **strategic business insights** for e-commerce optimization.

## 🎯 **Project Overview**

This platform transforms raw e-commerce data into actionable business intelligence through:
- **ML-powered delivery predictions** with 13.6% improvement over business rules
- **Customer segmentation & lifetime value analysis** using RFM methodology  
- **Sales forecasting** with confidence intervals and seasonality analysis
- **Strategic business recommendations** with impact-effort prioritization
- **Interactive visualizations** and real-time analytics

**🔗 Live Demo:** [Streamlit Cloud Deployment](your-deployment-url)

## 📊 **Key Features & Capabilities**

### 1. **Executive Dashboard**
- **Real-time KPIs:** Revenue, orders, AOV, delivery rates
- **Trend analysis** with interactive time-series charts
- **Customer segmentation overview** with revenue attribution
- **Business health scorecard** with performance indicators

### 2. **Advanced Delivery Prediction**
- **🎯 Single Order Predictions**
  - ML model predicting delivery times (MAE: 4.5 days)
  - Late delivery risk assessment with confidence intervals
  - Real-time prediction with interactive parameter selection
  - Geographic impact analysis (same-state vs cross-state)

- **📊 Batch Processing**
  - Upload CSV files for bulk order analysis
  - Export results with risk classifications
  - Operational alerts for high-risk shipments

- **🔄 Scenario Comparison**
  - Side-by-side analysis of different shipping configurations
  - Template-based quick comparisons
  - Cost-benefit optimization recommendations

- **🗺️ Geographic Intelligence**
  - Interactive Brazil map with delivery performance
  - State-to-state delivery time matrix
  - Regional logistics optimization insights

- **📈 Historical Performance**
  - Delivery trends with seasonal decomposition
  - Performance benchmarking by time periods
  - Continuous improvement tracking

### 3. **Customer Intelligence & Analytics**
- **RFM Segmentation**
  - 7 distinct customer segments (Champions, At-Risk, etc.)
  - Interactive 3D RFM analysis with filtering
  - Revenue concentration and Pareto analysis

- **Lifetime Value Prediction**
  - CLV distribution analysis by segment
  - Top customer identification and profiling
  - Revenue impact quantification

- **Churn Risk Analysis**
  - 4-tier risk classification (Low/Medium/High/Critical)
  - Targeted retention strategy recommendations
  - Exportable action plans for customer success teams

### 4. **Sales Analytics & Forecasting**
- **Advanced Time Series Analysis**
  - Multi-granularity trends (daily/weekly/monthly/quarterly)
  - Interactive date filtering and metric selection
  - Growth rate calculations and period comparisons

- **ML-Based Sales Forecasting**
  - Exponential smoothing with trend and seasonality
  - Configurable confidence intervals (80-99%)
  - Downloadable forecast reports

- **Product Performance Analytics**
  - Category revenue ranking and performance matrix
  - Product portfolio optimization insights
  - Market share analysis by category

- **Real-Time Performance Dashboard**
  - Interactive KPI gauges with target tracking
  - 30-day growth comparisons
  - Operational performance monitoring

### 5. **Strategic Business Intelligence**
- **Impact vs Effort Prioritization Matrix**
  - Initiative evaluation with business impact scoring
  - Resource allocation optimization
  - Strategic roadmap development

- **90-Day Action Plan**
  - Prioritized recommendations with timelines
  - Owner assignment and effort estimation
  - KPI tracking for initiative success

- **Executive Insights & Recommendations**
  - Data-driven business strategy suggestions
  - ROI-focused improvement opportunities
  - Exportable strategic reports

## 🔧 **Technical Architecture**

### **Machine Learning Models**
- **Delivery Prediction:** HistGradientBoosting Regressor (MAE: 4.5 days)
- **Late Risk Classification:** Balanced classifier with threshold tuning
- **Feature Engineering:** Geographic, temporal, and order characteristics
- **Model Performance:** 13.6% improvement over rule-based predictions

### **Technology Stack**
```python
Frontend:    Streamlit (Interactive UI)
Backend:     Python, Pandas, NumPy
ML/AI:       Scikit-learn, HistGradientBoosting
Visualization: Plotly, Folium (Interactive maps)
Analytics:   Statistical modeling, Time series analysis
Data Processing: 100k+ orders, 9 datasets, RFM analysis
```

### **Data Pipeline**
1. **Data Ingestion:** 9 CSV datasets (orders, customers, products, etc.)
2. **Feature Engineering:** Geographic encoding, temporal features, RFM scoring
3. **Model Training:** Time-based validation, hyperparameter optimization
4. **Real-time Prediction:** Streamlit integration with cached models
5. **Business Intelligence:** Automated insights and recommendations

## 🚀 **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- 8GB RAM recommended
- Git

### **Quick Start**
```bash
# Clone repository
git clone https://github.com/yourusername/olist-analytics-platform.git
cd olist-analytics-platform

# Create virtual environment
python -m venv olist_env
source olist_env/bin/activate  # On Windows: olist_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run main.py
```

### **Data Setup**
1. Download Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Extract CSV files to `data/` directory
3. Run Colab notebook to generate `models/olist_model_artifacts.pkl`
4. Launch application

## 📈 **Business Impact & Results**

### **Model Performance**
- **Delivery Prediction Accuracy:** MAE 4.5 days (13.6% improvement)
- **Risk Classification:** AUC 0.526 with tunable thresholds
- **Geographic Insights:** 7.2-day difference between same-state vs cross-state
- **Customer Segmentation:** 7 distinct segments with clear business value

### **Business Value Delivered**
- **Operational Efficiency:** Proactive late-delivery alerts
- **Customer Experience:** Accurate delivery expectations
- **Revenue Optimization:** Customer segment targeting strategies  
- **Cost Reduction:** Logistics optimization recommendations
- **Strategic Planning:** Data-driven business decisions

### **Key Findings**
- 36% of orders are same-state (faster delivery)
- 8.1% late delivery rate with predictable risk factors
- Revenue concentration: Top 20% customers generate 80%+ of revenue
- Clear seasonal patterns with optimization opportunities

## 🎯 **Use Cases & Applications**

### **For E-commerce Operations**
- **Delivery Promise Accuracy:** Set realistic customer expectations
- **Inventory Planning:** Regional demand forecasting
- **Carrier Optimization:** Route and SLA management
- **Customer Service:** Proactive issue resolution

### **For Business Strategy**
- **Market Expansion:** Geographic opportunity analysis
- **Customer Retention:** Targeted engagement campaigns
- **Product Strategy:** Category performance optimization
- **Resource Allocation:** Data-driven investment decisions

## 📋 **Project Structure**
```
olist-analytics-platform/
├── data/                          # CSV datasets (9 files)
├── models/                        # Trained ML models
│   └── olist_model_artifacts.pkl
├── notebooks/                     # Jupyter analysis notebooks
├── pages/                         # Streamlit page components
├── utils/                         # Helper functions
├── main.py                        # Main Streamlit application
├── requirements.txt               # Python dependencies  
└── README.md                      # This file
```

## 🔄 **Future Enhancements**

### **Technical Roadmap**
- [ ] **Real-time Model Updates:** Automated retraining pipeline
- [ ] **Advanced ML Models:** Deep learning for complex patterns
- [ ] **API Development:** RESTful endpoints for integration
- [ ] **Database Integration:** PostgreSQL/MongoDB for scalability

### **Business Features**
- [ ] **A/B Testing Framework:** Experiment management system
- [ ] **Alert System:** Automated notifications and reports
- [ ] **Mobile Optimization:** Responsive design improvements
- [ ] **Multi-language Support:** Portuguese/Spanish localization

## 👨‍💻 **About the Developer**

**Data Scientist & Business Analyst** with expertise in:
- Machine Learning & Predictive Analytics
- Business Intelligence & Strategic Planning  
- E-commerce Analytics & Customer Intelligence
- Interactive Dashboard Development

**Technical Skills:** Python, ML/AI, Statistical Analysis, Business Strategy, Data Visualization

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 **Contact**

- **LinkedIn:** [Your LinkedIn Profile]
- **Email:** your.email@example.com
- **Portfolio:** [Your Portfolio Website]

---

⭐ **Star this repository if you found it helpful!**
