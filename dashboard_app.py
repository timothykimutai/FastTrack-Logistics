import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page
import base64

st.set_page_config(page_title="FastTrack Logistics Dashboard", layout="wide")
st.title("FastTrack Logistics Dashboard")
st.markdown("""
This dashboard provides key insights into delivery performance, delays, and cost optimization for FastTrack Logistics.
""")
st.info("Use the sidebar and tabs to explore delivery KPIs, trends, and recommendations. Download filtered data for further analysis.")

# Load data
data = pd.read_csv("logistics_cleaned_for_dashboard.csv")
if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
    # Simulate coordinates for regions (for demo purposes)
    region_coords = {
        'North': (40.5, -73.5),
        'South': (34.0, -84.4),
        'East': (42.3, -71.0),
        'West': (37.8, -122.4)
    }
    data['Latitude'] = data['Region'].map(lambda r: region_coords.get(r, (37.0, -95.0))[0])
    data['Longitude'] = data['Region'].map(lambda r: region_coords.get(r, (37.0, -95.0))[1])

# Sidebar filters
regions = sorted(data['Region'].unique())
couriers = sorted(data['Courier_Name'].unique())
region_filter = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
courier_filter = st.sidebar.multiselect("Select Courier(s)", couriers, default=couriers)
date_range = st.sidebar.date_input("Select Delivery Date Range", [pd.to_datetime(data['Delivery_Date']).min(), pd.to_datetime(data['Delivery_Date']).max()])
delay_slider = st.sidebar.slider("Delivery Delay (hr)", float(data['Delivery_Delay_hr'].min()), float(data['Delivery_Delay_hr'].max()), (float(data['Delivery_Delay_hr'].min()), float(data['Delivery_Delay_hr'].max())))

# Filter data
df = data.copy()
if region_filter:
    df = df[df['Region'].isin(region_filter)]
if courier_filter:
    df = df[df['Courier_Name'].isin(courier_filter)]
if isinstance(date_range, list) and len(date_range) == 2:
    df = df[(pd.to_datetime(df['Delivery_Date']) >= pd.to_datetime(date_range[0])) & (pd.to_datetime(df['Delivery_Date']) <= pd.to_datetime(date_range[1]))]
df = df[(df['Delivery_Delay_hr'] >= delay_slider[0]) & (df['Delivery_Delay_hr'] <= delay_slider[1])]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("On-Time Delivery %", f"{(df['Delivery_Status']=='On Time').mean()*100:.2f}%")
col2.metric("Avg Delivery Time (hr)", f"{df['Time_Taken_hr'].mean():.2f}")
col3.metric("Avg Fuel Cost per Order", f"${df['Fuel_Cost'].mean():.2f}")
col4.metric("Avg Delivery Delay (hr)", f"{df['Delivery_Delay_hr'].mean():.2f}")
st.markdown("---")
st.subheader("Download Filtered Data")
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_logistics_data.csv">Download CSV</a>'
    return href

def get_excel_download_link(df):
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Logistics')
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="filtered_logistics_data.xlsx">Download Excel</a>'
    return href
st.markdown(get_table_download_link(df), unsafe_allow_html=True)
st.markdown(get_excel_download_link(df), unsafe_allow_html=True)
# Show summary statistics
with st.expander("Show Summary Statistics for Filtered Data"):
    st.write(df.describe(include='all'))

# Advanced search box
search_term = st.sidebar.text_input("Search Order ID or Customer ID")
if search_term:
    df = df[df['Order_ID'].str.contains(search_term, case=False) | df['Customer_ID'].str.contains(search_term, case=False)]

st.markdown("---")
st.markdown("---")

# Tabs for analysis views
tab1, tab2, tab3, tab4 = st.tabs(["KPIs & Trends", "Map View", "Courier Analysis", "Recommendations"])

with tab1:
    st.markdown("---")
    st.subheader("Predictive Modeling: Delivery Delay Prediction")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import shap
    # Features for prediction
    features = ["Distance_km", "Fuel_Cost", "Time_Taken_hr"]
    X = df[features]
    y = df["Delivery_Delay_hr"]
    custom_pred_threshold = st.slider("Set Predicted Delay Alert Threshold (hr)", float(df['Delivery_Delay_hr'].min()), float(df['Delivery_Delay_hr'].max()), float(df['Delivery_Delay_hr'].quantile(0.95)))
    if len(df) > 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
        st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
        st.write("Feature Importances:")
        importances = pd.Series(model.feature_importances_, index=features)
        st.bar_chart(importances)
        # SHAP explanations
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            st.markdown("**Model Explanation (SHAP Summary Plot):**")
            import matplotlib.pyplot as plt
            fig_shap, ax_shap = plt.subplots()
            shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")
        # Predict for current filtered data
        df['Predicted_Delay'] = model.predict(X)
        st.dataframe(df[["Order_ID","Delivery_Delay_hr","Predicted_Delay"]])
        # Highlight predicted delays above threshold
        pred_outliers = df[df['Predicted_Delay'] > custom_pred_threshold]
        if not pred_outliers.empty:
            st.warning(f"{pred_outliers.shape[0]} orders predicted to exceed {custom_pred_threshold:.2f} hr delay.")
            st.dataframe(pred_outliers[["Order_ID","Customer_ID","Region","Courier_Name","Predicted_Delay"]].style.applymap(lambda v: 'background-color: #ffe6e6' if isinstance(v, float) and v > custom_pred_threshold else ''))
            st.markdown(get_table_download_link(pred_outliers), unsafe_allow_html=True)
    else:
        st.info("Not enough data for predictive modeling.")

    st.markdown("---")
    st.subheader("Anomaly Detection: Delivery Delay Outliers")
    from sklearn.ensemble import IsolationForest
    anomaly_contamination = st.slider("Set Anomaly Detection Sensitivity (Contamination %)", 1, 20, 5) / 100.0
    iso = IsolationForest(contamination=anomaly_contamination, random_state=42)
    anomaly_features = ["Delivery_Delay_hr", "Distance_km", "Fuel_Cost", "Time_Taken_hr"]
    df['anomaly'] = iso.fit_predict(df[anomaly_features])
    anomalies = df[df['anomaly'] == -1]
    st.write(f"Detected {anomalies.shape[0]} anomalous deliveries (contamination={anomaly_contamination*100:.1f}%).")
    st.dataframe(anomalies[["Order_ID","Customer_ID","Region","Courier_Name","Delivery_Delay_hr","Fuel_Cost"]])
    st.markdown(get_table_download_link(anomalies), unsafe_allow_html=True)
    st.markdown(get_excel_download_link(anomalies), unsafe_allow_html=True)
    st.subheader("Delivery KPIs & Trends")
    # Highlight outliers in delivery delay
    outlier_threshold = st.slider("Highlight Delivery Delay Outliers (hr)", float(df['Delivery_Delay_hr'].min()), float(df['Delivery_Delay_hr'].max()), float(df['Delivery_Delay_hr'].quantile(0.95)))
    outlier_df = df[df['Delivery_Delay_hr'] > outlier_threshold]
    if not outlier_df.empty:
        st.warning(f"{outlier_df.shape[0]} orders have delivery delay above {outlier_threshold:.2f} hr.")
        st.dataframe(outlier_df[['Order_ID','Customer_ID','Region','Courier_Name','Delivery_Delay_hr','Fuel_Cost']].style.applymap(lambda v: 'background-color: #ffcccc' if isinstance(v, float) and v > outlier_threshold else ''))
        # Export outliers
        st.markdown(get_table_download_link(outlier_df), unsafe_allow_html=True)
        st.markdown(get_excel_download_link(outlier_df), unsafe_allow_html=True)
        st.markdown(get_excel_download_link(outlier_df), unsafe_allow_html=True)
    chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Box"])
    group_by = st.selectbox("Group By", ["Region", "Courier_Name", "Delivery_Status"])
    metric = st.selectbox("Metric", ["On-Time %", "Avg Delivery Time", "Avg Delay", "Avg Fuel Cost"])
    if metric == "On-Time %":
        agg_df = df.groupby(group_by)['Delivery_Status'].apply(lambda x: (x=='On Time').mean()*100).reset_index(name='On-Time %')
        y_col = 'On-Time %'
    elif metric == "Avg Delivery Time":
        agg_df = df.groupby(group_by)['Time_Taken_hr'].mean().reset_index(name='Avg Delivery Time')
        y_col = 'Avg Delivery Time'
    elif metric == "Avg Delay":
        agg_df = df.groupby(group_by)['Delivery_Delay_hr'].mean().reset_index(name='Avg Delay')
        y_col = 'Avg Delay'
    else:
        agg_df = df.groupby(group_by)['Fuel_Cost'].mean().reset_index(name='Avg Fuel Cost')
        y_col = 'Avg Fuel Cost'
    if chart_type == "Bar":
        fig = px.bar(agg_df, x=group_by, y=y_col, color=y_col, title=f'{y_col} by {group_by}')
    elif chart_type == "Line":
        fig = px.line(agg_df, x=group_by, y=y_col, title=f'{y_col} by {group_by}')
    elif chart_type == "Scatter":
        fig = px.scatter(agg_df, x=group_by, y=y_col, color=y_col, title=f'{y_col} by {group_by}')
    else:
        fig = px.box(df, x=group_by, y=y_col, title=f'{y_col} by {group_by}')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap (Numeric Features)")
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df[["Distance_km","Fuel_Cost","Time_Taken_hr","Delivery_Delay_hr"]].corr()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig_corr)

    st.markdown("---")
    st.subheader("Time Series Decomposition (Delivery Delay)")
    from statsmodels.tsa.seasonal import seasonal_decompose
    ts = df.groupby(pd.to_datetime(df['Delivery_Date']).dt.date)['Delivery_Delay_hr'].mean()
    if len(ts) > 10:
        result = seasonal_decompose(ts, model='additive', period=7)
        fig_ts, axs = plt.subplots(4, 1, figsize=(10,8))
        axs[0].plot(result.observed); axs[0].set_title('Observed')
        axs[1].plot(result.trend); axs[1].set_title('Trend')
        axs[2].plot(result.seasonal); axs[2].set_title('Seasonal')
        axs[3].plot(result.resid); axs[3].set_title('Residual')
        plt.tight_layout()
        st.pyplot(fig_ts)
    else:
        st.info("Not enough data for time series decomposition.")

with tab2:
    st.subheader("Region Performance Map")
    map_df = df.groupby(['Region', 'Latitude', 'Longitude']).agg({
        'Order_ID': 'count',
        'Delivery_Delay_hr': 'mean',
        'Fuel_Cost': 'sum',
        'Delivery_Status': lambda x: (x=='On Time').mean()*100
    }).reset_index().rename(columns={'Order_ID':'Total Orders','Delivery_Status':'On-Time %'})
    fig_map = px.scatter_mapbox(
        map_df, lat="Latitude", lon="Longitude", size="Total Orders", color="On-Time %",
        hover_name="Region", hover_data=["Total Orders", "On-Time %", "Delivery_Delay_hr", "Fuel_Cost"],
        color_continuous_scale="Viridis", size_max=40, zoom=3, mapbox_style="carto-positron",
        title="Region Performance Map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Courier Performance Comparison")
    courier_df = df.groupby('Courier_Name').agg({
        'Order_ID': 'count',
        'Delivery_Delay_hr': 'mean',
        'Fuel_Cost': 'sum',
        'Delivery_Status': lambda x: (x=='On Time').mean()*100
    }).reset_index().rename(columns={'Order_ID':'Total Orders','Delivery_Status':'On-Time %'})
    fig_courier = px.bar(
        courier_df, x='Courier_Name', y='On-Time %', color='On-Time %',
        title='On-Time % by Courier', labels={'On-Time %':'On-Time %'}
    )
    st.plotly_chart(fig_courier, use_container_width=True)
    st.dataframe(courier_df)
    st.markdown("---")
    st.subheader("Interactive Data Table")
    st.dataframe(df.style.applymap(lambda v: 'background-color: #e6f7ff' if isinstance(v, float) and v == df['Delivery_Delay_hr'].max() else ''))

with tab4:
    st.subheader("Recommendations & Insights")
    st.markdown("---")
    st.subheader("User Feedback & Suggestions")
    feedback = st.text_area("Share your feedback or suggestions to improve this dashboard:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
    st.markdown("""
    **Key Insights:**
    - Regions with lower on-time % may benefit from route optimization or courier reassignment.
    - Couriers with higher average delays should be reviewed for operational improvements.
    - Fuel cost outliers may indicate inefficient routes or vehicle issues.
    - Use advanced filters to identify specific patterns and opportunities for cost savings.
    """)
    st.markdown("---")
    st.markdown("**Business Recommendations:**")
    st.markdown("""
    - Reassign deliveries from couriers with high average delays to better performers.
    - Optimize routes in regions with high delays.
    - Investigate fuel cost outliers for potential savings.
    - Monitor KPIs regularly and automate reporting for continuous improvement.
    """)
