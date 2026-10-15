import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time, timedelta

st.set_page_config(page_title='VoltWise | Energy Intelligence', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
sns.set_theme(style='whitegrid', context='notebook', palette='deep')

# Professional dashboard styling
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,0.92); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #101c35 0%, #172b4d 100%); }
    [data-testid="stSidebar"] * { color: #eef4ff !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { background: rgba(255,255,255,0.06); padding: 9px 12px; border-radius: 9px; margin-bottom: 5px; }
    .hero { padding: 26px 30px; border-radius: 18px; background: linear-gradient(115deg,#122344 0%,#1e4d86 65%,#2875a9 100%); color: white; margin: 4px 0 22px 0; box-shadow: 0 8px 26px rgba(18,35,68,.12); }
    .hero h1 { color: white; font-size: 2.05rem; margin: 0 0 8px 0; font-weight: 750; }
    .hero p { color: #dbeafe; font-size: 1rem; margin: 0; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e4eaf3; padding: 17px 18px; border-radius: 14px; box-shadow: 0 3px 12px rgba(26,45,75,.045); }
    div[data-testid="stMetricLabel"] { color: #5b6b82; font-size: .88rem; }
    div[data-testid="stMetricValue"] { color: #142b4d; font-weight: 750; }
    div[data-testid="stTabs"] button { border-radius: 10px 10px 0 0; font-weight: 650; }
    div[data-testid="stForm"] { background: white; padding: 20px; border: 1px solid #e4eaf3; border-radius: 14px; }
    div[data-testid="stDataFrame"] { background: white; border-radius: 12px; }
    .section-note { color: #61718a; font-size: .92rem; margin-top: -8px; margin-bottom: 14px; }
    .small-pill { display:inline-block; background:#e6f4ee; color:#16704a; border-radius:20px; padding:5px 10px; font-size:.78rem; font-weight:700; }
    hr { border-color: #e3eaf4; }
</style>
<div class="hero">
  <h1>⚡ VoltWise Energy Intelligence</h1>
  <p>Electricity consumption analytics, usage patterns and 24-hour forecasting — all in one dashboard.</p>
</div>
""", unsafe_allow_html=True)
st.caption('Monitor readings • Explore usage patterns • Estimate future demand')
with st.expander('About this application and its data', expanded=False):
    st.write('Enter your own meter readings or use clearly labelled synthetic demonstration data. Forecasts are estimates based on the readings available; actual live household data requires a compatible meter or utility API.')

@st.cache_data
def make_demo(days=30):
    rng=np.random.default_rng(24)
    dates=pd.date_range(end=pd.Timestamp.now().floor('h'), periods=days*24, freq='h')
    hour=dates.hour.to_numpy(); dow=dates.dayofweek.to_numpy(); weekend=(dow>=5).astype(int)
    temp=29+4*np.sin(2*np.pi*(hour-7)/24)+rng.normal(0,1.4,len(dates))
    cons=1.4+1.1*np.exp(-((hour-8)/3.0)**2)+2.1*np.exp(-((hour-20)/3.2)**2)+0.12*np.maximum(temp-27,0)+0.2*weekend+rng.normal(0,.18,len(dates))
    return pd.DataFrame({'DateTime':dates,'Consumption':np.maximum(cons,.05).round(3),'Temperature':temp.round(1)})

def enrich(df):
    df=df.copy(); df['DateTime']=pd.to_datetime(df['DateTime'],errors='coerce')
    df['Consumption']=pd.to_numeric(df['Consumption'],errors='coerce')
    df['Temperature']=pd.to_numeric(df['Temperature'],errors='coerce')
    df=df.dropna(subset=['DateTime','Consumption']); df=df[df.Consumption>=0]
    df=df.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    if df.empty: raise ValueError('Enter at least one valid reading with non-negative consumption.')
    df['Hour']=df.DateTime.dt.hour; df['DayOfWeek']=df.DateTime.dt.dayofweek
    df['DateOnly']=df.DateTime.dt.date; df['Month']=df.DateTime.dt.to_period('M').astype(str)
    df['Weekend']=(df.DayOfWeek>=5).astype(int)
    return df

def regression_fit(frame, cols):
    x=frame[cols].to_numpy(float); y=frame.Consumption.to_numpy(float)
    means=np.nanmean(x,axis=0); means=np.where(np.isfinite(means),means,0)
    x=np.where(np.isfinite(x),x,means); scales=x.std(axis=0); scales=np.where(scales==0,1,scales)
    coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),(x-means)/scales]),y,rcond=None)[0]
    return coef,means,scales

def regression_predict(frame, cols, fitted):
    coef,means,scales=fitted; x=frame[cols].to_numpy(float); x=np.where(np.isfinite(x),x,means)
    return np.maximum(np.column_stack([np.ones(len(x)),(x-means)/scales])@coef,0)

if 'manual_readings' not in st.session_state:
    st.session_state.manual_readings=[]

with st.sidebar:
    st.markdown('## ⚡ VoltWise')
    st.caption('ENERGY ANALYTICS PLATFORM')
    st.divider()
    st.header('Data workspace')
    source=st.radio('Data source',['Enter readings manually','Use demo data'])
    st.divider()
    st.caption('Suggested unit: kWh consumed during each interval. Keep units consistent.')

if source=='Enter readings manually':
    st.subheader('Add a meter reading')
    st.markdown('<p class="section-note">Enter a reading from your meter or bill. Add readings at different times to build useful trends and forecasts.</p>', unsafe_allow_html=True)
    with st.form('reading_form', clear_on_submit=True):
        col1,col2,col3=st.columns(3)
        with col1: reading_date=st.date_input('Date',value=date.today())
        with col2: reading_time=st.time_input('Time',value=time(datetime.now().hour,0))
        with col3: consumption=st.number_input('Consumption (kWh)',min_value=0.0,value=1.0,step=0.1,help='Enter energy used in this interval.')
        temperature=st.number_input('Temperature (°C)',value=29.0,step=0.5)
        add=st.form_submit_button('Add reading',type='primary')
        if add:
            dt=datetime.combine(reading_date,reading_time)
            if any(r['DateTime']==dt for r in st.session_state.manual_readings):
                st.warning('A reading for this exact date and time already exists.')
            else:
                st.session_state.manual_readings.append({'DateTime':dt,'Consumption':float(consumption),'Temperature':float(temperature)})
                st.success('Reading added.')
    c1,c2=st.columns([1,1])
    with c1:
        if st.button('Add 7 days of example readings'):
            now=pd.Timestamp.now().floor('h'); rng=np.random.default_rng(99)
            for dt in pd.date_range(end=now,periods=7*24,freq='h'):
                if not any(r['DateTime']==dt.to_pydatetime() for r in st.session_state.manual_readings):
                    h=dt.hour; temp=29+3*np.sin(2*np.pi*(h-7)/24)+rng.normal(0,.8)
                    val=max(.1,1.4+1.2*np.exp(-((h-8)/3)**2)+2*np.exp(-((h-20)/3)**2)+.1*max(temp-27,0)+rng.normal(0,.15))
                    st.session_state.manual_readings.append({'DateTime':dt.to_pydatetime(),'Consumption':round(float(val),3),'Temperature':round(float(temp),1)})
            st.rerun()
    with c2:
        if st.button('Clear entered readings'):
            st.session_state.manual_readings=[]; st.rerun()
    if st.session_state.manual_readings:
        st.caption(f"Readings entered this session: {len(st.session_state.manual_readings)}")
        st.dataframe(pd.DataFrame(st.session_state.manual_readings).sort_values('DateTime'),use_container_width=True,hide_index=True)
        data=enrich(pd.DataFrame(st.session_state.manual_readings))
        source_label='Manually entered readings'
    else:
        st.warning('No readings entered yet. Add readings above, or click “Add 7 days of example readings” to test every chart. Readings are kept for this browser session.')
        st.stop()
else:
    data=enrich(make_demo())
    source_label='Synthetic demo data (not actual meter readings)'
    st.success('Demo data loaded. Switch to “Enter readings manually” in the sidebar to enter your own readings.')

st.markdown(f'<span class="small-pill">● ACTIVE DATA</span> &nbsp; **{source_label}**', unsafe_allow_html=True)
if len(data)<48: st.warning('You have fewer than 48 readings. Charts work, but forecasting accuracy may be limited. Add at least 48 readings for a basic holdout evaluation; longer history is better.')
latest=data.iloc[-1]
a,b,c,d=st.columns(4)
a.metric('Total consumption',f"{data.Consumption.sum():,.2f} kWh")
b.metric('Average per reading',f"{data.Consumption.mean():,.2f} kWh")
c.metric('Peak reading',f"{data.Consumption.max():,.2f} kWh")
d.metric('Latest reading',f"{latest.Consumption:.2f} kWh")

overview,analytics,forecast,records=st.tabs(['▦  Overview','⌁  Analytics','◷  Forecast','▤  Readings & export'])
with overview:
    st.subheader('Consumption over time')
    st.markdown('<p class="section-note">Track changes in energy use and identify spikes across the selected period.</p>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(12,4)); ax.plot(data.DateTime,data.Consumption,marker='.' if len(data)<100 else None,linewidth=1.5); ax.set_xlabel('Date and time'); ax.set_ylabel('Consumption (kWh)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    left,right=st.columns(2)
    daily=data.groupby('DateOnly',as_index=False).Consumption.sum()
    with left:
        st.subheader('Daily total consumption')
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(pd.to_datetime(daily.DateOnly),daily.Consumption,marker='o'); ax.set_ylabel('Daily total (kWh)'); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader('Average by hour')
        hourly=data.groupby('Hour',as_index=False).Consumption.mean()
        fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=hourly,x='Hour',y='Consumption',ax=ax); ax.set_xlabel('Hour of day'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
with analytics:
    st.subheader('Consumption analytics')
    st.markdown('<p class="section-note">Explore distribution, temperature relationships, weekday patterns and monthly demand.</p>', unsafe_allow_html=True)
    stats=data.Consumption.describe().to_frame('Consumption (kWh)'); stats.loc['median']=data.Consumption.median(); stats.loc['variance']=data.Consumption.var(); stats.loc['standard deviation']=data.Consumption.std()
    st.dataframe(stats.round(3),use_container_width=True)
    left,right=st.columns(2)
    with left:
        st.subheader('Histogram — usage distribution')
        fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(data=data,x='Consumption',kde=len(data)>2,ax=ax); ax.set_xlabel('Consumption (kWh)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader('Scatter plot — temperature vs consumption')
        fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=data,x='Temperature',y='Consumption',ax=ax); ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Consumption (kWh)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    left,right=st.columns(2)
    with left:
        st.subheader('Average consumption by weekday')
        weekday=data.groupby('DayOfWeek',as_index=False).Consumption.mean()
        fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=weekday,x='DayOfWeek',y='Consumption',ax=ax); ax.set_xticks(range(7),['Mon','Tue','Wed','Thu','Fri','Sat','Sun']); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader('Monthly consumption')
        monthly=data.groupby('Month',as_index=False).Consumption.sum()
        fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=monthly,x='Month',y='Consumption',ax=ax); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader('Correlation heatmap')
    corr=data[['Consumption','Temperature','Hour','DayOfWeek','Weekend']].corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,5)); sns.heatmap(corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1,fmt='.2f',ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Consumption–temperature correlation:',round(data.Temperature.corr(data.Consumption),3) if data.Temperature.nunique()>1 else 'Not enough temperature variation to calculate correlation.')
with forecast:
    st.subheader('24-hour consumption forecast')
    st.markdown('<p class="section-note">Forecast estimates are generated from the available history and evaluated on a chronological holdout set.</p>', unsafe_allow_html=True)
    if len(data)<48:
        st.warning('For this basic model, add at least 48 hourly readings. Click “Add 7 days of example readings” for a demonstration, or keep entering your actual meter readings over time.')
    else:
        model=data.copy(); med=model.Temperature.median(); med=float(med) if np.isfinite(med) else 25.0
        model['TempFeature']=model.Temperature.fillna(med); model['HourSin']=np.sin(2*np.pi*model.Hour/24); model['HourCos']=np.cos(2*np.pi*model.Hour/24); model['DaySin']=np.sin(2*np.pi*model.DayOfWeek/7); model['DayCos']=np.cos(2*np.pi*model.DayOfWeek/7); model['Trend']=np.arange(len(model))
        cols=['TempFeature','HourSin','HourCos','DaySin','DayCos','Weekend','Trend']; split=max(1,min(int(len(model)*.8),len(model)-1)); train,test=model.iloc[:split],model.iloc[split:]
        fitted=regression_fit(train,cols); pred=regression_predict(test,cols,fitted); actual=test.Consumption.to_numpy(); mae=float(np.mean(np.abs(actual-pred))); rmse=float(np.sqrt(np.mean((actual-pred)**2)))
        x,y,z=st.columns(3); x.metric('Test MAE',f'{mae:.3f} kWh'); y.metric('Test RMSE',f'{rmse:.3f} kWh'); z.metric('Test records',str(len(test)))
        st.caption('Metrics are calculated on the last 20% of the readings, held out from training. They measure historical test error, not a guarantee of future accuracy.')
        n=min(168,len(test)); fig,ax=plt.subplots(figsize=(12,4)); ax.plot(test.DateTime.iloc[:n],actual[:n],label='Actual'); ax.plot(test.DateTime.iloc[:n],pred[:n],label='Predicted'); ax.set_ylabel('Consumption (kWh)'); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        future_dates=pd.date_range(data.DateTime.max()+pd.Timedelta(hours=1),periods=24,freq='h'); future=pd.DataFrame({'DateTime':future_dates}); future['Hour']=future.DateTime.dt.hour; future['DayOfWeek']=future.DateTime.dt.dayofweek; future['Weekend']=(future.DayOfWeek>=5).astype(int); future['HourSin']=np.sin(2*np.pi*future.Hour/24); future['HourCos']=np.cos(2*np.pi*future.Hour/24); future['DaySin']=np.sin(2*np.pi*future.DayOfWeek/7); future['DayCos']=np.cos(2*np.pi*future.DayOfWeek/7); future['Trend']=np.arange(len(model),len(model)+24); future['TempFeature']=med
        future['PredictedConsumption']=regression_predict(future,cols,fitted)
        fig,ax=plt.subplots(figsize=(12,4)); ax.plot(future.DateTime,future.PredictedConsumption,marker='o'); ax.set_title('Next 24 hours — estimated consumption'); ax.set_ylabel('Predicted kWh'); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(future[['DateTime','PredictedConsumption']].round(3),use_container_width=True,hide_index=True)
        st.download_button('Download 24-hour forecast CSV',future[['DateTime','PredictedConsumption']].to_csv(index=False).encode('utf-8'),'electricity_forecast_24h.csv','text/csv')
with records:
    st.subheader('Reading history')
    st.markdown('<p class="section-note">Review the readings used for analysis and optionally export them for your records.</p>', unsafe_allow_html=True)
    st.dataframe(data[['DateTime','Consumption','Temperature']].sort_values('DateTime',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download readings CSV',data[['DateTime','Consumption','Temperature']].to_csv(index=False).encode('utf-8'),'electricity_readings.csv','text/csv')
    st.caption('CSV export is optional; you do not need a CSV to use the dashboard.')
