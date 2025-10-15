import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

#Load the trained model
model=load_model("Ann model.h5")  

st.title("ANN CSAT Prediction")

#Categorical Inputs
channel_name=st.selectbox("Channel Name",["Email","Chat","Phone"])
category=st.selectbox("Category",["Delivery","Product","Payment"])
sub_category=st.selectbox("Sub-category",["Delay","Damaged","Other"])
agent_name=st.selectbox("Agent Name",["Agent1","Agent2","Agent3"])
supervisor=st.selectbox("Supervisor",["Sup1","Sup2"])
manager=st.selectbox("Manager",["Mgr1","Mgr2"])
tenure_bucket=st.selectbox("Tenure Bucket",["0-1yr","1-3yr","3-5yr","5+yr"])
agent_shift=st.selectbox("Agent Shift",["Morning","Evening","Night"])

#Numeric Inputs
issue_reported_at=st.number_input("Issue Reported At (timestamp)")
issue_responded=st.number_input("Issue Responded (timestamp)")
item_price=st.number_input("Item Price")
connected_handling_time=st.number_input("Connected Handling Time")
response_year=st.number_input("Response Year")
response_month=st.number_input("Response Month")
response_day=st.number_input("Response Day")
response_weekday=st.number_input("Response Weekday")
response_week=st.number_input("Response Week")

#label encoding
cat_mapping = {
    "channel_name":{"Email":0,"Chat":1,"Phone":2},
    "category":{"Delivery":0,"Product":1,"Payment":2},
    "sub_category":{"Delay":0,"Damaged":1,"Other":2},
    "agent_name":{"Agent1":0,"Agent2":1,"Agent3":2},
    "supervisor":{"Sup1":0,"Sup2":1},
    "manager":{"Mgr1":0,"Mgr2":1},
    "tenure_bucket":{"0-1yr":0,"1-3yr":1,"3-5yr":2,"5+yr":3},
    "agent_shift":{"Morning":0,"Evening":1,"Night":2}
}

#Map categorical values
channel_name=cat_mapping["channel_name"][channel_name]
category=cat_mapping["category"][category]
sub_category=cat_mapping["sub_category"][sub_category]
agent_name=cat_mapping["agent_name"][agent_name]
supervisor=cat_mapping["supervisor"][supervisor]
manager=cat_mapping["manager"][manager]
tenure_bucket=cat_mapping["tenure_bucket"][tenure_bucket]
agent_shift=cat_mapping["agent_shift"][agent_shift]

#Button to predict
if st.button("Predict CSAT"):
    input_data=np.array([[channel_name, category, sub_category,
        issue_reported_at,issue_responded, item_price,connected_handling_time,agent_name,supervisor,manager,tenure_bucket,
        agent_shift,response_year,response_month,response_day,response_weekday,response_week]])
    
    #Predict
    prediction=model.predict(input_data)
    st.success(f"Predicted CSAT Value: {prediction[0][0]:.4f}")
