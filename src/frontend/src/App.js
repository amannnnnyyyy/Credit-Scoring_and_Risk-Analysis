import React, { useState } from 'react';
import "react-datepicker/dist/react-datepicker.css";
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import DatePicker from "react-datepicker";

ChartJS.register(ArcElement, Tooltip, Legend);



const App = () => {
  const [values, setValues] = useState({
    TotalRFMS: 0,
    ProviderId_ProviderId_2: 0,
    ProviderId_ProviderId_3: 0,
    ProviderId_ProviderId_4: 0,
    ProviderId_ProviderId_5: 0,
    ProviderId_ProviderId_6: 0,
    ProductId_ProductId_10: 0,
    ProductId_ProductId_11: 0,
    ProductId_ProductId_12: 0,
    ProductId_ProductId_13: 0,
    ProductId_ProductId_14: 0,
    ProductId_ProductId_15: 0,
    ProductId_ProductId_16: 0,
    ProductId_ProductId_19: 0,
    ProductId_ProductId_2: 0,
    ProductId_ProductId_20: 0,
    ProductId_ProductId_21: 0,
    ProductId_ProductId_22: 0,
    ProductId_ProductId_23: 0,
    ProductId_ProductId_24: 0,
    ProductId_ProductId_27: 0,
    ProductId_ProductId_3: 0,
    ProductId_ProductId_4: 0,
    ProductId_ProductId_5: 0,
    ProductId_ProductId_6: 0,
    ProductId_ProductId_7: 0,
    ProductId_ProductId_8: 0,
    ProductId_ProductId_9: 0,
    ProductCategory_data_bundles: 0,
    ProductCategory_financial_services: 0,
    ProductCategory_movies: 0,
    ProductCategory_other: 0,
    ProductCategory_ticket: 0,
    ProductCategory_transport: 0,
    ProductCategory_tv: 0,
    ProductCategory_utility_bill: 0,
    ChannelId_ChannelId_2: 0,
    ChannelId_ChannelId_3: 0,
    ChannelId_ChannelId_5: 0,
    Amount: 0,
    Value: 0,
    PricingStrategy: 0,
    FraudResult: 0,
    Total_Transaction_Amount: 0,
    Average_Transaction_Amount: 0,
    Transaction_Count: 0,
    Transaction_Hour: 0,
    Transaction_Day: 0,
    Transaction_Month: 0,
    Transaction_Year: 0,
    model_name: 'logistic_regression'
  });

  const [modalVisible, setModalVisible] = useState(false);
  const [modalContent, setModalContent] = useState('');
  const [percentageData, setPercentageData] = useState(null);


  const minValues = {
    TotalRFMS: 2,
    Amount: -1000000,
    Value: 2,
    PricingStrategy: 0,
    Total_Transaction_Amount: -104900000,
    Average_Transaction_Amount: -425000,
    Transaction_Count: 1,
    Transaction_Hour:0, 
    Transaction_Day:1, 
    Transaction_Month:1,
    Transaction_Year:2018
    
  };


  const maxValues = {
    TotalRFMS: 3049, 
    Amount: 9880000, 
    Value: 9880000, 
    PricingStrategy: 4,
    Total_Transaction_Amount: 83451240,
    Average_Transaction_Amount: 8601821,
    Transaction_Count: 4091,
    Transaction_Hour:23, 
    Transaction_Day:31, 
    Transaction_Month:12,
    Transaction_Year:2019
  };

  const normalizeValue = (value, min, max) => {
    if (!((value>max)||(value<min)))
      return (value - min) / (max - min);
    else if(value>max)
      return 1
    else if(value<min)
      return 0
  };

  const handleChanges = (e) => {
    const { name, value } = e.target;
    setValues({ ...values, [name]: value });

  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const normalizedValues = {
      ...values,
      TotalRFMS: normalizeValue(values.TotalRFMS, minValues.TotalRFMS, maxValues.TotalRFMS),
      Amount: normalizeValue(values.Amount, minValues.Amount, maxValues.Amount),
      Value: normalizeValue(values.Value, minValues.Value, maxValues.Value),
      PricingStrategy: normalizeValue(values.PricingStrategy, minValues.PricingStrategy, maxValues.PricingStrategy),
      Total_Transaction_Amount: normalizeValue(values.Total_Transaction_Amount, minValues.Total_Transaction_Amount, maxValues.Total_Transaction_Amount),
      Average_Transaction_Amount: normalizeValue(values.Average_Transaction_Amount, minValues.Average_Transaction_Amount, maxValues.Average_Transaction_Amount),
      Transaction_Count: normalizeValue(values.Transaction_Count, minValues.Transaction_Count, maxValues.Transaction_Count),
      Transaction_Hour:normalizeValue(values.Transaction_Hour, minValues.Transaction_Hour, maxValues.Transaction_Hour), 
      Transaction_Day:normalizeValue(values.Transaction_Day, minValues.Transaction_Day, maxValues.Transaction_Day), 
      Transaction_Month:normalizeValue(values.Transaction_Month, minValues.Transaction_Month, maxValues.Transaction_Month),
      Transaction_Year:normalizeValue(values.Transaction_Year, minValues.Transaction_Year, maxValues.Transaction_Year)
    };


    try {
        const response = await fetch('https://bati-backend-app-v1-1.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(normalizedValues),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        const percentage = parseInt(data.message);
        setModalContent('Model: ' + JSON.stringify(data.model) + '\n\n Chance of default = '+ data.message+'%');
        setPercentageData(percentage);
        setModalVisible(true);    } 
        
        catch (error) {
        console.error('Error:', error);
        setModalContent('Error: ' + error.message);
        setPercentageData(null);
        setModalVisible(true);    }};

  const handleReset = () => {
    setValues({
      TotalRFMS: 0,
      ProviderId_ProviderId_2:0,
      ProviderId_ProviderId_3:0,
      ProviderId_ProviderId_4:0,
      ProviderId_ProviderId_5:0,
      ProviderId_ProviderId_6:0,
      ProductId_ProductId_10: 0,
      ProductId_ProductId_11: 0,
      ProductId_ProductId_12: 0,
      ProductId_ProductId_13: 0,
      ProductId_ProductId_14: 0,
      ProductId_ProductId_15: 0,
      ProductId_ProductId_16: 0,
      ProductId_ProductId_19: 0,
      ProductId_ProductId_2: 0,
      ProductId_ProductId_20: 0,
      ProductId_ProductId_21: 0,
      ProductId_ProductId_22: 0,
      ProductId_ProductId_23: 0,
      ProductId_ProductId_24: 0,
      ProductId_ProductId_27: 0,
      ProductId_ProductId_3: 0,
      ProductId_ProductId_4: 0,
      ProductId_ProductId_5: 0,
      ProductId_ProductId_6: 0,
      ProductId_ProductId_7: 0,
      ProductId_ProductId_8: 0,
      ProductId_ProductId_9: 0,
      ProductCategory_data_bundles: 0,
      ProductCategory_financial_services: 0,
      ProductCategory_movies: 0,
      ProductCategory_other: 0,
      ProductCategory_ticket: 0,
      ProductCategory_transport: 0,
      ProductCategory_tv: 0,
      ProductCategory_utility_bill: 0,
      ChannelId_ChannelId_2: 0,
      ChannelId_ChannelId_3: 0,
      ChannelId_ChannelId_5: 0,
      Amount: 0,
      Value: 0,
      PricingStrategy: 0,
      FraudResult: 0,
      Total_Transaction_Amount: 0,
      Average_Transaction_Amount: 0,
      Transaction_Count: 0,
      Transaction_Hour: 0,
      Transaction_Day: 0,
      Transaction_Month: 0,
      Transaction_Year: 0,
      model_name: 'logistic_regression'
    });
  };

  
  const [startDate, setStartDate] = useState(new Date()); 
  const extractDateParts = (date) => {
    if (!date) return { hour: '', day: '', year: '', month: '' };
    return {
      hour: date.getHours(),
      day: date.getDate(),
      year: date.getFullYear(),
      month: date.getMonth(),
    };
  };

  const handleDateChange = (date) => {
    setStartDate(date);
    const { hour, day, year, month } = extractDateParts(date);
    setValues((prevValues) => ({
      ...prevValues,
      Transaction_Hour: hour,
      Transaction_Day: day,
      Transaction_Month: month + 1,
      Transaction_Year: year,
    }));
  };


  
  
  const closeModal = () => {
    setModalVisible(false);
    setPercentageData(null);
  };

  
  return (
    <div className='container'>
      <h1>Bati Bank Prediction</h1>
      <form onSubmit={handleSubmit}>

        <div className='form-row'>
         
          <div className='part'>

            <label htmlFor='date'>Transaction time</label>
            <DatePicker
              showIcon
              selected={startDate}
              onChange={handleDateChange}
              timeInputLabel="Time:"
              dateFormat="MM/dd/yyyy h:mm aa"
              showTimeInput
              name='date'
              className='datepicker'
            />
            

            <label htmlFor='ProviderId'>Provider ID</label>
            <select
              type='number'
              name='ProviderId'
              value={values.ProviderId}
              onChange={handleChanges}
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
              <option value={5}>5</option>
              <option value={6}>6</option>
            </select>

            <label htmlFor='ProductId'>Product ID</label>
            <select
              name='ProductId'
              value={values.ProductId}
              onChange={handleChanges}
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
              <option value={5}>5</option>
              <option value={6}>6</option>
              <option value={7}>7</option>
              <option value={8}>8</option>
              <option value={9}>9</option>
              <option value={10}>10</option>
              <option value={11}>11</option>
              <option value={12}>12</option>
              <option value={13}>13</option>
              <option value={14}>14</option>
              <option value={15}>15</option>
              <option value={16}>16</option>
              <option value={19}>19</option>
              <option value={20}>20</option>
              <option value={21}>21</option>
              <option value={22}>22</option>
              <option value={23}>23</option>
              <option value={24}>24</option>
              <option value={27}>27</option>
            </select>
           

            <label htmlFor='ChannelId'>Channel ID</label>
            <select
              name='ChannelId'
              value={values.ChannelId}
              onChange={handleChanges}
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
              <option value={5}>5</option>
              <option value={6}>6</option>
            </select>
         
            </div>

            <div className='part'>
            

            <label htmlFor='PricingStrategy'>Pricing Strategy</label>
            <input
              type='text'
              name='PricingStrategy'
              value={values.PricingStrategy}
              onChange={handleChanges}
            />

            <label htmlFor='FraudResult'>Fraud Result</label>
            <input
              type='text'
              name='FraudResult'
              value={values.FraudResult}
              onChange={handleChanges}
            />
 
         
           <label htmlFor='Average_Transaction_Amount'>Avg Trans Amount</label>
          <input
            type='number'
            name='Average_Transaction_Amount'
            value={values.Average_Transaction_Amount}
            onChange={handleChanges}
          />
           <label htmlFor='Total_Transaction_Amount'>Tot Trans Amount</label>
          <input
            type='number'
            name='Total_Transaction_Amount'
            value={values.Total_Transaction_Amount}
            onChange={handleChanges}
          />
           <label htmlFor='model_name'>Model Name</label>
          <select
            type='text'
            name='model_name'
            value={values.model_name}
            onChange={handleChanges}
          >
            <option value='logistic_regression'>Logistic Regression</option>
            <option value='random_forest'>Random Forest</option>
            <option value='decision_tree'>Decision Tree</option>
            <option value='gradient_boosting'>Gradient Boosting</option>
          </select>

          </div>
          <div className='part'>

          <label htmlFor='Transaction_Count'>Transaction Count</label>
          <input
            type='number'
            name='Transaction_Count'
            value={values.Transaction_Count}
            onChange={handleChanges}
          />
          
        <label htmlFor='Amount'>Amount</label>
            <input
              type='number'
              name='Amount'
              value={values.Amount}
              onChange={handleChanges}
            />

            <label htmlFor='Value'>Value</label>
            <input
              type='number'
              name='Value'
              value={values.Value}
              onChange={handleChanges}
            />
            <label htmlFor='TotalRFMS'>Total RFMS</label>
            <input
              type='number'
              name='TotalRFMS'
              value={values.TotalRFMS}
              onChange={handleChanges}
            />
          </div>
        </div>

        
        <button id='reset' type='button' onClick={handleReset}>Reset</button>
        <button id='submit' type='submit'>Submit</button>
      </form>
      {modalVisible && (
        <div className="modal">
          <div className="modal-content">
            <span className="close" onClick={closeModal}>&times;</span>
            <pre>{modalContent}</pre>
            {percentageData !== null && (
              <div style={{ width: '100%', height: '300px' }}>
                <Pie
                  data={{
                    labels: ['Success', 'Failure'],
                    datasets: [{
                      data: [100-percentageData, percentageData],
                      backgroundColor: ['#36A2EB', '#FF6384'],
                      hoverBackgroundColor: ['#36A2EB', '#FF6384']
                    }]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                  }}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;