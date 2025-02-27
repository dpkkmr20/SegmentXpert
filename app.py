from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained K-Means model
kmeans = joblib.load(r"C:\Users\HP\OneDrive\Documents\Data Science-Self\ML\Customer Segmentation Project\model\model.pkl")

# Cluster Mapping (Adjust based on your clustering results)
cluster_mapping = {
    3: 'True Friends 💎',
    2: 'Strangers ❌',
    1: 'Butterflies 🦋',
    0: 'Barnacles 🏗️'
}

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

         # Load the uploaded dataset
        retail = pd.read_csv(filepath)

        # Get the minimum and maximum date
        #min_date = pd.to_datetime(retail['InvoiceDate'].min())
        #max_date = pd.to_datetime(retail['InvoiceDate'].max())
        #duration = max_date - min_date

        retail.dropna(inplace=True,axis=0)

        #print("Number of duplicates before cleaning:",df.duplicated().sum())
        retail = retail.drop_duplicates(keep="first")
        #print("Number of duplicates after cleaning:",retail.duplicated().sum())

        # Removing returned products (Invoice numbers starting with C) from the data set
        #retail = retail[~retail["Invoice"].str.contains("C", na = False)]

        # Removing missing values from the dataset
        retail.dropna(inplace = True)

        retail['InvoiceDate']=pd.to_datetime(retail['InvoiceDate'])

        #changing the datatype of customer ID
        retail['Customer ID']=retail['Customer ID'].astype(str)

        total_customers=retail['Customer ID'].nunique()

        #Calculatong recency, frequency and monetary
        retail['Amount']=retail['Quantity']*retail['Price']
        rfm_m=retail.groupby('Customer ID')['Amount'].sum()
        rfm_m=rfm_m.reset_index()
        #print(rfm_m.head())
        #print(rfm_m.shape)

        # Separate positive (purchases) and negative (returns)
        purchase_df = rfm_m[rfm_m["Amount"] >= 0] 
        refund_df = rfm_m[rfm_m["Amount"] < 0]

        #retail[(retail['Quantity'] >0) & (retail['Invoice'].astype(str).str.contains('C'))]

        #There are no customers with canceled transactions having a positive quantity, which confirms that the collected data is accurate.
        
        #Merging dataset
        customer_df = purchase_df.merge(refund_df, on="Customer ID", how="outer", suffixes=("_purchase", "_refund"))
        customer_df.fillna(0,inplace=True)
        customer_df['TotalAmount']=customer_df['Amount_purchase']+customer_df['Amount_refund']
        customer_df.rename(columns={"TotalAmount": "Monetary"}, inplace=True)

        #Calculating frequency
        rfm_f=retail.groupby('Customer ID')['Invoice'].count()
        rfm_f=rfm_f.reset_index()
        rfm_f.columns=['Customer ID','Frequency']
        #rfm_f.head()
        #rfm_f.shape

        customer_df = customer_df.merge(rfm_f, on="Customer ID", how="left")

        # Calculating recent date for the dataset
        now=max(retail['InvoiceDate'])

        #Calculating recency
        rfm_r=now-retail.groupby('Customer ID')['InvoiceDate'].max()
        rfm_r=rfm_r.reset_index()
        rfm_r.columns=['Customer ID','Recency']
        rfm_r['Recency'] = rfm_r['Recency'].dt.days

        customer_df = customer_df.merge(rfm_r, on="Customer ID", how="left")

        customer_df.drop(columns={'Amount_purchase','Amount_refund'},inplace=True)
        
        #again, check for negative monetary values
        customer_with_refund=customer_df[customer_df['Monetary']<=0].shape[0]

        #Since monetary values naturally vary across customers, removing high-value outliers without proper reasoning might lead to incorrect conclusions.
        
        rfm_scores=customer_df.copy()

        # Calculating Recency, Frequency and Monetary Scores
        rfm_scores["RecencyScore"]  = pd.qcut(rfm_scores["Recency"], 10, labels=[10,9,8,7,6,5, 4, 3, 2, 1])
        rfm_scores["FrequencyScore"] = pd.qcut(rfm_scores["Frequency"].rank(method="first"), 10, labels=[1, 2, 3, 4, 5,6,7,8,9,10])
        rfm_scores["MonetaryScore"] = pd.qcut(rfm_scores["Monetary"], 10, labels=[1, 2, 3, 4, 5,6,7,8,9,10])

        #converting datatype from string to numeric 
        rfm_scores['RecencyScore'] = pd.to_numeric(rfm_scores['RecencyScore'], errors='coerce')
        rfm_scores['FrequencyScore'] = pd.to_numeric(rfm_scores['FrequencyScore'], errors='coerce')
        rfm_scores['MonetaryScore'] = pd.to_numeric(rfm_scores['MonetaryScore'], errors='coerce')

        customer_df_with_rfm_scores=rfm_scores.copy()

        rfm_scores.drop(columns={'Monetary','Frequency','Recency'},inplace=True)

        #storing it into x variable for training
        x=rfm_scores.loc[:,['RecencyScore','FrequencyScore','MonetaryScore']]

        # Predict clusters
        rfm_scores['Cluster'] = kmeans.predict(rfm_scores[['RecencyScore', 'FrequencyScore', 'MonetaryScore']])
        rfm_scores['Category'] = rfm_scores['Cluster'].map(cluster_mapping)
        rfm_scores.drop(columns={'RecencyScore', 'FrequencyScore', 'MonetaryScore'},inplace=True)

        # Save categorized data into multiple sheets
        result_filepath = os.path.join(RESULTS_FOLDER, 'categorized_customers.xlsx')
        with pd.ExcelWriter(result_filepath, engine='xlsxwriter') as writer:
            for category in rfm_scores['Category'].unique():
                rfm_scores[rfm_scores['Category'] == category].to_excel(writer, sheet_name=category, index=False)
            refund_df.to_excel(writer, sheet_name='Refunds', index=False)  # Save refund data separately

        # Count customers per category
        category_counts = rfm_scores['Category'].value_counts().to_dict()

        return jsonify({'success': True, 'counts': category_counts, 'file_path': result_filepath, 'total_customers': total_customers, 'customer_with_refund':customer_with_refund})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download')
def download_file():
    result_filepath = os.path.join(RESULTS_FOLDER, 'categorized_customers.xlsx')
    if os.path.exists(result_filepath):
        return send_file(result_filepath, as_attachment=True)
    return jsonify({'error': 'File not found'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
