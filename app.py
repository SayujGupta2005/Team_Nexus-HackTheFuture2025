from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import numpy as np
import pandas as pd
from chat import chat
import joblib
from feature_retrieval_arc import g, k, f
from feature_retrieval_arc import HouseholdDemographicModel, HouseholdExpenseModel

# Load the model
model = joblib.load("model\\xgb_best_model_v2.pkl")
classification_model = joblib.load("model\\xgb_classifier.pkl")
# classification_model = joblib.load("model\\rf_classification_model.pkl")
# regression_model_1 = joblib.load("model\\xgb_regressor_model_above95.pkl")
# regression_model_2 = joblib.load("model\\xgb_regressor_model_below95.pkl")
print("Model loaded successfully!!!")

sector_map = {
    "Rural": 1,
    "Urban": 2
}

state_map = {
    "Jammu and Kashmir": 1,
    "Himachal Pradesh": 2,
    "Punjab": 3,
    "Chandigarh": 4,
    "Uttarakhand": 5,
    "Haryana": 6,
    "Delhi": 7,
    "Rajasthan": 8,
    "Uttar Pradesh": 9,
    "Bihar": 10,
    "Sikkim": 11,
    "Arunachal Pradesh": 12,
    "Nagaland": 13,
    "Manipur": 14,
    "Mizoram": 15,
    "Tripura": 16,
    "Meghalaya": 17,
    "Assam": 18,
    "West Bengal": 19,
    "Jharkhand": 20,
    "Odisha": 21,
    "Chhattisgarh": 22,
    "Madhya Pradesh": 23,
    "Gujarat": 24,
    "Dadra and Nagar Haveli and Daman and Diu": 25,
    "Maharashtra": 27,
    "Andhra Pradesh": 28,
    "Karnataka": 29,
    "Goa": 30,
    "Lakshadweep": 31,
    "Kerala": 32,
    "Tamil Nadu": 33,
    "Puducherry": 34,
    "Andaman and Nicobar Islands": 35,
    "Telangana": 36,
    "Ladakh": 37,
}

nic_sections = {
    '01': 'A - Agriculture, Forestry and Fishing',
    '02': 'A - Agriculture, Forestry and Fishing',
    '03': 'A - Agriculture, Forestry and Fishing',
    '05': 'B - Mining and Quarrying',
    '06': 'B - Mining and Quarrying',
    '07': 'B - Mining and Quarrying',
    '08': 'B - Mining and Quarrying',
    '09': 'B - Mining and Quarrying',
    '10': 'C - Manufacturing',
    '11': 'C - Manufacturing',
    '12': 'C - Manufacturing',
    '13': 'C - Manufacturing',
    '14': 'C - Manufacturing',
    '15': 'C - Manufacturing',
    '16': 'C - Manufacturing',
    '17': 'C - Manufacturing',
    '18': 'C - Manufacturing',
    '19': 'C - Manufacturing',
    '20': 'C - Manufacturing',
    '21': 'C - Manufacturing',
    '22': 'C - Manufacturing',
    '23': 'C - Manufacturing',
    '24': 'C - Manufacturing',
    '25': 'C - Manufacturing',
    '26': 'C - Manufacturing',
    '27': 'C - Manufacturing',
    '28': 'C - Manufacturing',
    '29': 'C - Manufacturing',
    '30': 'C - Manufacturing',
    '31': 'C - Manufacturing',
    '32': 'C - Manufacturing',
    '33': 'C - Manufacturing',
    '35': 'D - Electricity, Gas, Steam and Air Conditioning Supply',
    '36': 'E - Water Supply, Sewerage, Waste Management and Remediation',
    '37': 'E - Water Supply, Sewerage, Waste Management and Remediation',
    '38': 'E - Water Supply, Sewerage, Waste Management and Remediation',
    '39': 'E - Water Supply, Sewerage, Waste Management and Remediation',
    '41': 'F - Construction',
    '42': 'F - Construction',
    '43': 'F - Construction',
    '45': 'G - Wholesale and Retail Trade; Repair of Motor Vehicles and Motorcycles',
    '46': 'G - Wholesale and Retail Trade; Repair of Motor Vehicles and Motorcycles',
    '47': 'G - Wholesale and Retail Trade; Repair of Motor Vehicles and Motorcycles',
    '49': 'H - Transportation and Storage',
    '50': 'H - Transportation and Storage',
    '51': 'H - Transportation and Storage',
    '52': 'H - Transportation and Storage',
    '53': 'H - Transportation and Storage',
    '55': 'I - Accommodation and Food Service Activities',
    '56': 'I - Accommodation and Food Service Activities',
    '58': 'J - Information and Communication',
    '59': 'J - Information and Communication',
    '60': 'J - Information and Communication',
    '61': 'J - Information and Communication',
    '62': 'J - Information and Communication',
    '63': 'J - Information and Communication',
    '64': 'K - Financial and Insurance Activities',
    '65': 'K - Financial and Insurance Activities',
    '66': 'K - Financial and Insurance Activities',
    '68': 'L - Real Estate Activities',
    '69': 'M - Professional, Scientific and Technical Activities',
    '70': 'M - Professional, Scientific and Technical Activities',
    '71': 'M - Professional, Scientific and Technical Activities',
    '72': 'M - Professional, Scientific and Technical Activities',
    '73': 'M - Professional, Scientific and Technical Activities',
    '74': 'M - Professional, Scientific and Technical Activities',
    '75': 'M - Professional, Scientific and Technical Activities',
    '77': 'N - Administrative and Support Service Activities',
    '78': 'N - Administrative and Support Service Activities',
    '79': 'N - Administrative and Support Service Activities',
    '80': 'N - Administrative and Support Service Activities',
    '81': 'N - Administrative and Support Service Activities',
    '82': 'N - Administrative and Support Service Activities',
    '84': 'O - Public Administration and Defence; Compulsory Social Security',
    '85': 'P - Education',
    '86': 'Q - Human Health and Social Work Activities',
    '87': 'Q - Human Health and Social Work Activities',
    '88': 'Q - Human Health and Social Work Activities',
    '90': 'R - Arts, Entertainment and Recreation',
    '91': 'R - Arts, Entertainment and Recreation',
    '92': 'R - Arts, Entertainment and Recreation',
    '93': 'R - Arts, Entertainment and Recreation',
    '94': 'S - Other Service Activities',
    '95': 'S - Other Service Activities',
    '96': 'S - Other Service Activities',
    '97': 'T - Activities of Households as Employers',
    '98': 'T - Activities of Households as Employers',
    '99': 'U - Activities of Extraterritorial Organizations and Bodies'
}

nco_sections = {
    '1': 'Managers',
    '2': 'Professionals',
    '3': 'Technicians and Associate Professionals',
    '4': 'Clerical Support Workers',
    '5': 'Service and Sales Workers',
    '6': 'Skilled Agricultural, Forestry, and Fishery Workers',
    '7': 'Craft and Related Trades Workers',
    '8': 'Plant and Machine Operators and Assemblers',
    '9': 'Elementary Occupations',
    '0': 'Armed Forces Occupations'
}

group_to_num = {'Armed Forces Occupations': 1,
 'Managers': 2,
 'Professionals': 3,
 'Technicians and Associate Professionals': 4,
 'Clerical Support Workers': 5,
 'Service and Sales Workers': 6,
 'Skilled Agricultural, Forestry, and Fishery Workers': 7,
 'Craft and Related Trades Workers': 8,
 'Plant and Machine Operators and Assemblers': 9,
 'Elementary Occupations': 10}

coef_list = [0.08599469450932685, 0.017472204136149014, -0.018403342544424033, 0.014986391813997592, 0.027851679145488552, 0.01154935460641519, -0.005017965598407833, -0.0013414755683872877, -0.0037792362754634186, -0.008714205476360579, 0.010008492824308534]
coef_list1 = [0.22190073014586958, 0.025360494795706734, 0.2682093576723438]
coef_list2 = [-0.17588991633308818, 0.16057196556897707, 0.24053641154395702, 0.00976584351888473, -0.01487601462035179]
coef_list3 = [0.2759033345016436, 0.23119865931256456, 0.010382606801460331]

# Initialize Flask
app = Flask(__name__)
# Allow requests from any origin to fix CORS issues
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5000"}})

df = pd.read_csv("data\\hh-level-test-data.csv")
person_df = pd.read_csv("data\\person-level-test-data.csv")
df.fillna(0, inplace=True)
person_df.fillna(0, inplace=True)

# Search route
@app.route('/search', methods=['POST'])
def search_household():
    """Search household details based on HCES ID."""
    data = request.json
    hh_id = data.get("HH_ID")
    
    result = df[df["HH_ID"] == hh_id]
    
    if result.empty:
        return jsonify({"error": "Household ID not found"}), 404
    
    return result.to_json(orient="records")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_expense():
    """Predict MPCE and Total Expense based on input data."""
    data = request.json
    # print(data)
    # hh_id = data.get("HH_ID",0)
    # print(hh_id)
    entry = data.get("entry")
    data.pop("entry")
    l = [data]
    # hh_df = pd.DataFrame(l)
    # person_df = pd.DataFrame(entry)
    # person_df.drop(columns=["serial"], inplace=True)
    # hh_df["HH_ID"] = hh_id
    # person_df["HH_ID"] = hh_id
    # print(hh_df.columns)
    # print(person_df.columns)
    # Extract basic inputs
    # entry[0]["HH_ID"] = hh_id
    sector = sector_map.get(data.get("Sector"), 0)
    state = state_map.get(data.get("State"), 0)
    nss_region = int(data.get("NSS Region", 0))
    district = int(data.get("District", 0))
    household_size = int(data.get("Household Size", 1))
    household_type = int(data.get("Household Type", 1))
    nco_3d = int(data.get("NCO 3D", 0))
    nic_5d = str(data.get("NIC 5D", "00")).zfill(5)
    nic_section = ord((nic_sections.get(nic_5d[:2],"01"))[0])-97+1
    nco_section = group_to_num[nco_sections.get(str(nco_3d)[0], "1")]
    mobile = data.get("Mobile Handset", 0)

    # Online Purchase
    purchases = [
        data.get("Clothing", 0), data.get("Footwear", 0), data.get("Furniture & Fixtures", 0),
        data.get("Mobile Handset", 0), data.get("Personal Goods", 0), data.get("Recreation Goods", 0),
        data.get("Household Ap0"
        "pliances", 0), data.get("Crockery & Utensils", 0), data.get("Sports Goods", 0),
        data.get("Medical Equipment", 0), data.get("Bedding", 0)
    ]

    # Household Possessions
    possessions = [
        data.get("Television", 0), data.get("Radio", 0), data.get("Laptop/PC", 0),
        data.get("Bicycle", 0), data.get("Motorcycle/Scooter", 0), data.get("Motorcar/Jeep/Van", 0),
        data.get("Trucks", 0), data.get("Animal Cart", 0), data.get("Refrigerator", 0),
        data.get("Washing Machine", 0), data.get("Air Conditioner", 0)
    ]

    online_activity = 0
    gl = 0
    for i in range(0, len(coef_list)):
        online_activity+= coef_list[i]*purchases[gl]

    entertainment = 0
    for i in range(0, len(coef_list1)):
        entertainment+= coef_list1[i]*possessions[gl]
        gl+=1
    
    vehicle = 0
    for j in range(0, len(coef_list2)):
        vehicle+= coef_list2[j]*possessions[gl]
        gl+=1
    
    electronic = 0
    for k in range(0, len(coef_list3)):
        electronic+= coef_list3[k]*possessions[gl]
        gl+=1

    head_entry = None
    for e in entry:
        try:
            # Convert relation to integer to compare with 1
            if int(e.get("relation", -1)) == 1:
                head_entry = e
                break
        except ValueError:
            continue

    if head_entry is None:
        return jsonify({"error": "Head of Household not found"}), 400
    
    head_age = int(head_entry.get("age", 0))
    head_gender = int(head_entry.get("gender",0))
    religion_head = int(head_entry.get("religion", 0))
    caste_head = int(head_entry.get("caste", 0))
    head_edu = int(head_entry.get("education_level", 0))
    head_education_years = int(head_entry.get("education_years", 0))
    nm = 0
    for i in entry:
        if i['gender']==1:
            nm+=1
    male_to_total_ratio = nm/len(entry)
    Is_couple = 0
    education = 0
    education_year = 0
    away_home = 0
    day_meal = 0
    home_meal = 0
    internet_use = 0
    x = 0
    # print(entry[0]["days_away"])
    for i in entry:
        education+= int(i['education_level'])
        education_year+= int(i['education_years'])
        away_home += int(i['days_away'])
        day_meal += int(i['meals_usual'])
        home_meal+= int(i['meals_home'])
        internet_use  = max(int(i['internet']), internet_use)
        x += (int(i['meals_school'])+int(i["meals_employer"])+int(i["meals_others"])+int(i["meals_payment"]))
        if i['marital']==2:
            Is_couple+=1
    education = education/len(entry)
    education_year = education_year/len(entry)
    away_home = away_home/len(entry)
    day_meal = day_meal/len(entry)
    home_meal = home_meal/len(entry)
    away_meal = x/len(entry)
    # Feature Engineering
    feature_vector = [
        sector, state, nss_region, district, household_type, household_size, religion_head, caste_head,
        nco_section, nic_section, mobile, online_activity, entertainment, vehicle, electronic, head_age, head_gender, head_edu,
        head_education_years, male_to_total_ratio, Is_couple, education, education_year, away_home, day_meal, home_meal, away_meal, internet_use
    ]
    # col1 = f(hh_df, person_df)
    # print(col1)
    # col23 = g(hh_df, person_df)
    # print(col23)
    # col4 = k(hh_df)
    # print(col4)

    feature_array = np.array(feature_vector).reshape(1, -1)
    
    
    # feature_array = feature_array.apply(pd.to_numeric, errors='coerce').fillna(0)
    # feature_array = feature_array.to_numpy(dtype=np.float32)
    print(feature_array)
    # # Prediction using real model
    classprediction = classification_model.predict(feature_array)[0]
    print(classprediction)
    prediction = model.predict(feature_array)[0]
    print(prediction)
    total_expense = int(prediction*household_size)
    mpce = int(prediction)
    print(total_expense, mpce)

    return jsonify({"Total Expense": total_expense, "MPCE": mpce})

@app.route('/analyze_state', methods=['POST'])
def analyze_state():
    data = request.json
    state = data.get("state")
    sector = data.get("sector")
    state_code = state_map.get(state)
    sector_code = sector_map.get(sector) if sector else None

    filtered = df.copy()
    if state_code:
        filtered = filtered[filtered["State"] == state_code]
    if sector_code:
        filtered = filtered[filtered["Sector"] == sector_code]
        india = df[df["Sector"] == sector_code]
    else:
        india = df.copy()

    return compute_distribution(filtered, india)

@app.route('/analyze_district', methods=['POST'])
def analyze_district():
    data = request.json
    district = data.get("district")
    sector = data.get("sector")
    sector_code = sector_map.get(sector) if sector else None

    filtered = df.copy()
    if district:
        filtered = filtered[filtered["District"] == int(district)]
    if sector_code:
        filtered = filtered[filtered["Sector"] == sector_code]
        india = df[df["Sector"] == sector_code]
    else:
        india = df.copy()

    return compute_distribution(filtered, india)

@app.route('/analyze_nss_region', methods=['POST'])
def analyze_nss_region():
    data = request.json
    nss_region = data.get("nss_region")
    sector = data.get("sector")
    sector_code = sector_map.get(sector) if sector else None

    filtered = df.copy()
    if nss_region:
        filtered = filtered[filtered["NSS-Region"] == int(nss_region)]
    if sector_code:
        filtered = filtered[filtered["Sector"] == sector_code]
        india = df[df["Sector"] == sector_code]
    else:
        india = df.copy()

    return compute_distribution(filtered, india)

def compute_distribution(filtered, india):
    if filtered.empty:
        return jsonify({"error": "No matching data found."})

    bins = [0, 5000, 10000, 20000, 40000, 60000, 80000, 100000, float('inf')]
    labels = ['0 - 5K', '5K - 10K', '10K - 20K', '20K - 40K', '40K - 60K', '60K - 80K', '80K - 1L', 'Above 1L']

    filtered = filtered.copy()
    filtered['ExpenseClass'] = pd.cut(filtered['TotalExpense'], bins=bins, labels=labels, include_lowest=True)
    local_dist = filtered['ExpenseClass'].value_counts().sort_index()

    india = india.copy()
    india['ExpenseClass'] = pd.cut(india['TotalExpense'], bins=bins, labels=labels, include_lowest=True)
    india_dist = india['ExpenseClass'].value_counts().sort_index()

    avg_total_local = round(filtered['TotalExpense'].mean(), 2)
    avg_mpce_local = round(avg_total_local / filtered['HH Size (For FDQ)'].mean(), 2)
    avg_total_india = round(india['TotalExpense'].mean(), 2)
    avg_mpce_india = round(avg_total_india / india['HH Size (For FDQ)'].mean(), 2)

    return jsonify({
        "local_distribution": local_dist.to_dict(),
        "india_distribution": india_dist.to_dict(),
        "avg_local": {"MPCE": avg_mpce_local, "TotalExpense": avg_total_local},
        "avg_india": {"MPCE": avg_mpce_india, "TotalExpense": avg_total_india}
    })

# Load data
hh_test = pd.read_csv("data\\hh-level-test-data.csv")
person_test = pd.read_csv("data\\person-level-test-data.csv")

# Routes
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        if question.lower() in ["hi", "hello", "hey"]:
            return jsonify({"answer": "👋 Hello! I'm your data assistant. How can I help you today?"})
        answer = chat(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)
