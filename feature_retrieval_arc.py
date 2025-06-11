import pandas as pd
from sklearn.metrics import mean_absolute_error  # optional, if you need further metrics
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class HouseholdExpenseModel:
        def __init__(self):
            # Define base columns for food and non-food features
            self.food_columns = [
                'meals_per_day', 'meals_school', 'meals_employer', 
                'meals_others', 'meals_paid', 'meals_home'
            ]
            self.non_food_columns = ['days_away']
            # Pipeline for training a CatBoost regressor with imputation.
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('catboost', CatBoostRegressor(random_state=42, verbose=0))
            ])
            self.combined_features = None  # Will be defined during training

        def preprocess(self, df_person, df_household):
            """Aggregates person-level features and merges them with household data."""
            # Rename columns for clarity
            df_person = df_person.rename(columns={
                "HH_ID": "HH_ID",
                "No. of days stayed away from home during last 30 days": "days_away",
                "No. of meals usually taken in a day": "meals_per_day",
                "No. of meals taken during last 30 days from school, balwadi etc.": "meals_school",
                "No. of meals taken during last 30 days from employer as perquisites or part of wage": "meals_employer",
                "No. of meals taken during last 30 days  others": "meals_others",
                "No. of meals taken during last 30 days on payment": "meals_paid",
                "No. of meals taken during last 30 days at home": "meals_home"
            })

            # Aggregate food-related features: compute both sum and mean
            agg_food = df_person.groupby('HH_ID')[self.food_columns].agg(['sum', 'mean'])
            agg_food.columns = ['_'.join(col) for col in agg_food.columns]  # Flatten MultiIndex
            agg_food.reset_index(inplace=True)

            # Aggregate non-food features (days_away): compute sum and mean
            agg_non_food = df_person.groupby('HH_ID')[self.non_food_columns].agg(['sum', 'mean'])
            agg_non_food.columns = ['_'.join(col) for col in agg_non_food.columns]
            agg_non_food.reset_index(inplace=True)

            # Merge the aggregated data with the household-level data
            df_merged = pd.merge(df_household, agg_food, on='HH_ID', how='left')
            df_merged = pd.merge(df_merged, agg_non_food, on='HH_ID', how='left')

            return df_merged

        def fit(self, df_person_train, df_household_train):
            """Fits the model on the training dataset."""
            df_merged = self.preprocess(df_person_train, df_household_train)
            # Define the combined feature list: all columns starting with 'meals' or 'days_away'
            self.combined_features = [col for col in df_merged.columns if col.startswith('meals') or col.startswith('days_away')]
            # Features and target: using TotalExpense as the target variable
            X_train = df_merged[self.combined_features]
            y_train = df_merged['TotalExpense']
            # Train the pipeline model
            self.model.fit(X_train, y_train)

        def predict(self, df_person, df_household):
            """Generates predictions for the provided dataset."""
            df_merged = self.preprocess(df_person, df_household)
            X = df_merged[self.combined_features]
            df_merged['Personal_level_constant'] = self.model.predict(X)
            return df_merged[['HH_ID', 'Personal_level_constant']]

        def evaluate(self, df_person, df_household):
            """Evaluates the model by computing the correlation between predicted expense and actual TotalExpense."""
            df_merged = self.preprocess(df_person, df_household)
            X = df_merged[self.combined_features]
            df_merged['Personal_level_constant'] = self.model.predict(X)
            # Compute Pearson correlation between predicted expense and actual TotalExpense
            correlation = np.corrcoef(df_merged['Personal_level_constant'], df_merged['TotalExpense'])[0, 1]
            return correlation, df_merged

        def save_model(self, filename="expense_model.pkl"):
            """Saves the complete model pipeline to a pickle file."""
            joblib.dump(self, filename)
            print(f"Model saved as {filename}")

        @staticmethod
        def load_model(filename="expense_model.pkl"):
            """Loads the model pipeline from a pickle file."""
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model

class HouseholdDemographicModel:
        def __init__(self):
            self.features = [
                "total_persons", "num_infants", "num_adults", "infant_dependency_ratio",
                "num_dependents", "overall_dependency_ratio", "avg_age_y", "std_age_y",
                "min_age_y", "max_age_y", "avg_education"
            ]
            self.lr_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('lr', LinearRegression())
            ])
            self.rf_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('rf', RandomForestRegressor(random_state=42))
            ])

        @staticmethod
        def compute_household_stats(group):
            print(group)
            total_persons = group.shape[0]
            num_infants = (group["age"] < 5).sum()
            num_adults = (group["age"] >= 18).sum()
            infant_dependency_ratio = num_infants / num_adults if num_adults > 0 else 0.0
            num_dependents = ((group["age"] < 18) | (group["age"] >= 65)).sum()
            overall_dependency_ratio = num_dependents / total_persons if total_persons > 0 else 0.0
            avg_age = group["age"].mean()
            std_age = group["age"].std()
            min_age = group["age"].min()
            max_age = group["age"].max()
            avg_education = group["education_years"].mean()
            
            return pd.Series({
                "total_persons": total_persons,
                "num_infants": num_infants,
                "num_adults": num_adults,
                "infant_dependency_ratio": infant_dependency_ratio,
                "num_dependents": num_dependents,
                "overall_dependency_ratio": overall_dependency_ratio,
                "avg_age": avg_age,
                "std_age": std_age,
                "min_age": min_age,
                "max_age": max_age,
                "avg_education": avg_education
            })

        def preprocess(self, df_person, df_household):
            # Compute household-level demographic stats from person-level data.
            household_stats = df_person.groupby("HH_ID").apply(self.compute_household_stats).reset_index()
            # Rename columns for consistency.
            household_stats = household_stats.rename(columns={
                "avg_age": "avg_age_y",
                "std_age": "std_age_y",
                "min_age": "min_age_y",
                "max_age": "max_age_y"
            })
            # Merge with household data.
            df_merged = pd.merge(df_household, household_stats, on="HH_ID", how="left")
            # Fill missing values for features and target.
            # df_merged[self.features + ["TotalExpense"]] = df_merged[self.features + ["TotalExpense"]].fillna(0)
            return df_merged

        def fit(self, df_person_train, df_household_train):
            df_merged = self.preprocess(df_person_train, df_household_train)
            X_train = df_merged[self.features]
            y_train = df_merged["TotalExpense"]
            self.lr_pipeline.fit(X_train, y_train)
            self.rf_pipeline.fit(X_train, y_train)

        def predict(self, df_person, df_household):
            df_merged = self.preprocess(df_person, df_household)
            X = df_merged[self.features]
            df_merged["demographic_param_lr"] = self.lr_pipeline.predict(X)
            df_merged["demographic_param_rf"] = self.rf_pipeline.predict(X)
            return df_merged

        def evaluate(self, df_person, df_household):
            df_merged = self.preprocess(df_person, df_household)
            X = df_merged[self.features]
            y = df_merged["TotalExpense"]
            pred_lr = self.lr_pipeline.predict(X)
            pred_rf = self.rf_pipeline.predict(X)
            lr_corr = pd.Series(pred_lr).corr(y)
            rf_corr = pd.Series(pred_rf).corr(y)
            df_merged["demographic_param_lr"] = pred_lr
            df_merged["demographic_param_rf"] = pred_rf
            return {"lr_corr": lr_corr, "rf_corr": rf_corr}, df_merged

        def save_model(self, filename="demographic_model.pkl"):
            joblib.dump(self, filename)
            print(f"Model saved as {filename}")

        @staticmethod
        def load_model(filename="demographic_model.pkl"):
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model  

def f(y,x):
    # Load test data.
    df_person_test = x
    df_household_test = y 
    print(df_person_test.columns)
    print(df_household_test.columns)
    # Load the saved model.
    demo_model = HouseholdDemographicModel.load_model("demographic_model.pkl")

    processed_df = pd.merge(demo_model.predict(df_person_test, df_household_test)[['HH_ID','demographic_param_lr']],on='HH_ID',how='left')

    return processed_df

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor

def g(x,y):

    # df_person_train = pd.read_csv(train_data_per_path)
    # df_household_train = pd.read_csv(train_data_hh_path)

    # df_person_test = pd.read_csv(test_data_per_path)
    # df_household_test = pd.read_csv(test_data_hh_path)


    expense_model = HouseholdExpenseModel.load_model('expense_model.pkl')
    # print(expense_model.predict(df_person_train, df_household_train))
    # print(expense_model.predict(df_person_test, df_household_test))

    processed_df = pd.merge(expense_model.predict(y, x),on='HH_ID',how='left')
    # processed_df_t = pd.merge(processed_df_t,expense_model.predict(df_person_train, df_household_train),on='HH_ID',how='left')

    cols = processed_df.columns.tolist()
    i1 = cols.index("having_constant")
    i2 = cols.index("online_constant")
    cols[i1], cols[i2] = cols[i2], cols[i1]
    test_processed = processed_df[cols]

    # cols = processed_df_t.columns.tolist()
    i1 = cols.index("having_constant")
    i2 = cols.index("online_constant")
    cols[i1], cols[i2] = cols[i2], cols[i1]
    train_processed = processed_df[cols]
    # test_processed.to_csv('test_processed.csv',index=False)
    return train_processed


def k(x):
    def load_models(model_path="trained_models.pkl"):
        """
        Loads the trained models from disk.
        """
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                trained_models = pickle.load(f)
            print(f"Models loaded from {model_path}")
            return trained_models
        else:
            print(f"Model file {model_path} not found.")
            return None

    def test_models(test, trained_models):
        """
        Uses the trained models to generate predictions on the test CSV.
        
        For each state, the function:
        - Uses the corresponding online model to compute the 'online_constant'.
        - Uses the corresponding assert model to compute the 'having_constant'.
        - Prints the test correlation between the predictions and the target (if available).
        
        Returns:
        A DataFrame containing HH_ID, online_constant, and having_constant.
        """    
        # Identify relevant columns in test data
        online_cols = [col for col in test.columns if col.startswith('Is_online_')]
        assert_cols = [col for col in test.columns if col.startswith('Is_HH_Have_')]
        
        test[online_cols] = test[online_cols].fillna(0)
        test[assert_cols] = test[assert_cols].fillna(0)
        
        # Define the target variable if available (for correlation evaluation)
        if 'TotalExpense' in test.columns:
            test['Target'] = test['TotalExpense']
        else:
            test['Target'] = np.nan
        
        # Prepare a DataFrame for the results
        results = test[['HH_ID', 'State']].copy()
        results['online_constant'] = np.nan
        results['having_constant'] = np.nan
        
        overall_preds_online = []
        overall_targets_online = []
        overall_preds_assert = []
        overall_targets_assert = []
        
        states = sorted(test['State'].unique())
        print("\nTesting for States:", states)
        
        for state in states:
            print(f"\nProcessing State: {state}")
            state_data = test[test['State'] == state].copy()
            if state not in trained_models:
                print(f"No trained model for State {state}. Skipping.")
                continue
            
            # Get models for the state
            poly_online, linreg_online = trained_models[state]['online']
            poly_assert, linreg_assert = trained_models[state]['assert']
            
            # Predict online feature
            X_online = state_data[online_cols]
            X_online_poly = poly_online.transform(X_online)
            preds_online = linreg_online.predict(X_online_poly)
            
            # Predict assert feature
            X_assert = state_data[assert_cols]
            X_assert_poly = poly_assert.transform(X_assert)
            preds_assert = linreg_assert.predict(X_assert_poly)
            
            results.loc[results['State'] == state, 'online_constant'] = preds_online
            results.loc[results['State'] == state, 'having_constant'] = preds_assert
            
            # Evaluate correlations if target variation exists
            if state_data['Target'].nunique() > 1:
                corr_online = np.corrcoef(preds_online, state_data['Target'])[0, 1]
                corr_assert = np.corrcoef(preds_assert, state_data['Target'])[0, 1]
                print(f"State {state} - Test Correlation (Online model): {corr_online:.4f}")
                print(f"State {state} - Test Correlation (Assert model): {corr_assert:.4f}")
            else:
                print(f"State {state} - Insufficient target variation for correlation evaluation.")
            
            overall_preds_online.extend(preds_online)
            overall_targets_online.extend(state_data['Target'])
            overall_preds_assert.extend(preds_assert)
            overall_targets_assert.extend(state_data['Target'])
        
        overall_corr_online = np.corrcoef(overall_preds_online, overall_targets_online)[0, 1] if np.unique(overall_targets_online).size > 1 else np.nan
        overall_corr_assert = np.corrcoef(overall_preds_assert, overall_targets_assert)[0, 1] if np.unique(overall_targets_assert).size > 1 else np.nan
        
        print("\nCombined Testing Correlations:")
        print(f"Overall Online model correlation: {overall_corr_online:.4f}")
        print(f"Overall Assert model correlation: {overall_corr_assert:.4f}")
        
        # Optionally drop the State column
        results = results.drop(columns=['State'])
        
        return results

    # ----------------------- Example Usage -----------------------
    # if __name__ == "__main__":
        # Paths for your training and testing CSV files
        # train_csv_path = train_data_hh_path
        # # test_csv_path = test_data_hh_path
        # model_save_path = 'trained_models.pkl'
        
        # Train models and save them
    trained_models = load_models(model_path = "trained_models.pkl")
    # test_results = test_models(test_csv_path, trained_models)

    # # Optionally, save the test results to CSV
    # test_results.to_csv("test_features.csv", index=False)

    # print("\nTest features head:")
    # print(test_results.head())
    processed_df = pd.merge(test_models(x,trained_models),on='HH_ID',how='left')
    return processed_df