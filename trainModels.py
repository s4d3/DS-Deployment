import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def trainModels():
    df = st.session_state["df"]
    features = st.session_state["features"]
    target = st.session_state["target"]

    X = df[features]
    y = df[target]

    test_size = st.slider("Test size (%)", 10, 50, 20)
    random_state = st.number_input("Random state", value = 42, step=1)

    models = {
        "LinearRegression" : LinearRegression(),
        "RandomForestRegressor" : RandomForestRegressor(),
        "Support Vector Regressor" : SVR(),
        "K-Nearest Neighbors Regressor" : KNeighborsRegressor()

    }

    selected_models = st.multiselect("Select models to train", list(models.keys()), default=list(models.keys()))

    if st.button("Train models"):
        if not selected_models:
            st.error("Please select at least one model to train")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state = random_state)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["scaler"] = scaler

        st.success("Data was split into training and testing sets")

        trained_models = {}
        for model_name in selected_models:
            model = models[model_name]
            model.fit(X_train, y_train)
            trained_models[model_name] = model
        
        st.session_state["trained_models"] = trained_models
        st.success("All models have been trained")
