import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd


from sklearn.metrics import (
    r2_score, mean_squared_error
    )


def printChart(model_name, model, X_test, y_test):
    st.subheader(f"{model_name}")

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"**R2 Score:** {r2:.2f}")
    st.write(f"**Mean Squared Error Score:** {mse:.2f}")
    st.write(f"**Root Mean Squared Error Score:** {rmse:.2f}")

    residuals = y_test - y_pred
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax3)
    ax3.axhline(0, color="red", linestyle="--")
    ax3.set_xlabel("Predicted values")
    ax3.set_ylabel("Residuals")
    ax3.set_title("Residual plot")
    st.pyplot(fig3)

    if hasattr(model, "coef_"):
        importances = model.coef_
        feature_names = st.session_state["features"]
        fi_df = pd.DataFrame({
            "Feature" : feature_names,
            "Coefficient": importances
        }).sort_values(by="Coefficient", ascending=False)

        fig4 = px.bar(fi_df, x="Feature", y="Coefficient", title="Feature Coefficients")
        st.plotly_chart(fig4)