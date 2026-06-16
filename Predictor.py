import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

#df = pd.read_csv('x_test.csv')
#x_test = df[['OCSP', 'NLR', 'epwv', 'admissionNHISS']]

model = joblib.load('AIS-Intravenous-Thrombolysis.pkl')

feature_names = ['OCSP', 'NLR', 'epwv', 'admissionNHISS']
    
# 设置 Streamlit 应用的标题
st.title("Prospective study with a 90-day follow-up")
st.sidebar.header("Selection Panel") # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")

admissionNHISS = st.number_input("admissionNHISS", min_value=0, max_value=42, value=1)
#admissionNHISS = st.sidebar.slider("admissionNHISS", min_value=0, max_value=42, value=0, step=1)
age = st.number_input("age", min_value=0, max_value=120, value=1)
SBP = st.number_input("SBP", min_value=0, max_value=300, value=1)
DBP = st.number_input("DBP", min_value=0, max_value=300, value=1)
N = st.number_input("N", min_value=0.0, max_value=50.0, value=0.01, step=0.01)
L = st.number_input("L", min_value=0.0, max_value=50.0, value=0.01, step=0.01)

OCSP = st.selectbox(
    "OCSP",                      # 显示标签
    ["TACI", "PACI", "POCI", "LACI"],    # 选项列表
    index=0                              # 默认选中第一个（可选）
)
ocsp_mapping = {"TACI": 1, "PACI": 2, "POCI": 3, "LACI": 4}
OCSP = ocsp_mapping[OCSP]
MBP = DBP + 0.4*(SBP-DBP)
epwv = 9.587-0.402*age+4.56*0.001*age*age-2.621*0.00001*age*age*MBP+3.176*0.001*age*MBP-1.832*0.01*MBP
NLR = N/L







feature_values = [OCSP, NLR, epwv, admissionNHISS]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: Good Prognosis, 1: Bad Prognosis)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice =(
            f"According to our model, you have a high risk of Bad Prognosis.  "
            f"The model predicts that your probability of having Bad Prognosis is {probability:.1f}%，"
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice =(
            f"According to our model, you have a low risk of Bad Prognosis."
            f"The model predicts that your probability of not having Good Prognosis is {probability:.1f}%，"
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    explainer = shap.TreeExplainer(model)

    # 将当前输入转换为 DataFrame（保留特征名）
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # 计算当前样本的 SHAP 值
    shap_values_sample = explainer.shap_values(input_df)
    
    # shap_values_sample 的常见形状：
    # - 对于二分类 XGBoost/RandomForest：list，长度为2，每个元素 shape (1, n_features)
    # - 或者直接是 array，shape (1, n_features, 2)
    if isinstance(shap_values_sample, list):
        # 根据预测类别取出对应类别的 SHAP 值（一维数组）
        shap_values_for_class = shap_values_sample[predicted_class][0]  # [0] 去掉样本维度
    else:
        # 假设 shape (1, n_features, n_classes)
        shap_values_for_class = shap_values_sample[0, :, predicted_class]
    
    # 获取对应类别的期望值（基线值）
    expected_value_class = explainer.expected_value[predicted_class]
    
    # 绘制力图
    shap.force_plot(
        expected_value_class,
        shap_values_for_class,
        input_df,
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
    st.subheader("LIME Explanation")
    
    lime_explainer = LimeTabularExplainer(
        training_data=x_test.values, 
        feature_names=x_test.columns.tolist(),
        class_names=['Good Prognosis', 'Bad Prognosis'],# Adjust class names to match your classification task
        mode='classification'
    )

    #Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=13
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=True) # Disable feature value table
    st.components.v1.html(lime_html, height=800,scrolling=True)
