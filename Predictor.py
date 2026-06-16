import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
import shap

from lime.lime_tabular import LimeTabularExplainer
#pip install xgboost==2.0.3 --no-deps
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
df2 =pd.read_csv('stroke-TNY-x_test.csv')
x_test = df2[['通脑饮', 'LAA', 'SAO', '年龄', '高血压', '吸烟史', '入院NIHSS', '入院mRS', 'NLR', '淋巴细胞', '尿素', '抗聚', '降压', '调脂']]

model = joblib.load('stroke-TNY.pkl')

feature_names = ['TNY', 'LAA', 'SAO', 'AGE', 'Hypertension', 'Smoke', 'admissionNIHSS', 'admissionmRS', 'NLR', 'L', 'Bun', 'antiaggregation', 'antihypertensive', 'lipid_lowering']
    
    
# 设置 Streamlit 应用的标题
st.title("卒中诊断模型")
st.sidebar.header("Selection Panel") # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")
TNY = st.selectbox("通脑饮", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")
LAA = st.selectbox("LAA", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")
SAO = st.selectbox("SAO", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")

AGE = st.number_input("年龄", min_value=0, max_value=120, value=1)

Hypertension = st.selectbox("高血压", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")
Smoke = st.selectbox("吸烟史", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")

NIHSS = st.number_input("入院NIHSS", min_value=0, max_value=42, value=0, step=1)
mRS = st.number_input("入院mRS", min_value=0, max_value=42, value=0, step=1)
N = st.number_input("中性粒细胞", min_value=0.01, max_value=50.00, value=0.01)
L = st.number_input("淋巴细胞", min_value=0.01, max_value=50.00, value=0.01)
NLR = N/L
Bun = st.number_input("尿素", min_value=0, max_value=50, value=1)

antiaggregation = st.selectbox("抗聚史", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")
antihypertensive = st.selectbox("降压史", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")
lipid_lowering = st.selectbox("调脂史", options=[0, 1], format_func=lambda x:"是"if x == 1 else "否")






feature_values = [TNY, LAA, SAO, AGE, Hypertension, Smoke, NIHSS, mRS, NLR, L, Bun, antiaggregation, antihypertensive, lipid_lowering]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: 不良结局低风险, 1: 不良结局高风险)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice =(
            f"根据我们的模型，该患者本次卒中预后不良的风险较高。 "
            f"具体预后不良的可能性为 {probability:.1f}%，"
            "建议进一步评估该患者的风险因素，针对性加强预防与治疗干预措施。"
        )

    # 如果预测类别为0（低风险）
    else:
        advice =(
            f"根据我们的模型，该患者本次卒中预后良好的可能较高。"
            f"具体预后良好的可能性为 {probability:.1f}%，"
            "但继续保持健康的生活习惯仍是必要的，请定期至医院体检并规律服用药物。"
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP 力图解释")
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
