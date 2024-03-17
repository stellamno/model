#%%
import streamlit as st
import pandas as pd
import joblib

# Set the title of the app
st.title('癫痫药效预测应用')

# Set default values for the example
default_values = {
    '性别': 0,
    '首次癫痫发作年龄': 0.535484,
    '发作频率': 0.001855,
    '首次癫痫发作到首次口服抗癫痫药物时间': 0.005952,
    '病因分类': 0,
    '基因结果有无异常': 0,
    '是否热性惊厥史': 0,
    '生产方式': 0,
    '是否围产期异常': 0,
    '是否家族史异常': 0,
    '首次癫痫发作形式_1': 0,
    '首次癫痫发作形式_2': 1,
    '治疗前脑电图放电部位_2.0': 0,
    '治疗前脑电图放电部位_3.0': 1,
    '治疗前脑电图放电部位_4.0': 0,
    '治疗前脑电图放电部位_5.0': 0
}

# 在Streamlit应用中使用这些默认值
sex = st.selectbox("性别 (0: 女, 1: 男)", [0, 1], index=default_values['性别'])
age_of_onset = st.number_input("首次癫痫发作年龄", value=default_values['首次癫痫发作年龄'])
seizure_frequency = st.number_input("发作频率", value=default_values['发作频率'])
time_to_first_treatment = st.number_input("首次癫痫发作到首次口服抗癫痫药物时间", value=default_values['首次癫痫发作到首次口服抗癫痫药物时间'])
etiology = st.selectbox("病因分类 (0: 无, 1: 有)", [0, 1], index=default_values['病因分类'])
genetic_results = st.selectbox("基因结果有无异常 (0: 无, 1: 有)", [0, 1], index=default_values['基因结果有无异常'])
febrile_seizure_history = st.selectbox("是否热性惊厥史 (0: 无, 1: 有)", [0, 1], index=default_values['是否热性惊厥史'])
birth_method = st.selectbox("生产方式 (0: 非顺产, 1: 顺产)", [0, 1], index=default_values['生产方式'])
perinatal_complications = st.selectbox("是否围产期异常 (0: 无, 1: 有)", [0, 1], index=default_values['是否围产期异常'])
family_history = st.selectbox("是否家族史异常 (0: 无, 1: 有)", [0, 1], index=default_values['是否家族史异常'])
first_seizure_type_1 = st.selectbox("首次癫痫发作形式_1 (0: 其他, 1: 局灶)", [0, 1], index=default_values['首次癫痫发作形式_1'])
first_seizure_type_2 = st.selectbox("首次癫痫发作形式_2 (0: 其他, 1: 全面)", [0, 1], index=default_values['首次癫痫发作形式_2'])
eeg_2 = st.selectbox("治疗前脑电图放电部位_2.0 (0: 无, 1: 有)", [0, 1], index=default_values['治疗前脑电图放电部位_2.0'])
eeg_3 = st.selectbox("治疗前脑电图放电部位_3.0 (0: 无, 1: 有)", [0, 1], index=default_values['治疗前脑电图放电部位_3.0'])
eeg_4 = st.selectbox("治疗前脑电图放电部位_4.0 (0: 无, 1: 有)", [0, 1], index=default_values['治疗前脑电图放电部位_4.0'])
eeg_5 = st.selectbox("治疗前脑电图放电部位_5.0 (0: 无, 1: 有)", [0, 1], index=default_values['治疗前脑电图放电部位_5.0'])

# 按钮提交预测
if st.button("Submit"):
    # 存储所有输入数据
    input_data = pd.DataFrame([[sex, age_of_onset, seizure_frequency, time_to_first_treatment, etiology,
                                genetic_results, febrile_seizure_history, birth_method, perinatal_complications,
                                family_history, first_seizure_type_1, first_seizure_type_2, eeg_2, eeg_3, eeg_4,
                                eeg_5]],
                              columns=['性别', '首次癫痫发作年龄', '发作频率', '首次癫痫发作到首次口服抗癫痫药物时间', '病因分类', '基因结果有无异常', '是否热性惊厥史',
                                       '生产方式', '是否围产期异常', '是否家族史异常', '首次癫痫发作形式_1', '首次癫痫发作形式_2', '治疗前脑电图放电部位_2.0',
                                       '治疗前脑电图放电部位_3.0', '治疗前脑电图放电部位_4.0', '治疗前脑电图放电部位_5.0'])

    # 加载模型并进行预测
    model = joblib.load("voting_clf.pkl")  # 确保模型文件路径正确
    prediction = model.predict(input_data)[0]

    # 显示预测结果
    st.write(f"预测结果: {'有效' if prediction == 1 else '无效'}")
