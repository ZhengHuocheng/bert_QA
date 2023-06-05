import streamlit as st
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
# 加载问答模型
model = QuestionAnsweringModel("bert", ".\QA_model\QA",use_cuda=False)
st.title("问答系统")
st.write("输入一个问题和一段文本，获取问题的答案")
# 用户输入问题和文本
question = st.text_input("请输入你的问题")
text = st.text_area("请输入文本")
to_predict = [
    {
        "context": text,
        "qas": [
            {
                "question": question,
                "id": "0",
            }
        ],
    }
]
# 提交按钮
if st.button("提交"):
    answers, probabilities = model.predict(to_predict)
    # 对输入的问题和文本进行预测
    # 获取最大概率对应的答案
    max_prob_index = probabilities[0]['probability'].index(max(probabilities[0]['probability']))
    max_prob_answer = answers[0]['answer'][max_prob_index]
    max_prob_probability = probabilities[0]['probability'][max_prob_index]
    # 显示答案
    #st.write("问题：", text)
    st.write("答案：", max_prob_answer)
    st.write("置信度：", max_prob_probability)