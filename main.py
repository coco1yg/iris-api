from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 创建FastAPI对象
app = FastAPI()

# 加载模型
model = joblib.load("iris_model.pkl")

# 定义输入数据格式
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 首页
@app.get("/")
def home():
    return {"message": "Iris API Running"}

# 预测接口
@app.post("/predict")
def predict(data: IrisData):

    result = model.predict([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])[0]

    return {
        "prediction": result
    }