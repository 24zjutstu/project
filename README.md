# project

# 多因子量化选股策略

## 项目简介
基于 A 股市场的多因子量化选股策略，包含因子构建、IC 检验、分层回测和绩效评估。

## 技术栈
- Python, Pandas, NumPy
- 数据源：akshare
- 因子：动量、RSI、波动率、换手率、成交量趋势
- 回测：分层回测 + IC 分析

## 因子构建流程
1. 数据清洗（去重、异常值处理、缺失值填充）
2. 单因子计算（截面标准化）
3. 复合因子（IC 滚动加权）
4. 分层回测（4 层多空）
5. IC 检验与绩效评估

## 运行方式
```bash
pip install -r requirements.txt
python main.py
