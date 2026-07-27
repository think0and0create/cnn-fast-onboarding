---
home: true
title: CNN 速成课
heroText: 项目驱动的卷积神经网络入门
tagline: 用 4 个项目，把 PyTorch 与优化理论从「会用」推进到「能讲清楚」
actions:
  - text: 开始 Ch 1
    link: /ch1/
    type: primary
  - text: 查看 GitHub
    link: https://github.com/
    type: secondary
features:
  - title: 项目驱动
    details: 不堆概念，每个 demo 是一个端到端项目，从 issue 到 method 到 evaluation 的完整闭环
  - title: 真实数字
    details: 跑出真数字、画真图、算真损失，不写「看起来不错」的模糊结论
  - title: 设计可迁移
    details: 掌握「模型怎么参数化 + 目标函数怎么设计」两问，后续 CNN 与更大模型是同一思路的实例
---

## 章节列表

- [Ch 1 · Python 光速扫盲](/ch1/)：基础语法、类型注解、uv 工具链
- [Ch 2 · 线性代数扫盲](/ch2/)：向量、矩阵、SVD 几何意义
- [Ch 3 · 最优化扫盲](/ch3/)：梯度下降、牛顿法、KKT、对偶
- [Ch 4 · 优化的简单应用](/ch4/)：OLS / Ridge / Lasso / LogReg / Softmax / SVM / 速度规划

## 阅读方式

每章 1-2 小时。按顺序读。每章末尾会明确暴露下章 Issue。

代码与测试在 `articles/<ch>/code/` 与 `articles/<ch>/tests/` 下，本地克隆后可 `uv sync && pytest` 跑通。
