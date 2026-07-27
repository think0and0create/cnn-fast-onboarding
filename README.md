# 项目驱动的 CNN 速成课

一门只讲卷积神经网络（CNN）的项目驱动课。主线：图像分类 → 目标检测 → 语义分割 → 迁移到自己的数据。

## 章节进度

| # | 章节 | 状态 | 链接 |
|---|---|---|---|
| 1 | Python 光速扫盲 | 已发布 | [article.md](articles/01-python-crash-course/article.md) |
| 2 | 线性代数扫盲 (NumPy) | 已发布 | [article.md](articles/02-linear-algebra-numpy/article.md) |
| 3 | 最优化扫盲 (PyTorch autograd) | 已发布 | [article.md](articles/03-optimization-torch/article.md) |
| 4 | 优化的简单应用 | 已发布 | [article.md](articles/04-optimization-applications/article.md) |

## 本地预览

```bash
npm install
npm run docs:dev   # http://localhost:8080
```

## 目录结构

```
├── articles/          # 每章一个子目录（核心内容）
│   └── <chNN>-<slug>/
│       ├── article.md # 章节正文
│       ├── code/      # 可运行代码
│       └── tests/     # 测试（不入库）
├── docs/              # VuePress 站点
│   └── .vuepress/     # VuePress 配置
└── scripts/           # 构建脚本
```

## 数据约定

数据不入库。每章脚本找不到数据时会报错，提示运行 `code/download_data.py`。真实数据放到 `data/<project_name>/raw/`。

## 反馈

- 教程问题 → 在该章文章下方留 issue
- 范围疑问 → issue 里说明
