---
name: minimal-coder
description: |
  编写简洁、规范代码的核心原则与方法论。
  
  使用场景：
  1. 编写新功能或修改现有代码时
  2. 需要分析和解决代码问题时
  3. 代码重构或简化复杂逻辑时
  4. 任何需要产出高质量代码的任务
  
  核心理念：先精准定位问题本质，再用最简洁直接的方式实现，避免过度工程化和防御性编程。
---

# Minimal Coder - 简洁编码指南

## 编码前的思考：精准定位问题

在写任何代码之前，必须先完成以下思考：

### 1. 问题本质分析

- **问题是什么？** - 用一句话清晰描述
- **为什么需要解决？** - 理解业务/技术价值
- **最小解决范围是什么？** - 砍掉所有非核心需求
- **是否有更简单的替代方案？** - 考虑配置、约定、工具替代代码

### 2. 输入输出定义

```
输入：[明确列出所有输入及其类型/格式]
输出：[明确期望的输出及其格式]
约束：[硬约束条件，如性能、兼容性等]
```

### 3. 方案选择原则

按优先级选择实现方式：
1. **无代码方案** > 配置 > 约定 > 代码
2. **标准库** > 成熟第三方库 > 自研
3. **一行代码** > 一个函数 > 一个类
4. **直接逻辑** > 抽象封装

---

## 编码原则：简洁至上

### 1. 变量与命名

- **命名要直白**：用 `data` 而不是 `processedData`，用 `name` 而不是 `userName`
- **类型即文档**：`users: list[str]` 比 `userList` 更清晰
- **避免匈牙利命名**：不要用 `strName`、`bIsValid` 等前缀
- **短作用域用短名**：循环变量用 `i`、`v`、`k`；临时变量用 `x`、`tmp`

```python
# 好
def process(users: list[str]) -> dict[str, int]:
    result = {}
    for u in users:
        result[u] = len(u)
    return result

# 不好
def processUserNameList(userNameList: list) -> dict:
    processedUserNameLengthDictionary = {}
    for userNameString in userNameList:
        processedUserNameLengthDictionary[userNameString] = len(userNameString)
    return processedUserNameLengthDictionary
```

### 2. 控制逻辑简化

**核心原则：能不用就不用**

#### 避免不必要的分支

```python
# 避免：用条件判断处理简单情况
if x > 0:
    sign = 1
elif x < 0:
    sign = -1
else:
    sign = 0

# 更好：直接用数学运算
sign = (x > 0) - (x < 0)
```

```python
# 避免：防御性判断
if data is not None:
    process(data)

# 更好：让调用方保证输入有效
process(data)  # 假设调用方已确保 data 不为 None
```

#### 避免不必要的循环

```python
# 避免：显式循环构建列表
result = []
for x in items:
    result.append(x * 2)

# 更好：推导式
result = [x * 2 for x in items]
```

```python
# 避免：循环查找
for item in items:
    if item.id == target_id:
        return item

# 更好：用数据结构
item_map = {item.id: item for item in items}
return item_map.get(target_id)
```

### 3. 禁用 try-except Fallback 逻辑

**绝不使用 try-except 来隐藏错误或提供默认行为。**

```python
# 严禁
try:
    value = int(user_input)
except:
    value = 0  # 隐藏错误，静默失败

# 正确做法 1：让错误暴露
value = int(user_input)  # 如果输入无效，直接抛出 ValueError

# 正确做法 2：前置验证
if user_input.isdigit():
    value = int(user_input)
else:
    raise ValueError(f"Invalid input: {user_input}")
```

**何时可以用 try-except：**
- 仅用于资源清理（`finally` 块）
- 仅用于捕获特定异常并重新抛出带上下文信息的异常
- 绝对不用 except 空捕获或返回默认值

### 4. 函数设计

#### 单一职责，直接实现

```python
# 好：直接实现，无过度封装
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

# 不好：过度设计
def read_file_advanced(
    path: str,
    encoding: str = 'utf-8',
    fallback_encoding: str = 'latin-1',
    max_size: int = None,
    error_handler: callable = None
) -> str:
    try:
        with open(path, encoding=encoding) as f:
            content = f.read()
            if max_size and len(content) > max_size:
                if error_handler:
                    return error_handler()
                raise IOError("File too large")
            return content
    except UnicodeDecodeError:
        with open(path, encoding=fallback_encoding) as f:
            return f.read()
```

#### 参数传递要简单

```python
# 好：直接传值
def greet(name: str) -> str:
    return f"Hello, {name}"

# 不好：包装成对象/字典
def greet(params: dict) -> str:
    name = params.get('name', 'Guest')
    return f"Hello, {name}"
```

### 5. 数据结构选择

- **能用基础类型就不用对象**：`list[str]` > `class UserNames`
- **能用元组就不用类**：`(name, age)` 比 `class Person` 简单
- **能用集合就不用列表判断**：`if x in valid_set` 比 `if x in valid_list` 高效
- **能用字典就不用 if-elif**：`action_map[action]()` 比多个 if-elif 清晰

```python
# 避免：if-elif 链
if action == 'create':
    create()
elif action == 'update':
    update()
elif action == 'delete':
    delete()

# 更好：字典映射
actions = {'create': create, 'update': update, 'delete': delete}
actions[action]()
```

---

## 代码风格规范

### Python

```python
# 1. 类型注解必须简洁
from typing import Optional  # 避免过度导入

def func(x: int) -> int:     # 好
def func(x: int | None) -> int | None:  # 3.10+ 用 | 替代 Optional/Union

# 2. 一行不超过 88 字符（Black 默认）
# 3. 函数不超过 20 行
# 4. 模块不超过 200 行
# 5. 命名：函数/变量 snake_case，类 CamelCase，常量 UPPER_SNAKE_CASE
```

### JavaScript/TypeScript

```javascript
// 1. 用 const/let，不用 var
const data = fetchData();

// 2. 箭头函数保持简洁
const double = x => x * 2;

// 3. 解构简化参数
const process = ({name, age}) => `${name} is ${age}`;

// 4. 不用 try-catch 做默认值
// 严禁：
try { JSON.parse(str); } catch { return {}; }
// 正确：JSON.parse(str) 或先验证 isValidJSON(str)
```

---

## 重构检查清单

完成代码后，问自己：

1. **能否再删 20%？** - 去掉注释掉的代码、未使用的参数、过度抽象
2. **是否有 try-except 隐藏错误？** - 移除或改为显式处理
3. **是否有循环可以用推导式/内置函数替代？** - map/filter/sum/any/all
4. **是否有 if-elif 链可以用字典替代？**
5. **函数参数是否超过 3 个？** - 考虑拆分或简化
6. **是否引入了不必要的类？** - 函数 + 基础数据结构通常足够
7. **变量名是否过度冗长？** - 上下文足够时，用短名

---

## 示例：完整重构流程

### 需求：计算用户平均分数

**原始代码（过度工程）：**

```python
class UserScoreCalculator:
    def __init__(self, scores: list, logger=None):
        self.scores = scores
        self.logger = logger or default_logger
    
    def calculate_average(self, fallback_value=0.0):
        try:
            if not self.scores:
                self.logger.warning("Empty scores")
                return fallback_value
            
            valid_scores = []
            for s in self.scores:
                try:
                    if isinstance(s, (int, float)) and s >= 0:
                        valid_scores.append(s)
                except:
                    continue
            
            if not valid_scores:
                return fallback_value
            
            total = sum(valid_scores)
            count = len(valid_scores)
            return total / count
        except Exception as e:
            self.logger.error(f"Calculation failed: {e}")
            return fallback_value
```

**重构后（简洁直接）：**

```python
def average(scores: list[float]) -> float:
    """计算平均分。scores 必须为非空列表。"""
    return sum(scores) / len(scores)
```

**关键改进：**
1. 去掉了类封装，改为纯函数
2. 去掉了 try-except fallback，改为前置约束（调用方确保输入有效）
3. 去掉了空列表检查，约束由文档说明
4. 去掉了日志，错误应向上传播而非静默处理
5. 去掉了数据清洗，假设调用方已提供干净数据

---

## 反模式速查

遇到以下写法，立即重构：

| 反模式 | 问题 | 修正 |
|--------|------|------|
| `try: ... except: pass` | 静默吞掉错误 | 移除或显式处理 |
| `try: ... except: return default` | 隐藏错误状态 | 前置验证或抛出 |
| `if x: return A else: return B` | 冗长 | `return A if x else B` |
| `for x in items: if cond: return x` | 线性查找 | 用 `next()` 或数据结构 |
| `class X: def process(self, data)` | 无状态类 | 改为纯函数 |
| `def func(params: dict)` | 参数不透明 | 显式列出参数 |
| `userNameList` | 匈牙利命名 | `users` |
| `result_list = []; for x in items: result_list.append(...)` | 冗余循环 | 推导式 |

---

## 执行原则

1. **先想后写**：在脑中或纸上完成设计，确认最简方案后再编码
2. **一次只做一件事**：一个函数一个职责，一个提交一个功能
3. **删除大于添加**：每添加一行代码，思考能否用更少行实现
4. **暴露问题而非隐藏**：错误应清晰可见，而不是被 try-except 掩盖
5. **信任调用方**：不做防御性编程，通过文档约定前置条件
