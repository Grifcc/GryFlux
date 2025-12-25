# GryFlux Framework - Parallel Calc Test

## 1. 测试目的

本目录提供一个 **可验证正确性**、同时能 **施压资源调度/并发执行** 的框架测试用例：`example`。

它重点覆盖：

- **DAG 依赖调度**：节点完成后事件驱动触发后继节点
- **资源池与并发限制**：通过资源类型限制某类节点并发度（这里刻意让 `multiplier` 更稀缺）
- **背压**：限制同时在途的数据包数量（`maxActivePackets`）
- **正确性校验**：对每个数据包计算一个可推导的闭式结果，统计 pass/fail
- **Profiling（可选）**：导出每个节点的耗时统计与时间线 JSON

> 入口代码在 `example.cpp`。

---

## 2. 计算图（Compute Graph）

### 2.1 节点与依赖

该测试构建一个 **5 层 / 8 节点** 的 DAG（包含输入与输出节点）：

- 节点：`x, y, z, a, b, c, d, f`
- 依赖：
  - `x` 为输入节点（由 packet id 产生）
  - `y <- x`
  - `z <- x`
  - `a <- y`
  - `b <- y`
  - `c <- (y, z)`
  - `d <- (a, b, c)`
  - `f <- (d, z)`（输出节点）

### 2.2 图示（清晰版）

```
Layer 1
                 x (InputNode)
                    │
   ├────────────────│───────┐
   │                        │
Layer 2                     │
  y (AddConst)          z (MulConst)  ─────────────┐
  [resource=adder]      [resource=multiplier]      │
   │                         │                     │
   ├────────────────┐────────│──────┐              │
   │                │               │              │
Layer 3             │               │              │
  a(Add+10)       b(Mul*3)│       c(Add2)          │
  [adder]         [multiplier]    [adder]          │
   │                │               │              │ 
   └────────────────│───────────────│              │
                    │                              │
Layer 4             │                              │ 
                 d(Fuse3Sum)                       │
                 [adder]                           │
                    │                              │
                    │                              │
Layer 5             │                              │
                 f (OutputNode)────────────────────│ 
```

### 2.3 资源类型（ResourcePool）

测试中注册：

- `adder`: 2 个实例（并发度 2）
- `multiplier`: 1 个实例（并发度 1，刻意制造瓶颈以观察调度）

这意味着：

- 所有 `multiplier` 节点（这里是 `z`、`b`）在任意时刻最多只能 **同时执行 1 个**
- `adder` 节点（`y/a/c/d`）最多 **同时执行 2 个**

---

## 3. 可验证的计算公式（Correctness）

每个数据包以 `id` 为输入，按以下公式逐步计算：

- `x = id`
- `y = x + 1`
- `z = x * 2`
- `a = y + 10`
- `b = y * 3`
- `c = y + z`
- `d = a + b + c`
- `f = d + z`

可推导闭式结果：

- `f = 9 * id + 15`

`example` 会在 consumer 中对每个包进行校验并统计：

- `pass`：结果等于期望
- `fail`：结果不等于期望（返回码会非 0）

---

## 3.1 数据包字段的提前开辟（避免运行时 alloc）

为了模拟更接近真实场景的“张量/特征向量”数据流，并减少节点执行过程中的动态分配开销，本示例的 `CalcPacket` 将每一层的输出从标量改为 **预先分配好的 `std::vector<double>`**：

- `x/y/z/a/b/c/d/f` 均为 `std::vector<double>`
- `CalcPacket` 构造时会对这些 vector 做 `resize`（一次性开辟固定长度），节点执行阶段只做逐元素读写，不再触发扩容分配

默认预分配长度在 `CalcPacket::kDefaultVectorSize`：

- 代码位置：src/app/example/packet/calc_packet.h

如果你希望按业务需求调整每个包的向量长度，有两种简单方式：

1) 直接修改 `kDefaultVectorSize`
2) 在 source 侧用 `CalcPacket(size_t vectorSize)` 构造（例如在 `CalcSource::produce()` 里改成 `std::make_unique<CalcPacket>(yourSize)`）

---

## 4. 编译与运行


在项目根目录：

```bash
mkdir -p build
cd build
cmake .. 
make -j8

# 运行（从 build 目录运行，二进制输出在 build/src/app/example/ 下）
./src/app/example/example --num-packets 500 --mid-sleep-ms 500 --resource-timeout-ms 10000
```

---

## 4.1 如何扩展：定义节点 / 资源（不展开代码）

如果你要在本测试基础上新增节点或资源，建议按下面的“最短路径”走：

1) **定义数据包字段（DataPacket）**

- 入口：`packet/calc_packet.h`
- 原则：字段按节点输出拆分、每个字段尽量由单一节点写入（降低并行写冲突风险）

2) **定义节点（Node）**

- 入口：`nodes/*/*.h` 与 `nodes/*/*.cpp`
- 节点统一继承 `GryFlux::NodeBase`，在 `execute(packet, ctx)` 里读取输入字段并写入输出字段

3) **定义资源（Context）**

- 入口：`context/*.h` 与 `context/*.cpp`
- 资源统一继承 `GryFlux::Context`，将“可限流/可独占”的能力封装成一个 context 实例

4) **注册资源池（ResourcePool）并绑定到节点**

- 注册位置：`example.cpp`（注册 `adder`/`multiplier` 的实例数量决定并发度）
- 绑定位置：同样在 `example.cpp` 的 `GraphTemplate::buildOnce(...)` 内
- 绑定规则：在 `TemplateBuilder::addTask(...)` 中为节点指定 `resourceType`（例如 `adder` 或 `multiplier`），框架会从资源池借用对应的 context 传入该节点执行

5) **更新计算图与正确性公式**

- 计算图依赖关系：建议同步更新本 README 的“计算图图示”
- 正确性校验：确保能推导出期望结果（或者至少能在 consumer 里稳定验证）


## 5. 常用参数说明（CLI）

运行 `--help` 可查看完整参数。

- `--num-packets N`：处理数据包数量（默认 200）
- `--threads N`：线程池大小（默认 12）
- `--max-active N`：最大在途数据包数（默认 0=自动）
- `--serial`：强制串行（等价 `--threads 1 --max-active 1`）
- `--mid-sleep-ms N` / `--mid-sleep-s S`：为中间节点（y/z/a/b/c/d）附加 sleep，用于放大调度效果
- `--resource-timeout-ms N`：资源获取超时（0=无限等待）
- `--adder-timeout-ms N`：`adder` 资源获取超时（默认继承 `--resource-timeout-ms`）
- `--multiplier-timeout-ms N`：`multiplier` 资源获取超时（默认继承 `--resource-timeout-ms`）
- `--print-first N`：打印前 N 个包的实际结果 vs 期望
- `--print-all`：打印所有包的结果 vs 期望

### 5.1 推荐的几组跑法

1) **快速正确性**：
```bash
./src/app/example/example --num-packets 200 --print-first 5
```

2) **资源调度压力**（放大并发与资源竞争）：
```bash
./src/app/example/example --num-packets 500 --mid-sleep-ms 500 --resource-timeout-ms 10000
```

3) **串行对照**（验证串并行一致性）：
```bash
./src/app/example/example --num-packets 200 --serial
```

4) **触发超时路径（验证不会死锁）**：
```bash
./src/app/example/example --num-packets 200 --mid-sleep-ms 500 --resource-timeout-ms 1
```

5) **只让某类资源更容易超时**（更精确地观察调度行为）：
```bash
# multiplier 更紧（更容易超时），adder 更松
./src/app/example/example --num-packets 200 --mid-sleep-ms 500 --adder-timeout-ms 10000 --multiplier-timeout-ms 1
```

---

## 6. Profiling：统计与时间线导出

### 6.1 如何启用 Profiling（编译期开关）

Profiling 为编译期开关：需要在编译时添加 `--enable_profile`。

```bash
bash build.sh --enable_profile
./install/bin/example --num-packets 500 --mid-sleep-ms 500 --resource-timeout-ms 10000
```
Tips：建议用新 build 目录避免缓存混编


### 6.2 运行后会得到什么

启用 profiling 后，程序会额外输出：

- **每个节点的统计摘要**（scheduled/started/finished/avg/total 等）
- **时间线文件**：`parallel_calc_timeline.json`

JSON 文件会输出到你运行程序时的当前工作目录。

### 6.3 可视化时间线

仓库提供了在线Viewer：

- 打开：http://profile.grifcc.top:8076/
- 上传： parallel_calc_timeline.json

上传json文件后即可直观看到：

- `multiplier` 节点（`z/b`）是否被串行化
- 各层节点的并发执行与依赖触发是否符合预期

---

## 7. 结果判定与排错

- 关注程序末尾：
  - `Verification: pass=... fail=...`
  - 退出码：`0` 表示全部通过，非 0 表示存在 fail

常见排错建议：

1) **出现 fail**：优先用 `--print-first 20` 定位是哪几个 id 不符合 `f = 9*id + 15`。

2) **出现卡住**：尝试减小并发与在途数：

```bash
./src/app/example/example --num-packets 200 --threads 2 --max-active 2 --mid-sleep-ms 500
```

3) **切换 profiling 开关后异常**：建议使用全新 build 目录（例如 `build_profile`），避免旧对象文件混编。

---

## 8. 文件结构速览

- `example.cpp`：构图、资源注册、参数解析、运行与统计输出
- `nodes/`：算子节点实现（add/mul/input/output）
- `context/`：资源上下文（adder/multiplier）
- `packet/`：数据包定义（CalcPacket）
- `source/` / `consumer/`：数据源与结果校验
