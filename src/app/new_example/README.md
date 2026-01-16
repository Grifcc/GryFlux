# GryFlux Framework - new_example

## 示例说明

本示例用于演示：

- 如何在 GryFlux 中构建自定义 DAG（5 层 9 节点：a-i）
- 如何使用资源池抽象“加法器/乘法器”这种受限资源
- 如何通过模拟耗时与 profiling 观察吞吐瓶颈
- 如何在节点中注入异常，观察框架的错误处理

## DAG 结构

节点依赖关系：

```
Layer1:  a
Layer2:  b, c, d      (b,c,d <- a)
Layer3:  e, f, g      (e <- b, f <- b, g <- b,c,d)
Layer4:  h            (h <- e,f,g)
Layer5:  i (output)   (i <- h,c)
```

当前资源绑定（以代码为准）：

- `multiplier`：b、c、f
- `adder`：g、h
- CPU（无资源）：a、d、e、i

## 模拟耗时（可用于制造瓶颈）

在 [src/app/new_example/nodes/dag/dag_nodes.cpp](src/app/new_example/nodes/dag/dag_nodes.cpp) 中通过 `sleep_for` 模拟耗时：

- `kCpuNodeDelay`
- `kAdderNodeDelay`
- `kMultiplierNodeDelay`

你可以通过调大 `kCpuNodeDelay` 制造 CPU/线程池瓶颈，或调大 `kMultiplierNodeDelay` 制造乘法器资源瓶颈。

## Profiling 编译与运行

profiling 是“编译期开关 + 运行时启用”两段式：

1) 编译期开关：编译时定义 `GRYFLUX_BUILD_PROFILING=1`
2) 运行时启用：示例中在 `kBuildProfiling` 为 true 时调用 `pipeline.setProfilingEnabled(true)`

推荐使用脚本编译并安装：

```bash
cd /workspace/gxh/GryFlux
bash ./build.sh --enable_profile
```

运行：

```bash
./install/bin/new_example
```

运行结束会输出 profiling 统计，并生成 `graph_timeline.json`（用于可视化）。

## 异常注入：id 为 10 的倍数触发

为了观察框架如何处理节点执行异常：

- 在节点 `g` 内部，当 `packet.id % 10 == 0` 时会执行一次“除 0”，并抛出 `std::runtime_error`
- 框架会捕获异常（见 TaskScheduler 的 `try/catch`），记录错误日志并对该 packet 执行 `markFailed()`
- 即使 packet 失败，框架仍会推进 DAG 完成收尾（释放资源、触发 output）

## 为什么日志不是按 id 顺序输出

这是正常现象：pipeline 为并发异步执行，consumer 日志反映“哪个 packet 先完成/先被消费”，不保证按 id 递增。

如果希望严格按 id 输出，需要在 consumer 中做“按 id 缓冲并顺序 flush”，或将并发度降为 1。

## 吞吐量分析：不同 delay 下瓶颈是谁？

本示例通过三个 delay 来模拟计算耗时（见 [src/app/new_example/nodes/dag/dag_nodes.cpp](src/app/new_example/nodes/dag/dag_nodes.cpp)）：

- `kCpuNodeDelay = Tc`
- `kAdderNodeDelay = Ta`
- `kMultiplierNodeDelay = Tm`

当前 DAG 资源绑定（以 [src/app/new_example/new_example.cpp](src/app/new_example/new_example.cpp) 为准）：

- CPU 节点：a、d、e、i（共 4 个）
- `adder` 节点：g、h（共 2 个）
- `multiplier` 节点：b、c、f（共 3 个）

并发能力：

- `adder` 实例数：`m_add`（在 `resourcePool->registerResourceType("adder", ...)` 里）
- `multiplier` 实例数：`m_mul`
- 线程池大小：`P`（在 `AsyncPipeline(..., P)` 里）

### 1) 理论最大吞吐量（上界）怎么算

对每一种“有限服务能力” $r$（线程池、adder 资源池、multiplier 资源池、数据源/消费者等），计算每个 packet 在该能力上的总服务需求 $D_r$，吞吐上界为：

$$X_r \le \frac{m_r}{D_r}$$

整体理论吞吐上界通常取最小者：

$$X_{max} \approx \min_r \frac{m_r}{D_r}$$

把它套到本例（按当前绑定关系）：

- multiplier：$D_{mul}=3\,Tm$，$m_{mul}=m\_mul$

$$X_{mul}\le\frac{m\_mul}{3\,Tm}$$

- adder：$D_{add}=2\,Ta$，$m_{add}=m\_add$

$$X_{add}\le\frac{m\_add}{2\,Ta}$$

- 线程池（所有节点最终都要占用线程执行）：

$$D_{cpu}=4\,Tc + 2\,Ta + 3\,Tm$$
$$X_{cpu}\le\frac{P}{4\,Tc + 2\,Ta + 3\,Tm}$$

因此：

$$X_{max} \approx \min\Big(\frac{P}{4Tc+2Ta+3Tm},\ \frac{m\_add}{2Ta},\ \frac{m\_mul}{3Tm}\Big)$$

> 注意：如果数据源 `produce()` 或 consumer 有额外 sleep/IO/重日志打印，也会形成新的上界项（例如单生产线程 + `sleep_for(2.5ms)` 大约是 400 pps 上界）。


